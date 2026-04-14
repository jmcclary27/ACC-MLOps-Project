from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.retrieval.search_faiss import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_BASE_DIR,
    DEFAULT_FAISS_BASE_DIR,
    SearchResult,
    load_embedding_model,
    load_faiss_index,
    load_metadata,
    resolve_embeddings_dir,
    resolve_faiss_dir,
    search_index,
)


# -------------------------------------------------------------------
# Paths / Config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -------------------------------------------------------------------
# Reranked result type
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RerankedSearchResult:
    rank: int
    original_rank: int
    faiss_score: float
    rerank_score: float
    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    embedding_model: str
    embedding_dim: int
    keyword_overlap: float
    heading_boost: float
    length_adjustment: float


# -------------------------------------------------------------------
# Text helpers
# -------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "by", "with",
    "under", "at", "from", "into", "this", "that", "these", "those", "is", "are",
    "be", "as", "it", "its", "any", "all", "may", "shall", "will", "not", "if",
    "than", "other", "such", "their", "there", "here", "which", "who", "whom",
    "what", "when", "where", "how", "why", "does", "do", "did", "about",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def is_heading_like(text: str) -> bool:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line:
        return False

    patterns = [
        r"^\d+\.\s+[A-Z]",
        r"^(section|clause|schedule|article)\s+\d+",
        r"^[A-Z][A-Za-z '&/\-]+$",
    ]
    return any(re.match(p, first_line, flags=re.IGNORECASE) for p in patterns)


def has_numbered_clause_prefix(text: str) -> bool:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return bool(re.match(r"^\d+(\.\d+)*\s+", first_line))


# -------------------------------------------------------------------
# Scoring helpers
# -------------------------------------------------------------------

def keyword_overlap_score(query: str, chunk_text: str) -> float:
    q_tokens = set(content_tokens(query))
    c_tokens = set(content_tokens(chunk_text))

    if not q_tokens or not c_tokens:
        return 0.0

    overlap = q_tokens.intersection(c_tokens)
    return len(overlap) / len(q_tokens)


def heading_boost_score(chunk_text: str) -> float:
    score = 0.0

    if is_heading_like(chunk_text):
        score += 0.10

    if has_numbered_clause_prefix(chunk_text):
        score += 0.05

    return score


def length_adjustment_score(chunk_text: str) -> float:
    length = len(chunk_text)

    # very short chunks are often less informative
    if length < 60:
        return -0.08

    # mild preference for clause-sized chunks
    if 80 <= length <= 260:
        return 0.04

    # very long chunks are a bit less precise
    if length > 420:
        return -0.04

    return 0.0


def compute_rerank_score(query: str, result: SearchResult) -> tuple[float, float, float, float]:
    kw_score = keyword_overlap_score(query, result.text)
    heading_score = heading_boost_score(result.text)
    length_score = length_adjustment_score(result.text)

    # weighted blend
    rerank_score = (
        0.75 * float(result.score)
        + 0.20 * kw_score
        + heading_score
        + length_score
    )

    return rerank_score, kw_score, heading_score, length_score


# -------------------------------------------------------------------
# Main reranking logic
# -------------------------------------------------------------------

def rerank_results(
    query: str,
    initial_results: list[SearchResult],
    top_k: int,
) -> list[RerankedSearchResult]:
    rescored: list[RerankedSearchResult] = []

    for original_rank, result in enumerate(initial_results, start=1):
        rerank_score, kw_score, heading_score, length_score = compute_rerank_score(
            query=query,
            result=result,
        )

        rescored.append(
            RerankedSearchResult(
                rank=0,  # filled after sorting
                original_rank=original_rank,
                faiss_score=float(result.score),
                rerank_score=float(rerank_score),
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                source_path=result.source_path,
                chunk_index=result.chunk_index,
                start_char=result.start_char,
                end_char=result.end_char,
                text=result.text,
                embedding_model=result.embedding_model,
                embedding_dim=result.embedding_dim,
                keyword_overlap=float(kw_score),
                heading_boost=float(heading_score),
                length_adjustment=float(length_score),
            )
        )

    rescored.sort(key=lambda x: x.rerank_score, reverse=True)

    final_results: list[RerankedSearchResult] = []
    for new_rank, row in enumerate(rescored[:top_k], start=1):
        final_results.append(
            RerankedSearchResult(
                rank=new_rank,
                original_rank=row.original_rank,
                faiss_score=row.faiss_score,
                rerank_score=row.rerank_score,
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                source_path=row.source_path,
                chunk_index=row.chunk_index,
                start_char=row.start_char,
                end_char=row.end_char,
                text=row.text,
                embedding_model=row.embedding_model,
                embedding_dim=row.embedding_dim,
                keyword_overlap=row.keyword_overlap,
                heading_boost=row.heading_boost,
                length_adjustment=row.length_adjustment,
            )
        )

    return final_results


def search_and_rerank(
    query: str,
    faiss_input_path: Path,
    embeddings_input_path: Path,
    embedding_model_name: str,
    candidate_k: int,
    top_k: int,
) -> list[RerankedSearchResult]:
    faiss_dir = resolve_faiss_dir(faiss_input_path)
    embeddings_dir = resolve_embeddings_dir(embeddings_input_path)

    logger.info("Using FAISS dir: %s", faiss_dir)
    logger.info("Using embeddings dir: %s", embeddings_dir)

    index = load_faiss_index(faiss_dir)
    metadata_rows = load_metadata(embeddings_dir)
    model = load_embedding_model(embedding_model_name)

    if index.ntotal != len(metadata_rows):
        raise RuntimeError(
            f"FAISS index vector count ({index.ntotal}) does not match "
            f"metadata row count ({len(metadata_rows)})."
        )

    initial_results = search_index(
        query=query,
        index=index,
        metadata_rows=metadata_rows,
        model=model,
        top_k=max(candidate_k, top_k),
    )

    return rerank_results(
        query=query,
        initial_results=initial_results,
        top_k=top_k,
    )


# -------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------

def print_results(query: str, results: list[RerankedSearchResult]) -> None:
    print(f"\nQuery: {query}")
    print(f"Top reranked results: {len(results)}")

    for result in results:
        print("\n" + "=" * 80)
        print(f"Rank: {result.rank}")
        print(f"Original Rank: {result.original_rank}")
        print(f"FAISS Score: {result.faiss_score:.6f}")
        print(f"Rerank Score: {result.rerank_score:.6f}")
        print(f"Keyword Overlap: {result.keyword_overlap:.6f}")
        print(f"Heading Boost: {result.heading_boost:.6f}")
        print(f"Length Adjustment: {result.length_adjustment:.6f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Doc ID: {result.doc_id}")
        print(f"Source: {result.source_path}")
        print(f"Chunk Index: {result.chunk_index}")
        print(f"Offsets: {result.start_char} -> {result.end_char}")
        print("Text:")
        print(result.text)


def maybe_save_results_json(path: Path | None, query: str, results: list[RerankedSearchResult]) -> None:
    if path is None:
        return

    payload = {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "rank": r.rank,
                "original_rank": r.original_rank,
                "faiss_score": r.faiss_score,
                "rerank_score": r.rerank_score,
                "keyword_overlap": r.keyword_overlap,
                "heading_boost": r.heading_boost,
                "length_adjustment": r.length_adjustment,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "source_path": r.source_path,
                "chunk_index": r.chunk_index,
                "start_char": r.start_char,
                "end_char": r.end_char,
                "text": r.text,
                "embedding_model": r.embedding_model,
                "embedding_dim": r.embedding_dim,
            }
            for r in results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved reranked results to: %s", path)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a FAISS index and rerank the top candidates."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Final number of reranked results to return.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of FAISS candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--faiss-input-path",
        type=Path,
        default=DEFAULT_FAISS_BASE_DIR,
        help="Path to a specific FAISS run directory, or the base FAISS directory.",
    )
    parser.add_argument(
        "--embeddings-input-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_BASE_DIR,
        help="Path to a specific embeddings run directory, or the base embeddings directory.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used to encode the query.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save reranked results as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k")

    results = search_and_rerank(
        query=args.query,
        faiss_input_path=args.faiss_input_path,
        embeddings_input_path=args.embeddings_input_path,
        embedding_model_name=args.embedding_model,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )

    print_results(args.query, results)
    maybe_save_results_json(args.output_json, args.query, results)


if __name__ == "__main__":
    main()