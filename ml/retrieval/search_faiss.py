from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required for this script. Install faiss-cpu first."
    ) from exc

from sentence_transformers import SentenceTransformer


# -------------------------------------------------------------------
# Paths / Config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FAISS_BASE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "faiss"
DEFAULT_EMBEDDINGS_BASE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "embeddings"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    embedding_model: str
    embedding_dim: int


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_latest_run_dir(base_dir: Path, required_files: list[str]) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base_dir}")

    valid_candidates: list[Path] = []
    for candidate in candidates:
        if all((candidate / fname).exists() for fname in required_files):
            valid_candidates.append(candidate)

    if not valid_candidates:
        raise FileNotFoundError(
            f"No valid run directories found under: {base_dir} "
            f"with required files: {required_files}"
        )

    valid_candidates.sort(key=lambda p: p.name)
    return valid_candidates[-1]


def resolve_faiss_dir(input_path: Path) -> Path:
    required = ["faiss.index", "manifest.json", "config.json"]

    if input_path.is_dir() and all((input_path / fname).exists() for fname in required):
        return input_path

    return resolve_latest_run_dir(input_path, required)


def resolve_embeddings_dir(input_path: Path) -> Path:
    required = ["chunk_embeddings.npy", "chunk_metadata.jsonl", "config.json"]

    if input_path.is_dir() and all((input_path / fname).exists() for fname in required):
        return input_path

    return resolve_latest_run_dir(input_path, required)


def load_faiss_index(faiss_dir: Path) -> faiss.Index:
    index_path = faiss_dir / "faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_metadata(embeddings_dir: Path) -> list[dict[str, Any]]:
    metadata_path = embeddings_dir / "chunk_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")
    return read_jsonl(metadata_path)


def load_embedding_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype(np.float32)


# -------------------------------------------------------------------
# Search logic
# -------------------------------------------------------------------

def search_index(
    query: str,
    index: faiss.Index,
    metadata_rows: list[dict[str, Any]],
    model: SentenceTransformer,
    top_k: int,
) -> list[SearchResult]:
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if not metadata_rows:
        raise ValueError("Metadata is empty.")

    query_vector = embed_query(model, query)
    distances, indices = index.search(query_vector, top_k)

    results: list[SearchResult] = []

    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0 or idx >= len(metadata_rows):
            continue

        row = metadata_rows[idx]
        results.append(
            SearchResult(
                rank=rank,
                score=float(score),
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                source_path=str(row["source_path"]),
                chunk_index=int(row["chunk_index"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                text=str(row["text"]),
                embedding_model=str(row["embedding_model"]),
                embedding_dim=int(row["embedding_dim"]),
            )
        )

    return results


# -------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------

def print_results(query: str, results: list[SearchResult]) -> None:
    print(f"\nQuery: {query}")
    print(f"Top results: {len(results)}")

    for result in results:
        print("\n" + "=" * 80)
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Doc ID: {result.doc_id}")
        print(f"Source: {result.source_path}")
        print(f"Chunk Index: {result.chunk_index}")
        print(f"Offsets: {result.start_char} -> {result.end_char}")
        print("Text:")
        print(result.text)


def maybe_save_results_json(path: Path | None, query: str, results: list[SearchResult]) -> None:
    if path is None:
        return

    payload = {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "rank": r.rank,
                "score": r.score,
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

    logger.info("Saved results to: %s", path)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a FAISS index of contract chunks."
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
        help="Number of top results to return.",
    )
    parser.add_argument(
        "--faiss-input-path",
        type=Path,
        default=DEFAULT_FAISS_BASE_DIR,
        help=(
            "Path to a specific FAISS run directory, or the base FAISS directory "
            "containing versioned runs."
        ),
    )
    parser.add_argument(
        "--embeddings-input-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_BASE_DIR,
        help=(
            "Path to a specific embeddings run directory, or the base embeddings "
            "directory containing versioned runs."
        ),
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
        help="Optional path to save results as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    faiss_dir = resolve_faiss_dir(args.faiss_input_path)
    embeddings_dir = resolve_embeddings_dir(args.embeddings_input_path)

    logger.info("Using FAISS dir: %s", faiss_dir)
    logger.info("Using embeddings dir: %s", embeddings_dir)

    index = load_faiss_index(faiss_dir)
    metadata_rows = load_metadata(embeddings_dir)
    model = load_embedding_model(args.embedding_model)

    if index.ntotal != len(metadata_rows):
        raise RuntimeError(
            f"FAISS index vector count ({index.ntotal}) does not match "
            f"metadata row count ({len(metadata_rows)})."
        )

    results = search_index(
        query=args.query,
        index=index,
        metadata_rows=metadata_rows,
        model=model,
        top_k=args.top_k,
    )

    print_results(args.query, results)
    maybe_save_results_json(args.output_json, args.query, results)


if __name__ == "__main__":
    main()