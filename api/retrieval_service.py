from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from api.document_store import StoredDocument

logger = logging.getLogger(__name__)

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    start_char: int
    end_char: int
    similarity_score: float
    rerank_score: float


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", _EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross encoder model: %s", _CROSS_ENCODER_MODEL_NAME)
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def embed_chunks(chunk_texts: list[str]) -> np.ndarray:
    if not chunk_texts:
        return np.empty((0, 384), dtype=np.float32)

    embedder = get_embedder()
    vectors = embedder.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    embedder = get_embedder()
    vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype(np.float32)[0]


def retrieve_top_chunks(
    document: StoredDocument,
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if document.embeddings.size == 0 or not document.chunks:
        return []

    query_vector = embed_query(query)
    similarities = np.dot(document.embeddings, query_vector)

    candidate_k = min(candidate_k, len(document.chunks))
    top_indices = np.argsort(similarities)[::-1][:candidate_k]

    candidates: list[RetrievedChunk] = []
    for idx in top_indices:
        chunk = document.chunks[int(idx)]
        candidates.append(
            RetrievedChunk(
                chunk_id=str(chunk["chunk_id"]),
                doc_id=document.document_id,
                source_path=document.filename,
                text=str(chunk["text"]),
                start_char=int(chunk["start_char"]),
                end_char=int(chunk["end_char"]),
                similarity_score=float(similarities[idx]),
                rerank_score=0.0,
            )
        )

    if not candidates:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query, candidate.text) for candidate in candidates]
    rerank_scores = [float(x) for x in cross_encoder.predict(pairs)]

    reranked: list[RetrievedChunk] = []
    for candidate, rerank_score in zip(candidates, rerank_scores):
        reranked.append(
            RetrievedChunk(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                source_path=candidate.source_path,
                text=candidate.text,
                start_char=candidate.start_char,
                end_char=candidate.end_char,
                similarity_score=candidate.similarity_score,
                rerank_score=rerank_score,
            )
        )

    reranked.sort(key=lambda x: x.rerank_score, reverse=True)
    return reranked[: min(top_k, len(reranked))]