"""
Document-level inference utilities for contract clause classification.

This module:
- splits contract text into clause-like chunks
- runs batch inference through api.model_loader
- returns structured per-clause predictions
"""

from __future__ import annotations

import re
from typing import Any

from api.model_loader import predict_clauses


def _append_chunk_with_offsets(
    text: str,
    chunks: list[dict[str, int | str]],
    start: int,
    end: int,
) -> None:
    """Trim boundary whitespace while preserving original character offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    if start < end:
        chunks.append(
            {
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
            }
        )


def split_into_clauses(text: str) -> list[dict[str, int | str]]:
    """
    Split contract text into simple clause-like segments.

    Current strategy:
    - split on sentence-ending punctuation followed by whitespace
    - split on one or more newlines
    - remove empty chunks

    Later, this can be replaced with smarter legal clause segmentation.
    """
    if not text or not text.strip():
        return []

    separator_pattern = re.compile(r"(?<=[.!?])\s+|\n+")
    chunks: list[dict[str, int | str]] = []

    cursor = 0
    for match in separator_pattern.finditer(text):
        _append_chunk_with_offsets(text, chunks, cursor, match.start())
        cursor = match.end()

    _append_chunk_with_offsets(text, chunks, cursor, len(text))
    return chunks


def predict_document(
    text: str,
    max_length: int = 512,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """
    Run inference over a full document.

    Args:
        text: Full document text
        max_length: Max token length passed to tokenizer
        batch_size: Batch size for model inference

    Returns:
        A list of dictionaries containing:
        - sentence
        - label
        - confidence
    """
    clause_chunks = split_into_clauses(text)
    if not clause_chunks:
        return []

    clause_texts = [str(chunk["text"]) for chunk in clause_chunks]

    predictions = predict_clauses(
        clause_texts,
        max_length=max_length,
        batch_size=batch_size,
    )

    results: list[dict[str, Any]] = []
    for chunk, prediction in zip(clause_chunks, predictions):
        results.append(
            {
                "text": chunk["text"],
                # Keep sentence for backward compatibility with existing frontend paths.
                "sentence": chunk["text"],
                "start_char": int(chunk["start_char"]),
                "end_char": int(chunk["end_char"]),
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            }
        )

    return results