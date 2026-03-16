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


def split_into_clauses(text: str) -> list[str]:
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

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    clauses = [part.strip() for part in raw_parts if part.strip()]
    return clauses


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
    clauses = split_into_clauses(text)
    if not clauses:
        return []

    predictions = predict_clauses(
        clauses,
        max_length=max_length,
        batch_size=batch_size,
    )

    results: list[dict[str, Any]] = []
    for clause, prediction in zip(clauses, predictions):
        results.append(
            {
                "sentence": clause,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            }
        )

    return results