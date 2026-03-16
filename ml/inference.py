"""
Inference utilities for contract clause classification.

This module provides a simple inference pipeline:
1. Split incoming contract text into sentence-like chunks
2. Run each chunk through the model loader
3. Return structured predictions

For now, this works with the mock model_loader implementation.
Later, you can replace model_loader.predict_clause() with real Hugging Face inference.
"""

from __future__ import annotations

import re
from typing import Any

from api.model_loader import predict_clause


def split_into_clauses(text: str) -> list[str]:
    """
    Split a contract into simple clause-like segments.

    Right now this uses a lightweight regex split on punctuation and newlines.
    Later, you can replace this with spaCy sentence segmentation or legal clause parsing.

    Args:
        text: Full contract text

    Returns:
        List of cleaned clause/sentence strings
    """
    if not text or not text.strip():
        return []

    # Split on:
    # - sentence punctuation followed by whitespace
    # - blank lines
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    clauses = [part.strip() for part in raw_parts if part.strip()]
    return clauses


def predict_document(text: str) -> list[dict[str, Any]]:
    """
    Run clause prediction across an entire document.

    Args:
        text: Full contract text

    Returns:
        A list of dictionaries containing:
        - sentence
        - label
        - confidence
    """
    clauses = split_into_clauses(text)

    results: list[dict[str, Any]] = []
    for clause in clauses:
        prediction = predict_clause(clause)
        results.append(
            {
                "sentence": clause,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            }
        )

    return results