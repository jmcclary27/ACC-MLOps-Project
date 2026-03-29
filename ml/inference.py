# inference.py
from __future__ import annotations

from typing import Any

from api.model_loader import predict_clauses
from ml.data.text_helpers import chunk_legal_text_with_offsets


def merge_adjacent_predictions(
    results: list[dict[str, Any]],
    merge_labels: set[str] | None = None,
    max_gap_chars: int = 5,
) -> list[dict[str, Any]]:
    """
    Merge adjacent chunks with the same predicted label.

    Args:
        results: prediction results sorted by document order
        merge_labels: optional allowlist of labels that may be merged.
            If None, all labels may be merged.
        max_gap_chars: maximum allowed gap between chunks for merge

    Returns:
        Merged results list
    """
    if not results:
        return []

    merged: list[dict[str, Any]] = [results[0].copy()]

    for current in results[1:]:
        previous = merged[-1]

        same_label = current["label"] == previous["label"]
        label_allowed = merge_labels is None or current["label"] in merge_labels
        gap = int(current["start_char"]) - int(previous["end_char"])
        near_enough = gap <= max_gap_chars

        if same_label and label_allowed and near_enough:
            previous["text"] = f"{previous['text']} {current['text']}".strip()
            previous["end_char"] = current["end_char"]
            previous["confidence"] = (
                float(previous["confidence"]) + float(current["confidence"])
            ) / 2.0
        else:
            merged.append(current.copy())

    return merged


def predict_document(
    text: str,
    max_length: int = 512,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """
    Run inference over a full document using legal-aware chunking.

    Returns:
        A list of dictionaries containing:
        - text
        - label
        - confidence
        - start_char
        - end_char
    """
    chunks = chunk_legal_text_with_offsets(text)
    if not chunks:
        return []

    chunk_texts = [str(chunk["text"]) for chunk in chunks]

    predictions = predict_clauses(
        chunk_texts,
        max_length=max_length,
        batch_size=batch_size,
    )

    results: list[dict[str, Any]] = []
    for chunk, prediction in zip(chunks, predictions):
        results.append(
            {
                "text": chunk["text"],
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
            }
        )

    # Merge clause-like adjacent predictions
    results = merge_adjacent_predictions(results)

    return results