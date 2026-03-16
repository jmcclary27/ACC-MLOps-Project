"""
Model loader and inference utilities for the trained DistilBERT clause classifier.

This module:
- loads the trained Hugging Face model once
- loads the tokenizer once
- performs single-text and batch inference
- returns labels using the model config's id2label mapping when available

Expected model path:
    ml/models/distilbert_clause_classifier
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("ml/models/distilbert_clause_classifier")

_model: AutoModelForSequenceClassification | None = None
_tokenizer: Any | None = None
_device: torch.device | None = None
_id2label: dict[int, str] | None = None


def _get_device() -> torch.device:
    """
    Choose the best available device for inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> None:
    """
    Load the tokenizer and model into memory once.

    Raises:
        FileNotFoundError: if the model directory does not exist
        RuntimeError: if the model/tokenizer cannot be loaded
    """
    global _model, _tokenizer, _device, _id2label

    if _model is not None and _tokenizer is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR.resolve()}"
        )

    logger.info("Loading tokenizer from %s", MODEL_DIR.resolve())
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    logger.info("Loading model from %s", MODEL_DIR.resolve())
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    _device = _get_device()
    _model.to(_device)
    _model.eval()

    raw_id2label = getattr(_model.config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        # Hugging Face sometimes stores keys as ints, but we normalize just in case
        _id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    else:
        _id2label = None

    logger.info("Model loaded successfully on device: %s", _device)


def _require_loaded() -> tuple[
    AutoModelForSequenceClassification,
    Any,
    torch.device,
]:
    """
    Ensure model assets are loaded and return them.
    """
    if _model is None or _tokenizer is None or _device is None:
        load_model()

    if _model is None or _tokenizer is None or _device is None:
        raise RuntimeError("Model failed to load correctly.")

    return _model, _tokenizer, _device


def _label_from_index(class_idx: int) -> str:
    """
    Convert a predicted class index into a readable label.
    """
    if _id2label is not None and class_idx in _id2label:
        return _id2label[class_idx]

    return f"class_{class_idx}"


def predict_clause(text: str, max_length: int = 512) -> dict[str, float | str]:
    """
    Predict the class for a single clause.

    Args:
        text: Input text to classify
        max_length: Max token length for tokenizer truncation

    Returns:
        A dictionary with:
        - label
        - confidence
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot predict on empty text.")

    model, tokenizer, device = _require_loaded()

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence_tensor, pred_idx_tensor = torch.max(probs, dim=1)

    pred_idx = int(pred_idx_tensor.item())
    confidence = float(confidence_tensor.item())
    label = _label_from_index(pred_idx)

    return {
        "label": label,
        "confidence": confidence,
    }


def predict_clauses(
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 16,
) -> list[dict[str, float | str]]:
    """
    Predict classes for multiple clauses in batches.

    Args:
        texts: List of clause strings
        max_length: Max token length for tokenizer truncation
        batch_size: Number of texts per inference batch

    Returns:
        A list of dictionaries, each containing:
        - label
        - confidence
    """
    if not texts:
        return []

    model, tokenizer, device = _require_loaded()

    cleaned_texts = [text.strip() for text in texts]
    nonempty_indices = [i for i, text in enumerate(cleaned_texts) if text]

    results: list[dict[str, float | str]] = [
        {"label": "empty", "confidence": 0.0} for _ in cleaned_texts
    ]

    if not nonempty_indices:
        return results

    nonempty_texts = [cleaned_texts[i] for i in nonempty_indices]

    for start in range(0, len(nonempty_texts), batch_size):
        batch_texts = nonempty_texts[start : start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidences, pred_indices = torch.max(probs, dim=1)

        for offset, (confidence_tensor, pred_idx_tensor) in enumerate(
            zip(confidences, pred_indices)
        ):
            original_idx = nonempty_indices[start + offset]
            pred_idx = int(pred_idx_tensor.item())
            confidence = float(confidence_tensor.item())
            label = _label_from_index(pred_idx)

            results[original_idx] = {
                "label": label,
                "confidence": confidence,
            }

    return results