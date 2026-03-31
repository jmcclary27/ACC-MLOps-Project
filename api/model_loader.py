from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.model_registry import resolve_model_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR: Path | None = None

_model: AutoModelForSequenceClassification | None = None
_tokenizer: Any | None = None
_device: torch.device | None = None
_id2label: dict[int, str] | None = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolved_model_dir() -> Path:
    global MODEL_DIR

    if MODEL_DIR is None:
        MODEL_DIR = resolve_model_dir()

    return MODEL_DIR


def reset_loaded_model() -> None:
    global _model, _tokenizer, _device, _id2label, MODEL_DIR
    _model = None
    _tokenizer = None
    _device = None
    _id2label = None
    MODEL_DIR = None


def load_model() -> None:
    global _model, _tokenizer, _device, _id2label

    if _model is not None and _tokenizer is not None:
        return

    model_dir = _resolved_model_dir()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    logger.info("Loading tokenizer from %s", model_dir.resolve())
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)

    logger.info("Loading model from %s", model_dir.resolve())
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    _device = _get_device()
    _model.to(_device)
    _model.eval()

    raw_id2label = getattr(_model.config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        _id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    else:
        _id2label = None

    logger.info("Model loaded successfully on device: %s", _device)
    logger.info("Resolved model directory: %s", model_dir.resolve())


def _require_loaded() -> tuple[
    AutoModelForSequenceClassification,
    Any,
    torch.device,
]:
    if _model is None or _tokenizer is None or _device is None:
        load_model()

    if _model is None or _tokenizer is None or _device is None:
        raise RuntimeError("Model failed to load correctly.")

    return _model, _tokenizer, _device


def _label_from_index(class_idx: int) -> str:
    if _id2label is not None and class_idx in _id2label:
        return _id2label[class_idx]

    return f"class_{class_idx}"


def predict_clause(text: str, max_length: int = 512) -> dict[str, float | str]:
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