"""
Model Loader Module

This module is responsible for loading the trained model and
making predictions. It currently uses mock logic, but the structure
is ready to be replaced with real Hugging Face inference later.
"""

from __future__ import annotations

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model: Any = None
_tokenizer: Any = None


def load_model() -> None:
    """
    Load the model and tokenizer into memory once.

    Later, replace this with actual Hugging Face loading logic, for example:
        AutoTokenizer.from_pretrained(...)
        AutoModelForSequenceClassification.from_pretrained(...)
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return

    logger.info("Loading model and tokenizer...")
    _model = "mock_model"
    _tokenizer = "mock_tokenizer"
    logger.info("Model loaded successfully.")


def predict_clause(text: str) -> dict[str, float | str]:
    """
    Predict the class of a single clause.

    Args:
        text: Clause text

    Returns:
        Dictionary with label and confidence
    """
    if _model is None or _tokenizer is None:
        load_model()

    clean_text = text.strip()
    logger.info("Making prediction for text: '%s...'", clean_text[:50])

    # Temporary mock logic
    # Replace this with real tokenization + model inference later
    if "terminate" in clean_text.lower():
        return {
            "label": "termination",
            "confidence": 0.94,
        }

    if "confidential" in clean_text.lower() or "non-disclosure" in clean_text.lower():
        return {
            "label": "confidentiality",
            "confidence": 0.96,
        }

    if "pay" in clean_text.lower() or "payment" in clean_text.lower():
        return {
            "label": "payment",
            "confidence": 0.93,
        }

    return {
        "label": "acceptable",
        "confidence": 0.91,
    }