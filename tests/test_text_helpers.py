from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ml.data.text_helpers import pdf_to_text, split_into_sentence, clean_text
# If your file actually lives elsewhere, use the correct import, for example:
# from ml.data.text_helpers import pdf_to_text, split_into_sentence, clean_text


SAMPLE_PDF = Path(__file__).resolve().parent / "sample.pdf"


def test_pdf_to_text() -> None:
    text = pdf_to_text(str(SAMPLE_PDF), as_text=True)

    assert isinstance(text, str)
    assert len(text.strip()) > 0

    # Avoid exact full-string equality for PDF extraction, which can vary slightly
    assert "Hello, this is a test PDF." in text
    assert "This agreement shall terminate upon breach." in text
    assert "The tenant must pay a rent of $500 on the first of each month." in text


def test_split_into_sentence() -> None:
    text = pdf_to_text(str(SAMPLE_PDF), as_text=True)
    sentences = split_into_sentence(text)

    expected = [
        "Hello, this is a test PDF.",
        "This agreement shall terminate upon breach.",
        "The tenant must pay a rent of $500 on the first of each month.",
    ]

    assert sentences == expected


def test_clean_text() -> None:
    text = pdf_to_text(str(SAMPLE_PDF), as_text=True)
    cleaned_text = clean_text(text)

    expected = (
        "Hello, this is a test PDF. "
        "This agreement shall terminate upon breach. "
        "The tenant must pay a rent of $500 on the first of each month."
    )

    assert cleaned_text == expected