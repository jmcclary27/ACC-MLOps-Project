from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    status: str
    extracted_text: str
    summary: str
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray


_DOCUMENTS: dict[str, StoredDocument] = {}


def save_document(document: StoredDocument) -> None:
    _DOCUMENTS[document.document_id] = document


def get_document(document_id: str) -> StoredDocument | None:
    return _DOCUMENTS.get(document_id)


def document_exists(document_id: str) -> bool:
    return document_id in _DOCUMENTS


def list_documents() -> list[StoredDocument]:
    return list(_DOCUMENTS.values())