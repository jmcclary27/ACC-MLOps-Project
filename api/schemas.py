from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    filename: str
    text_length: int
    chunk_count: int
    summary: str
    extracted_text: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    filename: str
    text_length: int
    chunk_count: int
    summary: str
    extracted_text: str


class SearchRequest(BaseModel):
    document_id: str
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultResponse(BaseModel):
    chunk_id: str
    score: float
    text: str
    start_char: int
    end_char: int


class SearchResponse(BaseModel):
    document_id: str
    query: str
    results: List[SearchResultResponse]


class QARequest(BaseModel):
    document_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    start_char: int
    end_char: int
    text: str


class QAResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]