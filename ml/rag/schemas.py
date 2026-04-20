from pydantic import BaseModel
from typing import List


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    start_char: int
    end_char: int
    text: str


class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]