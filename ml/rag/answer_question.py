from __future__ import annotations

from typing import Any

from ml.rag.prompting import build_grounded_prompt
from ml.rag.schemas import Citation, RAGResponse


def call_api_llm(prompt: str) -> str:
    """
    Replace this with your actual API LLM call.
    It should return only the assistant's text response.
    """
    raise NotImplementedError("Hook up your API LLM here.")


def answer_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> RAGResponse:
    prompt = build_grounded_prompt(question, retrieved_chunks)
    answer_text = call_api_llm(prompt).strip()

    citations = [
        Citation(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            source_path=chunk["source_path"],
            start_char=chunk["start_char"],
            end_char=chunk["end_char"],
            text=chunk["text"],
        )
        for chunk in retrieved_chunks
    ]

    return RAGResponse(
        answer=answer_text,
        citations=citations,
    )