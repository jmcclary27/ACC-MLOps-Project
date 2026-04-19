# ml/rag/answer_question.py
from __future__ import annotations

from pathlib import Path

from ml.rag.prompting import build_grounded_prompt
from ml.rag.schemas import Citation, RAGResponse
from ml.retrieval.search_faiss_cross_rerank import search_and_cross_rerank

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()


def call_api_llm(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0,
    )

    return response.output_text.strip()


def run_rag(
    question: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> RAGResponse:
    results = search_and_cross_rerank(
        query=question,
        faiss_input_path=Path("ml/data/processed/faiss"),
        embeddings_input_path=Path("ml/data/processed/embeddings"),
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k=candidate_k,
        top_k=top_k,
    )

    if not results:
        return RAGResponse(answer="not found", citations=[])

    chunks = [
        Citation(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            source_path=r.source_path,
            start_char=r.start_char,
            end_char=r.end_char,
            text=r.text,
        )
        for r in results
    ]

    prompt = build_grounded_prompt(
        question,
        [chunk.model_dump() for chunk in chunks],
    )

    answer = call_api_llm(prompt).strip()

    return RAGResponse(
        answer=answer if answer else "not found",
        citations=chunks,
    )