from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from api.document_store import StoredDocument
from api.retrieval_service import RetrievedChunk, retrieve_top_chunks

load_dotenv()

client = OpenAI()


def wants_multiple_results(question: str) -> bool:
    q = question.lower()
    multi_markers = [
        "examples",
        "multiple",
        "several",
        "list",
        "all",
        "compare",
        "clauses",
        "provisions",
    ]
    return any(marker in q for marker in multi_markers)


def filter_dynamic_citations(
    retrieved: list[RetrievedChunk],
    question: str,
) -> list[RetrievedChunk]:
    if not retrieved:
        return []

    max_citations = 5 if wants_multiple_results(question) else 1
    best_score = retrieved[0].rerank_score
    filtered: list[RetrievedChunk] = []

    for chunk in retrieved:
        if chunk.rerank_score < 0.15:
            continue

        if best_score > 0 and chunk.rerank_score < best_score * 0.80:
            continue

        filtered.append(chunk)

        if len(filtered) >= max_citations:
            break

    if filtered:
        return filtered

    return retrieved[:1]


def build_grounded_qa_prompt(question: str, chunks: list[dict[str, object]]) -> str:
    context_parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            (
                f"[Chunk {i}]\n"
                f"chunk_id: {chunk['chunk_id']}\n"
                f"start_char: {chunk['start_char']}\n"
                f"end_char: {chunk['end_char']}\n"
                f"text: {chunk['text']}"
            )
        )

    context = "\n\n".join(context_parts)

    return f"""
You are answering a question about an uploaded contract.

Use only the provided context chunks.
If the answer is not supported by the context, say that the document does not clearly provide the answer.

Answer concisely and directly.
Prefer the smallest set of facts needed to answer the question.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


def answer_question_over_document(
    document: StoredDocument,
    question: str,
    top_k: int = 5,
) -> dict[str, object]:
    retrieved = retrieve_top_chunks(
        document=document,
        query=question,
        top_k=max(top_k, 8),
        candidate_k=max(20, top_k),
    )

    if not retrieved:
        return {
            "answer": "I could not find relevant information in this document.",
            "citations": [],
        }

    llm_context = retrieved[: min(5, len(retrieved))]
    final_citations = filter_dynamic_citations(retrieved, question)

    context_dicts = [
        {
            "chunk_id": row.chunk_id,
            "doc_id": row.doc_id,
            "source_path": row.source_path,
            "start_char": row.start_char,
            "end_char": row.end_char,
            "text": row.text,
        }
        for row in llm_context
    ]

    citation_dicts = [
        {
            "chunk_id": row.chunk_id,
            "doc_id": row.doc_id,
            "source_path": row.source_path,
            "start_char": row.start_char,
            "end_char": row.end_char,
            "text": row.text,
        }
        for row in final_citations
    ]

    prompt = build_grounded_qa_prompt(question, context_dicts)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0,
    )

    answer = response.output_text.strip()
    if not answer:
        answer = "I could not generate an answer from this document."

    return {
        "answer": answer,
        "citations": citation_dicts,
    }