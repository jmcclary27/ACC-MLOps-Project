from __future__ import annotations


def build_grounded_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id", "unknown_chunk")
        text = chunk.get("text", "").strip()
        context_blocks.append(f"[Chunk ID: {chunk_id}]\n{text}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are a contract question-answering assistant.

Answer the user's question using only the provided context.
Do not use outside knowledge.
Do not guess.
If the context does not contain enough evidence to answer the question, reply exactly:

not found

When you answer, rely only on the retrieved chunks.

Question:
{question}

Context:
{context_text}
""".strip()

    return prompt