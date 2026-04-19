from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def build_summary_prompt(document_text: str) -> str:
    trimmed_text = document_text[:16000]

    return f"""
You are summarizing an uploaded contract for a user interface.

Using only the document text, produce:
1. one sentence identifying the agreement
2. one sentence stating the most important business/legal points
3. one optional sentence only if there is an especially important term like termination, payment, governing law, arbitration, or confidentiality

Keep the total summary under 120 words.

DOCUMENT:
\"\"\"
{trimmed_text}
\"\"\"
""".strip()


def summarize_document(document_text: str) -> str:
    if not document_text.strip():
        return "No summary available."

    prompt = build_summary_prompt(document_text)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0,
    )

    summary = response.output_text.strip()
    return summary if summary else "No summary available."