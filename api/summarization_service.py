from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def build_summary_prompt(document_text: str) -> str:
    trimmed_text = document_text[:10000]

    return f"""
You are generating a very short summary for a contract viewer UI.

Using only the document text, write exactly 1 short paragraph of no more than 2 sentences.

Your goal is only to say:
- what kind of agreement this is
- who the parties are, if clear
- what the agreement is generally about

Do not list specific clauses unless absolutely necessary.
Do not mention many details.
Do not use bullet points.
Do not use numbering.
Do not use labels.
Maximum 40 words.

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
        max_output_tokens=60,
    )

    summary = response.output_text.strip()

    if not summary:
        return "No summary available."

    summary = " ".join(summary.split())

    for prefix in ("1. ", "2. ", "3. ", "- ", "* ", "Summary: "):
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()

    return summary