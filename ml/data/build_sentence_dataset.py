# build_sentence_dataset.py
from __future__ import annotations

import csv
import json
from pathlib import Path

from ml.text_helpers import chunk_legal_text_with_offsets

CUAD_JSON_PATH = Path("raw/CUADv1.json")
OUTPUT_PATH = Path("processed/clean_chunk_dataset.csv")

SELECTED_LABELS = {
    "Governing Law",
    "Termination For Convenience",
    "Exclusivity",
    "Non-Compete",
    "No-Solicit Of Employees",
    "IP Ownership Assignment",
    "Cap on Liability",
    "Change of Control",
}


def overlaps(span1_start: int, span1_end: int, span2_start: int, span2_end: int) -> bool:
    return not (span1_end <= span2_start or span1_start >= span2_end)


with open(CUAD_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows: list[dict[str, str]] = []

for contract in data["data"]:
    contract_title = contract.get("title", "unknown_contract")

    for paragraph in contract["paragraphs"]:
        context = paragraph["context"]
        chunks = chunk_legal_text_with_offsets(context)

        if not chunks:
            continue

        for qa in paragraph["qas"]:
            label = qa["id"].rsplit("__", 1)[-1].strip()

            if label not in SELECTED_LABELS:
                continue

            for answer in qa.get("answers", []):
                answer_text = answer["text"]
                answer_start = answer["answer_start"]
                answer_end = answer_start + len(answer_text)

                for chunk in chunks:
                    chunk_start = int(chunk["start_char"])
                    chunk_end = int(chunk["end_char"])

                    if overlaps(chunk_start, chunk_end, answer_start, answer_end):
                        rows.append(
                            {
                                "text": str(chunk["text"]),
                                "clause": label,
                                "contract": contract_title,
                                "start_char": str(chunk_start),
                                "end_char": str(chunk_end),
                            }
                        )

# Remove duplicates
unique_rows = {
    (
        row["text"],
        row["clause"],
        row["contract"],
        row["start_char"],
        row["end_char"],
    )
    for row in rows
}

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "clause", "contract", "start_char", "end_char"])

    for text, clause, contract, start_char, end_char in sorted(unique_rows):
        writer.writerow([text, clause, contract, start_char, end_char])

print(f"Saved {len(unique_rows)} labeled chunks to {OUTPUT_PATH}")