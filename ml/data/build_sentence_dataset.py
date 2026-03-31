# build_sentence_dataset.py
from __future__ import annotations

import csv
import json
from pathlib import Path

from ml.data.text_helpers import chunk_legal_text_with_offsets

BASE_DIR = Path(__file__).resolve().parent

CUAD_JSON_PATH = BASE_DIR / "raw" / "CUADv1.json"
OUTPUT_PATH = BASE_DIR / "processed" / "clean_chunk_dataset.csv"

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

OTHER_LABEL = "Other"
MIN_OVERLAP_RATIO = 0.5


def overlap_length(span1_start: int, span1_end: int, span2_start: int, span2_end: int) -> int:
    return max(0, min(span1_end, span2_end) - max(span1_start, span2_start))


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

        selected_answers: list[dict[str, int | str]] = []

        for qa in paragraph["qas"]:
            label = qa["id"].rsplit("__", 1)[-1].strip()

            if label not in SELECTED_LABELS:
                continue

            for answer in qa.get("answers", []):
                answer_text = answer["text"]
                answer_start = int(answer["answer_start"])
                answer_end = answer_start + len(answer_text)

                selected_answers.append(
                    {
                        "label": label,
                        "start": answer_start,
                        "end": answer_end,
                    }
                )

        for chunk in chunks:
            chunk_text = str(chunk["text"]).strip()
            chunk_start = int(chunk["start_char"])
            chunk_end = int(chunk["end_char"])

            if not chunk_text:
                continue

            best_label = OTHER_LABEL
            best_overlap = 0.0

            for answer in selected_answers:
                answer_start = int(answer["start"])
                answer_end = int(answer["end"])
                answer_len = max(1, answer_end - answer_start)

                olap = overlap_length(chunk_start, chunk_end, answer_start, answer_end)
                overlap_ratio = olap / answer_len

                if overlap_ratio >= MIN_OVERLAP_RATIO and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_label = str(answer["label"])

            rows.append(
                {
                    "text": chunk_text,
                    "clause": best_label,
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

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "clause", "contract", "start_char", "end_char"])

    for text, clause, contract, start_char, end_char in sorted(unique_rows):
        writer.writerow([text, clause, contract, start_char, end_char])

print(f"Saved {len(unique_rows)} chunk rows to {OUTPUT_PATH}")