import json
import csv
import re
from pathlib import Path

CUAD_JSON_PATH = Path("raw/CUADv1.json")
OUTPUT_PATH = Path("processed/clean_sentence_dataset.csv")

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

# Simple sentence splitter with offsets
def split_into_sentences_with_offsets(text):
    sentences = []
    pattern = re.compile(r'[^.!?]+[.!?]?')
    for match in pattern.finditer(text):
        sentence = match.group().strip()
        if sentence:
            sentences.append({
                "text": sentence,
                "start": match.start(),
                "end": match.end()
            })
    return sentences

# Load CUAD
with open(CUAD_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

# Parse contracts
for contract in data["data"]:
    for paragraph in contract["paragraphs"]:
        context = paragraph["context"]
        sentences = split_into_sentences_with_offsets(context)

        for qa in paragraph["qas"]:
            label = qa["id"].rsplit("__", 1)[-1].strip()

            if label not in SELECTED_LABELS:
                continue

            for ans in qa.get("answers", []):
                answer_start = ans["answer_start"]
                answer_end = answer_start + len(ans["text"])

                for sentence in sentences:
                    # Check overlap
                    if not (
                        sentence["end"] <= answer_start or
                        sentence["start"] >= answer_end
                    ):
                        rows.append({
                            "text": sentence["text"],
                            "clause": label
                        })

# Remove duplicates
unique_rows = {(r["text"], r["clause"]) for r in rows}

# Write CSV
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "clause"])
    for text, clause in unique_rows:
        writer.writerow([text, clause])

print(f"Saved {len(unique_rows)} labeled sentences to {OUTPUT_PATH}")