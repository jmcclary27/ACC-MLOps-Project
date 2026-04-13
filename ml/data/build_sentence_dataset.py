# ml/data/build_sentence_dataset.py
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

RANDOM_SEED = 42
NEGATIVE_TO_POSITIVE_RATIO = 3.0
MAX_OTHER_PER_NO_LABEL_PARAGRAPH = 6

MIN_TEXT_LEN = 35
MAX_TEXT_LEN = 2500
MIN_POSITIVE_TEXT_LEN = 60
MIN_POSITIVE_ANSWER_LEN = 25
MIN_POSITIVE_ANSWER_RATIO = 0.15
REALIGN_WINDOW = 300


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    answer_text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    return normalize_ws(text)


def safe_slice(text: str, start: int, end: int) -> str:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    return text[start:end]


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return overlap_length(a_start, a_end, b_start, b_end) > 0


def contains_answer_loosely(container: str, answer: str) -> bool:
    if not answer:
        return True
    return normalize_ws(answer) in normalize_ws(container)


def realign_answer_span(context: str, answer_text: str, answer_start: int) -> tuple[int, int, bool]:
    answer_text = answer_text or ""
    expected_len = len(answer_text)
    direct_slice = safe_slice(context, answer_start, answer_start + expected_len)

    if direct_slice == answer_text:
        return answer_start, answer_start + expected_len, True

    if normalize_ws(direct_slice) == normalize_ws(answer_text):
        return answer_start, answer_start + expected_len, True

    window_start = max(0, answer_start - REALIGN_WINDOW)
    window_end = min(len(context), answer_start + expected_len + REALIGN_WINDOW)
    window = context[window_start:window_end]

    pos = window.find(answer_text)
    if pos != -1:
        realigned_start = window_start + pos
        return realigned_start, realigned_start + len(answer_text), True

    return answer_start, answer_start + expected_len, False


def build_chunks(context: str) -> list[Chunk]:
    raw_chunks = chunk_legal_text_with_offsets(context)
    chunks: list[Chunk] = []

    for raw in raw_chunks:
        text = str(raw["text"])
        start = int(raw["start_char"])
        end = int(raw["end_char"])

        cleaned = clean_text(text)
        if not cleaned:
            continue

        chunks.append(Chunk(text=cleaned, start=start, end=end))

    return chunks


def looks_like_definition(text: str) -> bool:
    t = normalize_ws(text)
    patterns = [
        r'^[\"“”\']?[A-Z0-9\-\(\)\/\.,&\s]{2,100}[\"“”\']?\s+(means|shall mean)\b',
        r'^\d+(\.\d+)*\s+[\"“”\']?.{1,100}?(means|shall mean)\b',
        r'\bhas the meaning set forth\b',
        r'\bshall have the meaning\b',
        r'\bis defined in\b',
        r'^\(?[A-Za-z0-9\.\-]+\)?\s+means\b',
        r'^as used in this agreement\b',
        r'^the following terms shall\b',
        r'^capitalized terms .* shall have the meanings\b',
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def looks_like_confidential_legend(text: str) -> bool:
    t = normalize_ws(text).lower()
    patterns = [
        r'confidential treatment has been requested',
        r'certain confidential information',
        r'filed separately with the securities and exchange commission',
        r'competitively harmful if publicly disclosed',
        r'has been omitted',
        r'redacted copy',
        r'execution version',
        r'complete, unredacted copies',
        r'note:\s*portions of this exhibit',
        r'confidential portions',
    ]
    return any(re.search(p, t) for p in patterns)


def looks_like_toc_or_index(text: str) -> bool:
    t = normalize_ws(text)

    if "table of contents" in t.lower():
        return True

    section_hits = len(re.findall(r'\bsection\s+\d+(\.\d+)*\b', t, flags=re.IGNORECASE))
    article_hits = len(re.findall(r'\barticle\s+[ivx0-9]+\b', t, flags=re.IGNORECASE))
    dot_leader_hits = len(re.findall(r'\.{3,}', t))

    if section_hits >= 3 or article_hits >= 3 or dot_leader_hits >= 2:
        return True

    if re.search(r'^\d+\.\s+[A-Z][A-Z \-]{3,}$', t):
        return True

    return False


def looks_like_signature_or_footer(text: str) -> bool:
    t = normalize_ws(text)
    patterns = [
        r'^/s/',
        r'\bsignature\b',
        r'^by:',
        r'^name:',
        r'^title:',
        r'^date:',
        r'\bin witness whereof\b',
        r'^chief executive officer\b',
        r'^chief scientific officer\b',
        r'^ceo\b',
        r'page\s+\d+(/\d+)?',
        r'^\[no further text on this page\]',
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def looks_like_title_or_party_block(text: str) -> bool:
    t = normalize_ws(text)

    title_patterns = [
        r'^(this\s+)?[A-Z][A-Z \-&,/()]{8,}(agreement|amendment|license|contract)\b',
        r'^(co-branding|services|consulting|distribution|reseller|manufacturing|supply|franchise|agency|promotion|development)\s+agreement\b',
        r'^amendment no\.',
        r'^exhibit\s*\(?[a-z0-9]+\)?',
        r'^miscellaneous provisions\b',
    ]

    party_patterns = [
        r'\bprincipal place of business\b',
        r'\bhaving its principal office\b',
        r'\bhaving offices located at\b',
        r'\bhereinafter referred to as\b',
        r'\ba Delaware corporation\b',
        r'\ba California corporation\b',
        r'\ba company organized and existing under the laws of\b',
        r'\bwith a place of business located at\b',
        r'\bwhose address is\b',
        r'\bcollectively the "parties"\b',
        r'\beach a "party"\b',
        r'^if to\b',
        r'^attn:',
    ]

    return any(re.search(p, t, flags=re.IGNORECASE) for p in title_patterns + party_patterns)


def looks_like_recital(text: str) -> bool:
    t = normalize_ws(text)
    patterns = [
        r'^whereas\b',
        r'^now therefore\b',
        r'^witnesseth\b',
        r'^for valuable consideration\b',
        r'^in consideration of the mutual\b',
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def looks_like_address_heavy(text: str) -> bool:
    t = normalize_ws(text)

    zip_like = len(re.findall(r'\b\d{5}(?:-\d{4})?\b', t))
    commas = t.count(",")

    if zip_like >= 1 and commas >= 2:
        return True

    if re.search(
        r'\b(suite|ste\.?|avenue|ave\.?|road|rd\.?|street|st\.?|boulevard|blvd\.?|floor|drive|dr\.?)\b',
        t,
        flags=re.IGNORECASE,
    ) and commas >= 2:
        return True

    return False


def looks_clipped(text: str, context: str, start: int, end: int) -> bool:
    if not text:
        return True

    prev_char = context[start - 1] if start > 0 else ""
    next_char = context[end] if end < len(context) else ""
    first_char = text[0]
    last_char = text[-1]

    start_mid_word = bool(prev_char and prev_char.isalnum() and first_char.isalnum())
    end_mid_word = bool(next_char and next_char.isalnum() and last_char.isalnum())

    if start_mid_word or end_mid_word:
        return True

    if re.search(r'^(ordance|nts|tion|ment|ing|ly)\b', text, flags=re.IGNORECASE):
        return True

    return False


def ends_like_fragment(text: str) -> bool:
    t = normalize_ws(text)

    bad_endings = {
        "in", "of", "to", "for", "by", "under", "within", "upon", "with", "from",
        "into", "than", "that", "which", "and", "or", "but", "including",
        "excluding", "during", "after", "before",
    }

    last_token_match = re.search(r'([A-Za-z]+)[^A-Za-z]*$', t)
    if not last_token_match:
        return False

    last_token = last_token_match.group(1).lower()
    if last_token in bad_endings:
        return True

    if not re.search(r'[.;:)]["\']?$', t):
        # allow some list-style items if long enough and substantive
        if len(t) < 140:
            return True

    return False


def has_page_noise(text: str) -> bool:
    t = normalize_ws(text)
    patterns = [
        r'page\s+\d+\s*/\s*\d+',
        r'issue\s+\d+\s+page\s+\d+/\d+',
        r'\bpage\s+\d+\b',
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def is_too_short_or_long(text: str) -> bool:
    t = normalize_ws(text)
    return len(t) < MIN_TEXT_LEN or len(t) > MAX_TEXT_LEN


def is_low_value_other(text: str, context: str, start: int, end: int) -> bool:
    t = normalize_ws(text)

    if is_too_short_or_long(t):
        return True
    if looks_like_definition(t):
        return True
    if looks_like_confidential_legend(t):
        return True
    if looks_like_toc_or_index(t):
        return True
    if looks_like_signature_or_footer(t):
        return True
    if looks_like_title_or_party_block(t):
        return True
    if looks_like_recital(t):
        return True
    if looks_like_address_heavy(t):
        return True
    if looks_clipped(t, context, start, end):
        return True
    if has_page_noise(t):
        return True

    return False


def is_fragmentary_positive(text: str, answer_text: str) -> bool:
    t = normalize_ws(text)
    answer = normalize_ws(answer_text)

    if len(answer) < MIN_POSITIVE_ANSWER_LEN:
        return True
    if len(t) < MIN_POSITIVE_TEXT_LEN:
        return True
    if ends_like_fragment(t):
        return True
    if has_page_noise(t):
        return True

    answer_ratio = len(answer) / max(len(t), 1)
    if answer_ratio < MIN_POSITIVE_ANSWER_RATIO and len(t) > 350:
        return True

    bad_leadins = [
        r':\s*$',
        r'you covenant .* you will not.*:\s*$',
        r'^section\s+\d+(\.\d+)*\b.*:\s*$',
    ]
    if any(re.search(p, t, flags=re.IGNORECASE) for p in bad_leadins):
        return True

    return False


def should_keep_positive(
    text: str,
    label: str,
    answer_text: str,
    context: str,
    start: int,
    end: int,
) -> bool:
    t = normalize_ws(text)

    if len(t) > MAX_TEXT_LEN:
        return False
    if looks_clipped(t, context, start, end):
        return False
    if looks_like_confidential_legend(t):
        return False
    if looks_like_toc_or_index(t):
        return False
    if looks_like_title_or_party_block(t):
        return False
    if looks_like_recital(t):
        return False
    if not contains_answer_loosely(t, answer_text):
        return False
    if is_fragmentary_positive(t, answer_text):
        return False

    return True


def find_covering_chunks(chunks: list[Chunk], span_start: int, span_end: int) -> list[int]:
    overlapping = [
        idx
        for idx, chunk in enumerate(chunks)
        if spans_overlap(chunk.start, chunk.end, span_start, span_end)
    ]
    if not overlapping:
        return []
    return list(range(overlapping[0], overlapping[-1] + 1))


def merge_span_group(
    context: str,
    label: str,
    answer_text: str,
    chunks: list[Chunk],
    covered_idx: list[int],
) -> dict[str, str] | None:
    first = chunks[covered_idx[0]]
    last = chunks[covered_idx[-1]]

    merged_start = first.start
    merged_end = last.end
    merged_text = clean_text(safe_slice(context, merged_start, merged_end))

    if not should_keep_positive(
        text=merged_text,
        label=label,
        answer_text=answer_text,
        context=context,
        start=merged_start,
        end=merged_end,
    ):
        return None

    return {
        "text": merged_text,
        "clause": label,
        "start_char": str(merged_start),
        "end_char": str(merged_end),
        "source_answer_text": clean_text(answer_text),
        "contains_answer": str(contains_answer_loosely(merged_text, answer_text)),
    }


def dedupe_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, str]] = []

    for row in rows:
        key = (
            row["text"],
            row["clause"],
            row["contract"],
            row["start_char"],
            row["end_char"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda r: (
            r["contract"],
            int(r["start_char"]),
            r["clause"],
            r["text"],
        ),
    )


def sample_negatives(chunks: list[Chunk], max_negatives: int) -> list[Chunk]:
    if len(chunks) <= max_negatives:
        return chunks
    return random.sample(chunks, max_negatives)


def main() -> None:
    random.seed(RANDOM_SEED)

    with open(CUAD_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows: list[dict[str, str]] = []

    positive_count = 0
    negative_count = 0
    alignment_fallbacks = 0
    dropped_positive_count = 0
    dropped_negative_count = 0

    for contract in data["data"]:
        contract_title = contract.get("title", "unknown_contract")

        for paragraph_index, paragraph in enumerate(contract["paragraphs"]):
            context = paragraph.get("context", "")
            if not context.strip():
                continue

            chunks = build_chunks(context)
            if not chunks:
                continue

            spans: list[Span] = []

            for qa in paragraph.get("qas", []):
                label = qa["id"].rsplit("__", 1)[-1].strip()
                if label not in SELECTED_LABELS:
                    continue

                for answer in qa.get("answers", []):
                    answer_text = answer.get("text", "")
                    answer_start = int(answer.get("answer_start", 0))

                    realigned_start, realigned_end, verified = realign_answer_span(
                        context=context,
                        answer_text=answer_text,
                        answer_start=answer_start,
                    )

                    if not verified:
                        alignment_fallbacks += 1

                    if realigned_end <= realigned_start:
                        continue

                    spans.append(
                        Span(
                            label=label,
                            start=realigned_start,
                            end=realigned_end,
                            answer_text=answer_text,
                        )
                    )

            if not spans:
                clean_negatives: list[Chunk] = []
                for chunk in chunks:
                    if is_low_value_other(chunk.text, context, chunk.start, chunk.end):
                        dropped_negative_count += 1
                        continue
                    clean_negatives.append(chunk)

                sampled = sample_negatives(clean_negatives, MAX_OTHER_PER_NO_LABEL_PARAGRAPH)

                for chunk in sampled:
                    all_rows.append(
                        {
                            "text": chunk.text,
                            "clause": OTHER_LABEL,
                            "contract": contract_title,
                            "start_char": str(chunk.start),
                            "end_char": str(chunk.end),
                            "source_answer_text": "",
                            "paragraph_index": str(paragraph_index),
                            "contains_answer": "",
                        }
                    )
                    negative_count += 1
                continue

            positive_rows_for_paragraph: list[dict[str, str]] = []
            positive_covered_ranges: list[tuple[int, int]] = []

            for span in spans:
                covered_idx = find_covering_chunks(chunks, span.start, span.end)
                if not covered_idx:
                    dropped_positive_count += 1
                    continue

                merged = merge_span_group(
                    context=context,
                    label=span.label,
                    answer_text=span.answer_text,
                    chunks=chunks,
                    covered_idx=covered_idx,
                )
                if merged is None:
                    dropped_positive_count += 1
                    continue

                merged["contract"] = contract_title
                merged["paragraph_index"] = str(paragraph_index)

                positive_rows_for_paragraph.append(merged)
                positive_covered_ranges.append(
                    (int(merged["start_char"]), int(merged["end_char"]))
                )
                positive_count += 1

            all_rows.extend(positive_rows_for_paragraph)

            candidate_negative_chunks: list[Chunk] = []
            for chunk in chunks:
                if is_low_value_other(chunk.text, context, chunk.start, chunk.end):
                    dropped_negative_count += 1
                    continue

                overlaps_positive = any(
                    spans_overlap(chunk.start, chunk.end, pos_start, pos_end)
                    for pos_start, pos_end in positive_covered_ranges
                )
                if overlaps_positive:
                    continue

                candidate_negative_chunks.append(chunk)

            max_negatives = max(
                1,
                int(len(positive_rows_for_paragraph) * NEGATIVE_TO_POSITIVE_RATIO),
            )
            sampled_negatives = sample_negatives(candidate_negative_chunks, max_negatives)

            for chunk in sampled_negatives:
                all_rows.append(
                    {
                        "text": chunk.text,
                        "clause": OTHER_LABEL,
                        "contract": contract_title,
                        "start_char": str(chunk.start),
                        "end_char": str(chunk.end),
                        "source_answer_text": "",
                        "paragraph_index": str(paragraph_index),
                        "contains_answer": "",
                    }
                )
                negative_count += 1

    final_rows = dedupe_rows(all_rows)
    final_rows = sort_rows(final_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "clause",
                "contract",
                "start_char",
                "end_char",
                "source_answer_text",
                "paragraph_index",
                "contains_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved {len(final_rows)} rows to {OUTPUT_PATH}")
    print(f"Positive rows kept: {positive_count}")
    print(f"Negative rows kept: {negative_count}")
    print(f"Alignment fallbacks used: {alignment_fallbacks}")
    print(f"Positive rows dropped by filters: {dropped_positive_count}")
    print(f"Negative rows dropped by filters: {dropped_negative_count}")


if __name__ == "__main__":
    main()