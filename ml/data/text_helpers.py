# text_helpers.py
from __future__ import annotations

import re
from typing import Any

import anyascii
import pymupdf4llm
import spacy


# Load once, not on every function call
_NLP = spacy.load("en_core_web_sm")


def pdf_to_text(path: str, as_text: bool = False, as_json: bool = False) -> Any:
    if as_text and as_json:
        raise ValueError("either as_text=True OR as_json=True")

    if as_text:
        return pymupdf4llm.to_text(path)

    if as_json:
        return pymupdf4llm.to_json(path)

    return pymupdf4llm.to_markdown(path)


def split_by_clause_numbers(text: str) -> list[str]:
    """
    Split while keeping numbered legal clause markers attached to the content
    that follows them.

    Examples:
    - 1.2 Payment Terms
    - 9.10 Termination
    - 12.3 Confidentiality
    """
    return re.split(r"(?=\b\d{1,3}\.\d+\s)", text)


def split_into_sentence(text: str, minimum_character_length: int = 2) -> list[str]:
    doc = _NLP(text)
    sentences: list[str] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text or len(sent_text) <= minimum_character_length:
            continue
        sentences.append(sent_text)

    return sentences


def clean_text(text: str, unicode_to_ascii: bool = False) -> str:
    if unicode_to_ascii:
        text = anyascii.anyascii(text)

    text = " ".join(text.split())
    return text


def normalize_chunk_text(text: str) -> str:
    """
    Light normalization that preserves legal content while cleaning spacing.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def strip_extraction_noise(text: str) -> str:
    """
    Remove recurring extraction artifacts such as page footers and standalone
    page numbers before chunking.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if re.search(r"Source:\s+.*10-K", stripped, re.IGNORECASE):
            continue

        if re.match(r"^\d+\s+Source:", stripped, re.IGNORECASE):
            continue

        if re.match(r"^\d{1,3}$", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_header_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z/&,\-() ]+$", stripped):
        return True

    if re.match(
        r"^(Section|Clause|Schedule|Article)\s+\d+([.\d]+)?[:.]?\s*",
        stripped,
        re.IGNORECASE,
    ):
        return True

    if len(stripped.split()) <= 8 and stripped.upper() == stripped and re.search(r"[A-Z]", stripped):
        return True

    known_headers = {
        "background",
        "agreed terms",
        "definitions and interpretation",
        "appointment",
        "prices and payment",
        "duration and termination",
        "effects of termination",
        "confidentiality",
        "governing law and jurisdiction",
        "notices",
        "variation",
        "waiver",
        "severance",
        "entire agreement",
        "third party rights",
        "no partnership or agency",
        "counterparts",
        "trade marks and intellectual property",
        "product liability and insurance",
        "advertising and promotion",
        "anti-bribery compliance",
        "supply of products",
        "supplier's undertakings",
        "distributor's undertakings",
        "term and termination",
        "payment terms",
        "limitation of liability",
        "indemnification",
        "force majeure",
    }
    if stripped.lower() in known_headers:
        return True

    return False


def is_enumeration_marker(text: str) -> bool:
    stripped = text.strip()

    patterns = [
        r"^\d+(\.\d+)*\.?$",         # 1, 1., 1.1, 2.4
        r"^\(?[a-zA-Z]\)$",          # (a)
        r"^\(?[ivxlcdmIVXLCDM]+\)$", # (i), (ii)
        r"^[a-zA-Z]\.$",             # a.
        r"^[ivxlcdmIVXLCDM]+\.$",    # i.
        r"^[-•*]$",                  # bullet only
    ]
    return any(re.match(pattern, stripped) for pattern in patterns)


def is_junk_chunk(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if is_enumeration_marker(stripped):
        return True

    if re.match(r"^source:\s", stripped, re.IGNORECASE):
        return True

    if re.match(r"^exhibit\s+\S+", stripped, re.IGNORECASE):
        return True

    if re.match(r"^(schedule\s+\d+)", stripped, re.IGNORECASE):
        return True

    if re.match(r"^signed by", stripped, re.IGNORECASE):
        return True

    if re.match(r"^director/?secretary$", stripped, re.IGNORECASE):
        return True

    if re.match(r"^director$", stripped, re.IGNORECASE):
        return True

    if re.match(r"^name\s*\(please print\)$", stripped, re.IGNORECASE):
        return True

    if re.match(r"^\)+$", stripped):
        return True

    if len(stripped) < 8:
        return True

    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) < 2:
        return True

    alnum_chars = re.findall(r"[A-Za-z0-9]", stripped)
    alpha_words = re.findall(r"[A-Za-z]{2,}", stripped)
    if alnum_chars and len(alpha_words) == 0:
        return True

    return False


def _find_non_overlapping_span(
    full_text: str,
    fragment: str,
    search_start: int,
    search_end: int | None = None,
) -> tuple[int, int] | None:
    """
    Find a fragment inside full_text beginning at search_start.
    Returns absolute start/end indices.
    """
    if not fragment:
        return None

    start = full_text.find(fragment, search_start, search_end)
    if start == -1:
        return None

    return start, start + len(fragment)


def _make_chunk(text: str, start_char: int, end_char: int) -> dict[str, int | str] | None:
    normalized = normalize_chunk_text(text)
    if is_junk_chunk(normalized):
        return None

    return {
        "text": normalized,
        "start_char": start_char,
        "end_char": end_char,
    }


def _force_section_breaks(text: str) -> str:
    """
    Insert paragraph breaks before likely top-level section headers and numbered
    legal clauses so they become their own paragraph-like blocks.

    This improves:
    - top-level section splitting
    - intro material splitting
    - numbered clause splitting like 1.2, 9.10, 12.3
    """
    patterns = [
        # Top-level numeric headers: "1. Definitions", "12. Termination"
        r"(?<!\n)(?<!\d)\s(?=(\d{1,3}\.\s+[A-Z][^\n]{0,120}))",

        # Numbered subclauses: "1.2 Payment", "9.10 Survival", "12.3 Confidentiality"
        r"(?<!\n)\s(?=(\d{1,3}\.\d{1,3}\s+[A-Z(][^\n]{0,120}))",

        # Section / Article / Clause style headers
        r"(?<!\n)\s(?=((Section|Article|Clause)\s+\d+([.\d]+)?[:.]?\s+[A-Z][^\n]{0,120}))",

        # ALL CAPS headers
        r"(?<!\n)\s(?=([A-Z][A-Z/&,\-() ]{4,}))",
    ]

    forced = text
    for pattern in patterns:
        forced = re.sub(pattern, "\n\n", forced)

    return forced


def _split_intro_material(
    stripped_paragraph: str,
    paragraph_text: str,
    paragraph_start: int,
) -> list[dict[str, int | str]]:
    """
    Better split front-matter / intro blocks when extraction collapses them into a
    single long paragraph.

    Targets things like:
    - THIS AGREEMENT ...
    - dated ...
    - between ...
    - where / whereas ...
    """
    intro_patterns = [
        r"(?=\bdated\b)",
        r"(?=\bbetween\b)",
        r"(?=\bby and between\b)",
        r"(?=\bwhereas\b)",
        r"(?=\bnow[,]?\s*therefore\b)",
        r"(?=\bwitnesseth\b)",
        r"(?=\brecitals?\b)",
    ]

    parts = [stripped_paragraph]
    for pattern in intro_patterns:
        next_parts: list[str] = []
        for part in parts:
            split_parts = re.split(pattern, part, flags=re.IGNORECASE)
            next_parts.extend(split_parts)
        parts = next_parts

    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        return []

    base_offset = paragraph_text.find(stripped_paragraph)
    chunks: list[dict[str, int | str]] = []
    cursor = 0

    for part in parts:
        span = _find_non_overlapping_span(stripped_paragraph, part, cursor)
        if span is None:
            continue

        local_start, local_end = span
        cursor = local_end

        abs_start = paragraph_start + base_offset + local_start
        abs_end = paragraph_start + base_offset + local_end

        chunk = _make_chunk(part, abs_start, abs_end)
        if chunk is not None:
            chunks.append(chunk)

    return chunks


def _split_long_paragraph_by_subclauses(
    stripped_paragraph: str,
    paragraph_text: str,
    paragraph_start: int,
) -> list[dict[str, int | str]]:
    """
    Split long legal paragraphs using:
    1. subclause markers like (a), (b), (i)
    2. numbered clause markers like 1.2, 2.4, 9.10, 12.3

    Keeps the marker attached to the following content.
    """
    base_offset = paragraph_text.find(stripped_paragraph)

    def build_chunks(parts: list[str]) -> list[dict[str, int | str]]:
        chunks: list[dict[str, int | str]] = []
        cursor = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            span = _find_non_overlapping_span(stripped_paragraph, part, cursor)
            if span is None:
                continue

            local_start, local_end = span
            cursor = local_end

            abs_start = paragraph_start + base_offset + local_start
            abs_end = paragraph_start + base_offset + local_end

            chunk = _make_chunk(part, abs_start, abs_end)
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    subclauses = re.split(
        r"(?=\(\s*[a-zA-ZivxlcdmIVXLCDM]+\s*\))",
        stripped_paragraph,
    )
    if len(subclauses) > 1:
        subclause_chunks = build_chunks(subclauses)
        if subclause_chunks:
            return subclause_chunks

    numbered_parts = re.split(r"(?=\b\d{1,3}\.\d{1,3}\s)", stripped_paragraph)
    if len(numbered_parts) > 1:
        numbered_chunks = build_chunks(numbered_parts)
        if numbered_chunks:
            return numbered_chunks

    return []


def _split_long_paragraph_by_sentences(
    stripped_paragraph: str,
    paragraph_text: str,
    paragraph_start: int,
) -> list[dict[str, int | str]]:
    """
    Sentence fallback when a paragraph is still too long and does not split
    naturally by subclause markers.
    """
    doc = _NLP(stripped_paragraph)
    sentence_chunks: list[dict[str, int | str]] = []
    cursor = 0
    base_offset = paragraph_text.find(stripped_paragraph)

    for sent in doc.sents:
        sent_text_raw = sent.text.strip()
        if not sent_text_raw:
            continue

        span = _find_non_overlapping_span(stripped_paragraph, sent_text_raw, cursor)
        if span is None:
            continue

        local_start, local_end = span
        cursor = local_end

        abs_start = paragraph_start + base_offset + local_start
        abs_end = paragraph_start + base_offset + local_end

        chunk = _make_chunk(sent_text_raw, abs_start, abs_end)
        if chunk is not None:
            sentence_chunks.append(chunk)

    return sentence_chunks


def _split_paragraph_with_offsets(
    paragraph_text: str,
    paragraph_start: int,
    long_paragraph_threshold: int = 500,
) -> list[dict[str, int | str]]:
    """
    Split one paragraph-like block into clause-sized chunks.

    Strategy:
    - keep short/medium legal paragraphs intact
    - for long paragraphs, split intro material first
    - then split by subclause markers
    - then use sentence fallback
    """
    stripped_paragraph = paragraph_text.strip()
    if not stripped_paragraph:
        return []

    local_start = paragraph_text.find(stripped_paragraph)
    abs_start = paragraph_start + local_start
    abs_end = abs_start + len(stripped_paragraph)

    if len(stripped_paragraph) <= long_paragraph_threshold:
        chunk = _make_chunk(stripped_paragraph, abs_start, abs_end)
        return [chunk] if chunk is not None else []

    intro_chunks = _split_intro_material(
        stripped_paragraph=stripped_paragraph,
        paragraph_text=paragraph_text,
        paragraph_start=paragraph_start,
    )
    if intro_chunks:
        return intro_chunks

    subclause_chunks = _split_long_paragraph_by_subclauses(
        stripped_paragraph=stripped_paragraph,
        paragraph_text=paragraph_text,
        paragraph_start=paragraph_start,
    )
    if subclause_chunks:
        return subclause_chunks

    return _split_long_paragraph_by_sentences(
        stripped_paragraph=stripped_paragraph,
        paragraph_text=paragraph_text,
        paragraph_start=paragraph_start,
    )


def chunk_legal_text_with_offsets(text: str) -> list[dict[str, int | str]]:
    """
    Chunk legal text using paragraph-first logic with offsets.

    Strategy:
    - remove recurring extraction noise first
    - force section and clause boundaries into paragraph breaks
    - split on paragraph boundaries
    - preserve short/medium legal paragraphs as a whole
    - split long paragraphs by intro markers, then subclause markers
    - use sentence fallback only when needed
    - filter obvious junk fragments and metadata
    """
    if not text or not text.strip():
        return []

    cleaned_text = strip_extraction_noise(text)
    cleaned_text = _force_section_breaks(cleaned_text)

    chunks: list[dict[str, int | str]] = []

    paragraph_pattern = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
    for match in paragraph_pattern.finditer(cleaned_text):
        paragraph_text = match.group()
        paragraph_start = match.start()

        paragraph_chunks = _split_paragraph_with_offsets(
            paragraph_text=paragraph_text,
            paragraph_start=paragraph_start,
        )
        chunks.extend(paragraph_chunks)

    merged_chunks: list[dict[str, int | str]] = []
    i = 0

    while i < len(chunks):
        current = chunks[i]
        current_text = str(current["text"]).strip()

        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            next_text = str(next_chunk["text"]).strip()
            gap = int(next_chunk["start_char"]) - int(current["end_char"])

            if (
                is_header_like(current_text)
                and not is_header_like(next_text)
                and gap < 50
            ):
                merged_chunks.append(
                    {
                        "text": f"{current_text}\n{next_text}",
                        "start_char": int(current["start_char"]),
                        "end_char": int(next_chunk["end_char"]),
                    }
                )
                i += 2
                continue

        merged_chunks.append(current)
        i += 1

    return merged_chunks