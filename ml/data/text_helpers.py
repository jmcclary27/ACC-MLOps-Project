# text_helpers.py
from __future__ import annotations

import re
from typing import Any

import anyascii
import pymupdf.layout
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


def is_header_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # Common legal heading patterns
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z/&,\-() ]+$", stripped):
        return True

    if re.match(r"^(Section|Clause|Schedule)\s+\d+([.\d]+)?[:.]?\s*", stripped, re.IGNORECASE):
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
    }
    if stripped.lower() in known_headers:
        return True

    return False


def is_enumeration_marker(text: str) -> bool:
    stripped = text.strip()

    patterns = [
        r"^\d+(\.\d+)*\.?$",   # 1, 1., 1.1, 2.4
        r"^\(?[a-zA-Z]\)$",    # (a)
        r"^\(?[ivxlcdmIVXLCDM]+\)$",  # (i), (ii)
        r"^[a-zA-Z]\.$",       # a.
        r"^[ivxlcdmIVXLCDM]+\.$",  # i.
        r"^[-•*]$",            # bullet only
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

    if re.match(r"^\)+$", stripped):
        return True

    if len(stripped) < 8:
        return True

    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) < 2:
        return True

    # Reject text that is overwhelmingly symbols/digits
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


def _split_paragraph_with_offsets(
    paragraph_text: str,
    paragraph_start: int,
    full_text: str,
    long_paragraph_threshold: int = 700,
) -> list[dict[str, int | str]]:
    """
    Split one paragraph-like block into clause-sized chunks.
    Keeps structure when possible, falls back to sentence splitting only for long text.
    """
    stripped_paragraph = paragraph_text.strip()
    if not stripped_paragraph:
        return []

    # Preserve shorter paragraph blocks as a unit
    if len(stripped_paragraph) <= long_paragraph_threshold:
        normalized = normalize_chunk_text(stripped_paragraph)
        local_start = paragraph_text.find(stripped_paragraph)
        abs_start = paragraph_start + local_start
        abs_end = abs_start + len(stripped_paragraph)

        if not is_junk_chunk(normalized):
            return [{
                "text": normalized,
                "start_char": abs_start,
                "end_char": abs_end,
            }]
        return []

    # For long blocks, use sentence fallback
    doc = _NLP(stripped_paragraph)
    sentence_chunks: list[dict[str, int | str]] = []
    cursor = 0

    for sent in doc.sents:
        sent_text_raw = sent.text.strip()
        if not sent_text_raw:
            continue

        span = _find_non_overlapping_span(stripped_paragraph, sent_text_raw, cursor)
        if span is None:
            continue

        local_start, local_end = span
        cursor = local_end

        abs_start = paragraph_start + paragraph_text.find(stripped_paragraph) + local_start
        abs_end = paragraph_start + paragraph_text.find(stripped_paragraph) + local_end
        normalized = normalize_chunk_text(sent_text_raw)

        if is_junk_chunk(normalized):
            continue

        sentence_chunks.append({
            "text": normalized,
            "start_char": abs_start,
            "end_char": abs_end,
        })

    return sentence_chunks


def chunk_legal_text_with_offsets(text: str) -> list[dict[str, int | str]]:
    """
    Chunk legal text using paragraph-first logic with offsets.

    Strategy:
    - split on paragraph boundaries first
    - preserve short/medium legal paragraphs as a whole
    - fallback to sentence splitting only for long paragraphs
    - filter obvious junk fragments like '1.1', '(a)', 'Source: ...'
    """
    if not text or not text.strip():
        return []

    chunks: list[dict[str, int | str]] = []

    # Find paragraph-like blocks including their absolute positions
    paragraph_pattern = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
    for match in paragraph_pattern.finditer(text):
        paragraph_text = match.group()
        paragraph_start = match.start()

        paragraph_chunks = _split_paragraph_with_offsets(
            paragraph_text=paragraph_text,
            paragraph_start=paragraph_start,
            full_text=text,
        )
        chunks.extend(paragraph_chunks)

    # Merge header with following chunk when appropriate
    merged_chunks: list[dict[str, int | str]] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_text = str(current["text"]).strip()

        if (
            is_header_like(current_text)
            and i + 1 < len(chunks)
            and not is_header_like(str(chunks[i + 1]["text"]).strip())
        ):
            nxt = chunks[i + 1]
            merged_chunks.append({
                "text": f"{current_text}\n{str(nxt['text']).strip()}",
                "start_char": int(current["start_char"]),
                "end_char": int(nxt["end_char"]),
            })
            i += 2
            continue

        merged_chunks.append(current)
        i += 1

    return merged_chunks