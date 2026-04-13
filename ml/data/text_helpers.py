# ml/data/text_helpers.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import anyascii
import pymupdf4llm
import spacy


def _load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


_NLP = _load_nlp()


# -------------------------------------------------------------------
# Public extraction helpers
# -------------------------------------------------------------------

def pdf_to_text(path: str, as_text: bool = False, as_json: bool = False) -> Any:
    if as_text and as_json:
        raise ValueError("either as_text=True OR as_json=True")

    if as_text:
        return pymupdf4llm.to_text(path)

    if as_json:
        return pymupdf4llm.to_json(path)

    return pymupdf4llm.to_markdown(path)


def clean_text(text: str, unicode_to_ascii: bool = False) -> str:
    if unicode_to_ascii:
        text = anyascii.anyascii(text)

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_chunk_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = text.replace(";", ";")
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')

    # remove inline source / filing junk that sometimes gets collapsed into intro text
    text = re.sub(
        r"source:\s+.*?(?:10-k|exhibit\s+\d+(?:\.\d+)?)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return text.strip()


def split_into_sentence(text: str, minimum_character_length: int = 2) -> list[str]:
    doc = _NLP(text)
    sentences: list[str] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) <= minimum_character_length:
            continue
        sentences.append(sent_text)

    return sentences


def split_by_clause_numbers(text: str) -> list[str]:
    parts = re.split(
        r"(?=(?:^|\s)(?:\d{1,3}(?:\.\d{1,3})+|\d{1,3}\.)\s+[A-Z])",
        text,
    )
    return [p.strip() for p in parts if p and p.strip()]


# -------------------------------------------------------------------
# Span helpers
# -------------------------------------------------------------------

@dataclass
class LineSpan:
    text: str
    start: int
    end: int
    line_no: int


@dataclass
class BlockSpan:
    text: str
    start: int
    end: int


def _iter_line_spans(text: str) -> list[LineSpan]:
    lines: list[LineSpan] = []
    cursor = 0

    for i, raw_line in enumerate(text.splitlines(keepends=True)):
        line_start = cursor
        line_end = cursor + len(raw_line)
        line_text = raw_line.rstrip("\r\n")
        lines.append(LineSpan(text=line_text, start=line_start, end=line_end, line_no=i))
        cursor = line_end

    return lines


def _make_chunk(text: str, start_char: int, end_char: int) -> dict[str, int | str] | None:
    normalized = normalize_chunk_text(text)
    if _is_junk_chunk(normalized):
        return None

    return {
        "text": normalized,
        "start_char": start_char,
        "end_char": end_char,
    }


# -------------------------------------------------------------------
# Noise and artifact handling
# -------------------------------------------------------------------

def _normalized_repetition_key(line: str) -> str:
    s = line.strip().lower()
    s = re.sub(r"\bpage\s+\d+(\s+of\s+\d+)?\b", "page_num", s)
    s = re.sub(r"\b\d+\b", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_page_number(line: str) -> bool:
    s = line.strip()
    return bool(
        re.fullmatch(r"\d{1,4}", s)
        or re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", s, flags=re.IGNORECASE)
        or re.fullmatch(r"-\s*\d+\s*-", s)
    )


def _looks_like_extraction_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    patterns = [
        r"^source:\s+.*10-k.*$",
        r"^source:\s+.*exhibit.*$",
        r"^\d+\s+source:.*$",
        r"^exhibit\s+[a-z0-9.\-]+$",
        r"^edgar\s+filing.*$",
        r"^sec\s+filing.*$",
        r"^fuse medical.*10-k.*$",
        r"^.*10-k.*exhibit.*$",
    ]
    return any(re.match(p, s, flags=re.IGNORECASE) for p in patterns)


def _looks_like_signature_artifact(line: str) -> bool:
    s = line.strip()
    patterns = [
        r"^signed by\b",
        r"^signature\b",
        r"^director/?secretary\b",
        r"^director\b",
        r"^title\b$",
        r"^date\b$",
        r"^name\s*\(please print\)$",
        r"^\)+$",
    ]
    return any(re.match(p, s, flags=re.IGNORECASE) for p in patterns)


def _is_signature_block_text(text: str) -> bool:
    s = normalize_chunk_text(text).lower()

    score = 0
    markers = [
        "signed by",
        "director/secretary",
        "name (please print)",
        "in accordance with section 127",
        "corporations act",
        "signature orthopaedics",
    ]
    for marker in markers:
        if marker in s:
            score += 1

    return score >= 2


def _detect_repeated_header_footer_keys(lines: list[LineSpan]) -> set[str]:
    counter: Counter[str] = Counter()

    for line in lines:
        s = line.text.strip()
        if not s:
            continue
        if len(s) <= 120:
            counter[_normalized_repetition_key(s)] += 1

    return {k for k, v in counter.items() if v >= 3}


def _is_probable_header_footer(line: str, repeated_keys: set[str]) -> bool:
    s = line.strip()
    if not s:
        return False

    key = _normalized_repetition_key(s)

    if key in repeated_keys:
        return True
    if _looks_like_page_number(s):
        return True
    if _looks_like_extraction_noise(s):
        return True

    return False


# -------------------------------------------------------------------
# Structural classifiers
# -------------------------------------------------------------------

def is_header_like(text: str) -> bool:
    s = text.strip()
    if not s:
        return False

    if len(s) > 180:
        return False

    patterns = [
        r"^\d+\.\s+[A-Z][^\n]{0,160}$",
        r"^(Section|Clause|Article|Schedule|Appendix|Exhibit)\s+\d+([.\d]+)?[:.]?\s*.+$",
        r"^[A-Z][A-Z0-9/&,()' \-]{3,}$",
    ]
    if any(re.match(p, s, flags=re.IGNORECASE) for p in patterns):
        return True

    known_headers = {
        "agreed terms",
        "definitions",
        "definitions and interpretation",
        "appointment",
        "distributor's undertakings",
        "supplier's undertakings",
        "supply of products",
        "prices and payment",
        "gst and taxes",
        "advertising and promotion",
        "anti-bribery compliance",
        "trade marks and intellectual property",
        "product liability and insurance",
        "duration and termination",
        "effects of termination",
        "confidentiality",
        "entire agreement",
        "assignment and other dealings prohibited",
        "freedom to contract",
        "third party rights",
        "no partnership or agency",
        "governing law and jurisdiction",
    }
    if s.lower() in known_headers:
        return True

    words = s.split()
    if len(words) <= 12 and s.upper() == s and re.search(r"[A-Z]", s):
        return True

    return False


def _is_top_level_clause_header(text: str) -> bool:
    s = text.strip()
    return bool(re.match(r"^\d+\.\s+[A-Z][^\n]{0,160}$", s))


def _is_subclause_start(text: str) -> bool:
    s = text.strip()
    return bool(
        re.match(r"^\d+\.\d+\s+.+$", s)
        or re.match(r"^\([a-zA-Z]\)\s+.+$", s)
        or re.match(r"^\([ivxlcdmIVXLCDM]+\)\s+.+$", s)
        or re.match(r"^[a-zA-Z]\.\s+.+$", s)
        or re.match(r"^[ivxlcdmIVXLCDM]+\.\s+.+$", s)
        or re.match(r"^[•\-*]\s+.+$", s)
    )


def _is_marker_only(text: str) -> bool:
    s = text.strip()
    patterns = [
        r"^\d+(\.\d+)*\.?$",
        r"^\(?[a-zA-Z]\)?\.?$",
        r"^\(?[ivxlcdmIVXLCDM]+\)?\.?$",
        r"^[•\-*]$",
    ]
    return any(re.match(p, s) for p in patterns)


def _is_junk_chunk(text: str) -> bool:
    s = text.strip()
    if not s:
        return True

    if _is_marker_only(s):
        return True

    if len(s) < 8:
        return True

    if _looks_like_page_number(s):
        return True

    if _looks_like_extraction_noise(s):
        return True

    if _looks_like_signature_artifact(s):
        return True

    if _is_signature_block_text(s):
        return True

    words = re.findall(r"[A-Za-z]{2,}", s)
    if len(words) < 2:
        return True

    return False


# -------------------------------------------------------------------
# Block building
# -------------------------------------------------------------------

def _keep_line_for_chunking(line: LineSpan, repeated_keys: set[str]) -> bool:
    s = line.text.strip()

    if not s:
        return True

    if _is_probable_header_footer(s, repeated_keys):
        return False

    return True


def _group_intro_lines(kept_lines: list[LineSpan]) -> tuple[BlockSpan | None, int]:
    if not kept_lines:
        return None, 0

    intro_lines: list[LineSpan] = []

    for line in kept_lines:
        s = line.text.strip()

        if not s:
            if intro_lines:
                intro_lines.append(line)
            continue

        if s.lower() == "agreed terms":
            break
        if _is_top_level_clause_header(s):
            break

        intro_lines.append(line)

    meaningful = [ln for ln in intro_lines if ln.text.strip()]
    if len(meaningful) < 2:
        return None, 0

    start = meaningful[0].start
    end = meaningful[-1].end

    text = _ORIGINAL_TEXT_CACHE[start:end]

    text = re.sub(
        r"source:\s+.*?(?=\bthis agreement\b)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\bexhibit\s+\d+(?:\.\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10-k\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsec filing\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\d{2,}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None, 0

    return BlockSpan(text=text, start=start, end=end), len(intro_lines)


def _build_blocks_from_lines(text: str) -> list[BlockSpan]:
    global _ORIGINAL_TEXT_CACHE
    _ORIGINAL_TEXT_CACHE = text

    lines = _iter_line_spans(text)
    repeated_keys = _detect_repeated_header_footer_keys(lines)
    kept = [ln for ln in lines if _keep_line_for_chunking(ln, repeated_keys)]

    blocks: list[BlockSpan] = []

    intro_block, intro_used = _group_intro_lines(kept)
    start_idx = 0
    if intro_block is not None:
        if not _is_junk_chunk(intro_block.text):
            blocks.append(intro_block)
        start_idx = intro_used

    current: list[LineSpan] = []

    def flush_current() -> None:
        if not current:
            return

        block_start = current[0].start
        block_end = current[-1].end
        block_text = text[block_start:block_end]
        stripped = block_text.strip()

        current.clear()

        if not stripped:
            return

        local_start = block_text.find(stripped)
        abs_start = block_start + local_start
        abs_end = abs_start + len(stripped)
        candidate = text[abs_start:abs_end]

        if _is_signature_block_text(candidate):
            return

        blocks.append(BlockSpan(text=candidate, start=abs_start, end=abs_end))

    for line in kept[start_idx:]:
        s = line.text.strip()

        if not s:
            flush_current()
            continue

        if current and (_is_top_level_clause_header(s) or is_header_like(s)):
            flush_current()

        current.append(line)

        # split bullets / subclauses into their own blocks
        if _is_subclause_start(s):
            flush_current()
            continue

        # standalone headers become their own block
        if is_header_like(s):
            flush_current()
            continue

    flush_current()
    return blocks


# -------------------------------------------------------------------
# Block splitting
# -------------------------------------------------------------------

def _safe_inline_split_points(text: str) -> list[int]:
    """
    Find safe inline clause/subclause starts inside a block.

    Goals:
    - split real numbered clauses and bullets
    - split definition entries like "Territory: ..."
    - split bullets after semicolons
    - avoid false splits in clause references like:
      "in accordance with clause 11. Territory..."
    """
    split_points: set[int] = set()

    patterns = [
        # numbered subclauses at start or after newline / hard punctuation
        r"(?:(?<=^)|(?<=[\n:;]))\s*(\d{1,3}\.\d+\s+[A-Z(])",

        # top-level numbered clauses at start or after newline only
        r"(?:(?<=^)|(?<=\n))\s*(\d{1,3}\.\s+[A-Z])",

        # alphabetic bullets at start / newline / colon / semicolon
        r"(?:(?<=^)|(?<=[\n:;]))\s*(\([a-zA-Z]\)\s+)",

        # roman bullets at start / newline / colon / semicolon
        r"(?:(?<=^)|(?<=[\n:;]))\s*(\([ivxlcdmIVXLCDM]+\)\s+)",

        # ✅ definition entries (robust, works mid-sentence)
        r"(?<!\w)([A-Z][A-Za-z ]{2,40}:\s+)",

        # bullets after semicolons
        r"(?<=;)\s*(\([a-z]\)\s+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            group_start = match.start(1)
            prefix = text[max(0, group_start - 120):group_start].lower()

            # 🚨 If we are already inside a definition list, ALWAYS split
            if re.search(r"[a-z][a-z ]{2,40}:\s", prefix):
                split_points.add(group_start)
                continue

            # 🚨 Otherwise block true clause references
            if re.search(r"\b(clause|section|article)\s+\d+\.\s*$", prefix):
                continue

            split_points.add(group_start)

    return sorted(split_points)


def _split_block_by_inline_markers(block: BlockSpan) -> list[BlockSpan]:
    text = block.text
    points = _safe_inline_split_points(text)

    if not points:
        return [block]

    if points[0] != 0:
        points = [0] + points

    pieces: list[BlockSpan] = []

    for i, start in enumerate(points):
        end = points[i + 1] if i + 1 < len(points) else len(text)
        piece = text[start:end].strip()
        if not piece:
            continue

        local_start = text.find(piece, start, end)
        abs_start = block.start + local_start
        abs_end = abs_start + len(piece)
        pieces.append(BlockSpan(text=piece, start=abs_start, end=abs_end))

    return pieces


def _sentence_groups(block: BlockSpan) -> list[tuple[int, int]]:
    text = block.text
    doc = _NLP(text)
    spans: list[tuple[int, int]] = []

    for sent in doc.sents:
        start = sent.start_char
        end = sent.end_char

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if start < end:
            spans.append((start, end))

    return spans


def _split_long_block_by_sentences(
    block: BlockSpan,
    max_chars: int = 300,
) -> list[BlockSpan]:
    sent_spans = _sentence_groups(block)
    if not sent_spans:
        return [block]

    pieces: list[BlockSpan] = []
    group_start = sent_spans[0][0]
    group_end = sent_spans[0][1]

    for sent_start, sent_end in sent_spans[1:]:
        proposed_len = sent_end - group_start

        if proposed_len > max_chars:
            text_piece = block.text[group_start:group_end].strip()
            if text_piece:
                abs_start = block.start + group_start
                abs_end = block.start + group_end
                pieces.append(BlockSpan(text=text_piece, start=abs_start, end=abs_end))
            group_start = sent_start
            group_end = sent_end
        else:
            group_end = sent_end

    text_piece = block.text[group_start:group_end].strip()
    if text_piece:
        abs_start = block.start + group_start
        abs_end = block.start + group_end
        pieces.append(BlockSpan(text=text_piece, start=abs_start, end=abs_end))

    # extra fallback for oversized chunks that still survive sentence grouping
    final_pieces: list[BlockSpan] = []
    for piece in pieces:
        if len(piece.text) <= max_chars:
            final_pieces.append(piece)
        else:
            final_pieces.extend(_force_split_very_long_block(piece, max_chars=max_chars))

    return final_pieces


def _force_split_very_long_block(
    block: BlockSpan,
    max_chars: int = 350,
) -> list[BlockSpan]:
    """
    For very long chunks with poor sentence boundaries, split by semicolons first,
    then by commas as a last resort.
    """
    text = block.text
    if len(text) <= max_chars:
        return [block]

    # first try semicolon-based splitting
    cut_points = [0]
    for m in re.finditer(r";\s+", text):
        cut_points.append(m.end())

    if len(cut_points) > 1:
        cut_points = sorted(set(cut_points))
        pieces: list[BlockSpan] = []
        start = cut_points[0]

        for i in range(1, len(cut_points)):
            end = cut_points[i]
            if end - start >= max_chars:
                piece = text[start:end].strip()
                if piece:
                    local_start = text.find(piece, start, end)
                    abs_start = block.start + local_start
                    abs_end = abs_start + len(piece)
                    pieces.append(BlockSpan(text=piece, start=abs_start, end=abs_end))
                start = end

        if start < len(text):
            piece = text[start:].strip()
            if piece:
                local_start = text.find(piece, start)
                abs_start = block.start + local_start
                abs_end = abs_start + len(piece)
                pieces.append(BlockSpan(text=piece, start=abs_start, end=abs_end))

        if pieces:
            return pieces

    # fallback: split by commas if still enormous and no semicolons helped
    pieces: list[BlockSpan] = []
    comma_points = [0]
    for m in re.finditer(r",\s+", text):
        comma_points.append(m.end())

    if len(comma_points) > 1:
        comma_points = sorted(set(comma_points))
        start = comma_points[0]

        for i in range(1, len(comma_points)):
            end = comma_points[i]
            if end - start >= max_chars:
                piece = text[start:end].strip()
                if piece:
                    local_start = text.find(piece, start, end)
                    abs_start = block.start + local_start
                    abs_end = abs_start + len(piece)
                    pieces.append(BlockSpan(text=piece, start=abs_start, end=abs_end))
                start = end

        if start < len(text):
            piece = text[start:].strip()
            if piece:
                local_start = text.find(piece, start)
                abs_start = block.start + local_start
                abs_end = abs_start + len(piece)
                pieces.append(BlockSpan(text=piece, start=abs_start, end=abs_end))

        if pieces:
            return pieces

    return [block]


def _split_block(block: BlockSpan, max_chars: int = 350) -> list[BlockSpan]:
    text = block.text.strip()
    if not text:
        return []

    if _is_signature_block_text(text):
        return []

    # keep clean headers whole if they are short
    if is_header_like(text) and len(text) <= max_chars:
        return [block]

    first_pass = _split_block_by_inline_markers(block)
    final: list[BlockSpan] = []

    for piece in first_pass:
        if len(piece.text) > max_chars:
            second_pass = _split_long_block_by_sentences(piece, max_chars=max_chars)

            for sub_piece in second_pass:
                if len(sub_piece.text) > max_chars:
                    final.extend(
                        _force_split_very_long_block(sub_piece, max_chars=max_chars)
                    )
                else:
                    final.append(sub_piece)
        else:
            final.append(piece)

    return final


# -------------------------------------------------------------------
# Header merges and cleanup
# -------------------------------------------------------------------

def _merge_headers_with_following_content(
    chunks: list[dict[str, int | str]],
    max_gap: int = 80,
) -> list[dict[str, int | str]]:
    merged: list[dict[str, int | str]] = []
    i = 0

    while i < len(chunks):
        current = chunks[i]
        current_text = str(current["text"]).strip()

        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            next_text = str(nxt["text"]).strip()
            gap = int(nxt["start_char"]) - int(current["end_char"])

            if is_header_like(current_text) and gap <= max_gap:
                merged.append(
                    {
                        "text": f"{current_text}\n{next_text}",
                        "start_char": int(current["start_char"]),
                        "end_char": int(nxt["end_char"]),
                    }
                )
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def _merge_orphan_tail_chunks(
    chunks: list[dict[str, int | str]],
    max_short_len: int = 40,
    max_gap: int = 10,
) -> list[dict[str, int | str]]:
    """
    Fix cases like:
    - "12.3 For the purposes of clause 12.2"
    - then "(a) ..."
    """
    if not chunks:
        return chunks

    merged: list[dict[str, int | str]] = []
    i = 0

    while i < len(chunks):
        current = chunks[i]
        current_text = str(current["text"]).strip()

        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            next_text = str(nxt["text"]).strip()
            gap = int(nxt["start_char"]) - int(current["end_char"])

            if (
                len(current_text) <= max_short_len
                and next_text.startswith("(")
                and gap <= max_gap
            ):
                merged.append(
                    {
                        "text": f"{current_text} {next_text}",
                        "start_char": int(current["start_char"]),
                        "end_char": int(nxt["end_char"]),
                    }
                )
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def _remove_bad_chunks(chunks: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    cleaned: list[dict[str, int | str]] = []

    for chunk in chunks:
        text = str(chunk["text"]).strip()

        if _is_junk_chunk(text):
            continue

        # remove obvious inline source junk that survived normalization
        if re.search(r"source:\s+.*10-k", text, flags=re.IGNORECASE):
            continue

        # kill fake clause-reference fragments
        if re.match(r"^\d+\.\d+\s+and\s+\d+\.\d+,", text, flags=re.IGNORECASE):
            continue

        if re.match(r"^\([a-z]\)\s+to\s+clause\s+\d+\.\d+", text, flags=re.IGNORECASE):
            continue

        cleaned.append(chunk)

    return cleaned


# -------------------------------------------------------------------
# Public main API
# -------------------------------------------------------------------

def chunk_legal_text_with_offsets(
    text: str,
    max_chunk_chars: int = 350,
) -> list[dict[str, int | str]]:
    """
    Chunk legal text into clause-sized pieces while preserving offsets into the
    original extracted text.

    Main goals for this version:
    - keep agreement intro together
    - remove signature block junk
    - avoid false splits on clause references like "2.1 and 2.2"
    - split big clauses more aggressively
    - keep headers attached to the following clause when useful
    """
    if not text or not text.strip():
        return []

    blocks = _build_blocks_from_lines(text)

    raw_chunks: list[dict[str, int | str]] = []
    for block in blocks:
        pieces = _split_block(block, max_chars=max_chunk_chars)
        for piece in pieces:
            chunk = _make_chunk(piece.text, piece.start, piece.end)
            if chunk is not None:
                raw_chunks.append(chunk)

    chunks = _merge_headers_with_following_content(raw_chunks)
    chunks = _merge_orphan_tail_chunks(chunks)
    chunks = _remove_bad_chunks(chunks)

    deduped: list[dict[str, int | str]] = []
    seen: set[tuple[int, int, str]] = set()

    for chunk in chunks:
        key = (
            int(chunk["start_char"]),
            int(chunk["end_char"]),
            str(chunk["text"]),
        )
        if key not in seen:
            deduped.append(chunk)
            seen.add(key)

    deduped.sort(key=lambda x: int(x["start_char"]))
    return deduped


def chunk_legal_text(text: str, max_chunk_chars: int = 350) -> list[str]:
    return [
        str(chunk["text"])
        for chunk in chunk_legal_text_with_offsets(text, max_chunk_chars=max_chunk_chars)
    ]