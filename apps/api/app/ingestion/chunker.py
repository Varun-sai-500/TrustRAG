"""
TRUSTRAG — character-based text chunker with word-boundary windows.

Chunks document pages using sliding windows with configured size and overlap.
Window edges snap to whitespace so chunks never start/end mid-word (mid-word
cuts pollute BM25 sparse vectors and read as broken fragments in evidence).
Snapping is coverage-safe: the end snaps back at most `overlap` characters
(the next window still overlaps) and the start only advances over characters
the previous window already covered — no silent text loss.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Max characters to scan back for a word boundary when snapping a window edge.
_WORD_BOUNDARY_LOOKBACK = 64


def chunk_text(
    pages: list[dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[dict[str, Any]]:
    """
    Split page texts into overlapping character chunks.

    Each chunk records:
      - text: string content of the chunk
      - page: page number it belongs to
      - chunk_index: integer index of chunk in document
      - character_offset: start character index in page
    """
    chunks = []
    chunk_index = 0

    # Guard against a misconfigured chunk_overlap >= chunk_size, which would make
    # the step size zero or negative and hang the loop below forever.
    step = chunk_size - chunk_overlap
    if step <= 0:
        logger.warning(
            "chunk_overlap >= chunk_size, forcing minimum step to avoid infinite loop",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        step = max(1, chunk_size)

    from app.ingestion.preprocessor import detect_chunk_zone, normalize_text

    for page_obj in pages:
        page_num = page_obj["page"]
        raw_text = page_obj.get("text", "")
        text = normalize_text(raw_text)

        if not text.strip():
            continue

        length = len(text)
        start = 0
        prev_end: int | None = None

        # Slide character window
        while start < length:
            # Snap start forward past a leading word fragment, but only over
            # characters the previous window already covered (no silent loss).
            if start > 0 and prev_end is not None and not text[start].isspace():
                frag_end = start
                while frag_end < prev_end and not text[frag_end].isspace():
                    frag_end += 1
                if frag_end <= prev_end and frag_end < length and text[frag_end].isspace():
                    start = frag_end + 1
            if start >= length:
                break

            end = min(start + chunk_size, length)
            # Snap end back to a word boundary so chunks never cut mid-word.
            # Bounded by the overlap so the next window still overlaps (no gaps);
            # overlong tokens (URLs, hashes) keep the hard cut as fallback.
            if end < length and not text[end].isspace():
                max_snap = min(_WORD_BOUNDARY_LOOKBACK, chunk_overlap)
                low = max(start, end - max_snap)
                for i in range(end - 1, low - 1, -1):
                    if text[i].isspace():
                        end = i
                        break

            chunk_content = text[start:end].strip()

            if chunk_content:
                zone = detect_chunk_zone(chunk_content, page=page_num)
                chunks.append(
                    {
                        "text": chunk_content,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "character_offset": start,
                        "zone": zone,
                        # Provenance: OCR fallback flags ride page → chunk.
                        "ocr_used": bool(page_obj.get("ocr_used", False)),
                        "ocr_confidence": page_obj.get("ocr_confidence"),
                    }
                )
                chunk_index += 1

            # Check termination
            if end >= length:
                break

            # Slide by step size (size - overlap)
            prev_end = end
            start += step

    return chunks
