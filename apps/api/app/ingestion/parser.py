"""
TRUSTRAG — Document parsers for PDF, DOCX, TXT, MD, CSV, JSON, and HTML files.

Multi-format extraction, encoding detection (chardet), and temporal validity metadata auditing.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, BinaryIO

import chardet
import defusedxml.ElementTree as ET  # noqa: N817
import pymupdf as fitz  # PyMuPDF

from app.core.exceptions import IngestionError, UnsupportedFormatError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ISO format matching: YYYY-MM-DD
DATE_PATTERN_FROM = re.compile(r"effective\s+from:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
DATE_PATTERN_UNTIL = re.compile(r"effective\s+until:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

# Magic bytes (file signatures) for format validation
# Maps extension -> list of valid magic byte prefixes
MAGIC_BYTES = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],  # ZIP-based (docx, xlsx, pptx)
    ".zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".gif": [b"GIF87a", b"GIF89a"],
    ".bmp": [b"BM"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
}

# Per-format max decompression ratio (compressed_size / decompressed_size) to prevent zip bombs
# PDF: ~10x typical; ZIP-based: ~100x max; images: minimal compression
MAX_DECOMPRESSION_RATIO = {
    ".pdf": 50,
    ".docx": 100,
    ".zip": 100,
    ".png": 10,
    ".jpg": 10,
    ".jpeg": 10,
    ".gif": 10,
    ".bmp": 2,
    ".tiff": 10,
}

DEFAULT_MAX_RATIO = 100


def validate_magic_bytes(filename: str, stream: BinaryIO) -> None:
    """
    Validate file signature (magic bytes) matches the declared extension.
    Reads minimal bytes from stream start; stream position is preserved.
    """
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in MAGIC_BYTES:
        return  # No signature check for this format

    expected_signatures = MAGIC_BYTES[ext]
    pos = stream.tell()
    try:
        header = stream.read(16)
        if not header:
            raise IngestionError("Empty file", detail=f"File '{filename}' has no content")
        for sig in expected_signatures:
            if header.startswith(sig):
                return  # Valid signature
        raise IngestionError(
            "File signature mismatch",
            detail=(
                f"File '{filename}' has extension '{ext}' but content "
                "does not match expected format"
            ),
        )
    finally:
        stream.seek(pos)


def check_decompression_bomb(filename: str, compressed_size: int, decompressed_size: int) -> None:
    """
    Check if decompression ratio exceeds safe threshold (zip bomb protection).
    """
    if compressed_size <= 0:
        return
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    max_ratio = MAX_DECOMPRESSION_RATIO.get(ext, DEFAULT_MAX_RATIO)
    actual_ratio = decompressed_size / compressed_size
    if actual_ratio > max_ratio:
        raise IngestionError(
            "Decompression bomb detected",
            detail=(
                f"File '{filename}' decompression ratio {actual_ratio:.1f}x exceeds "
                f"maximum {max_ratio}x for format '{ext}'"
            ),
        )


def extract_dates(text: str) -> tuple[datetime | None, datetime | None]:
    """
    Search text for metadata expressions indicating effective periods:
      - 'Effective from: YYYY-MM-DD'
      - 'Effective until: YYYY-MM-DD'
    """
    eff_from = None
    eff_until = None

    # Scan first 2000 characters for metadata headers
    header_snippet = text[:2000]

    match_from = DATE_PATTERN_FROM.search(header_snippet)
    if match_from:
        with contextlib.suppress(ValueError):
            eff_from = datetime.strptime(match_from.group(1), "%Y-%m-%d").replace(tzinfo=UTC)

    match_until = DATE_PATTERN_UNTIL.search(header_snippet)
    if match_until:
        with contextlib.suppress(ValueError):
            eff_until = datetime.strptime(match_until.group(1), "%Y-%m-%d").replace(tzinfo=UTC)

    return eff_from, eff_until


def parse_pdf(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse a PDF file page-by-page.
    Returns a list of dicts: [{"page": page_num, "text": page_text,
    "ocr_used": bool, "ocr_confidence": float | None}].

    Pages with sufficient native text keep native extraction. Pages below the
    native-text density threshold fall back to RapidOCR-ONNX (ingestion.ocr.*)
    when enabled. OCR failures fail open to whatever native text exists.
    """
    from app.core.config import get_model_config
    from app.ingestion import ocr as ocr_module

    cfg = get_model_config()
    try:
        with fitz.open(stream=stream.read(), filetype="pdf") as doc:
            pages = []
            for i, page in enumerate(doc):
                native_text = page.get_text().strip()
                text, ocr_used, ocr_confidence = native_text, False, None
                if cfg.ocr_enabled and ocr_module.should_ocr_page(
                    native_text, cfg.ocr_min_native_chars
                ):
                    try:
                        pix = page.get_pixmap(dpi=cfg.ocr_dpi)
                        ocr_result = ocr_module.ocr_image_bytes(
                            pix.tobytes("png"),
                            min_confidence=cfg.ocr_min_confidence,
                        )
                        text = ocr_result.text.strip()
                        ocr_used, ocr_confidence = True, ocr_result.confidence
                        logger.info(
                            "OCR fallback used for PDF page",
                            page=i + 1,
                            confidence=ocr_confidence,
                        )
                    except Exception as exc:
                        # Fail open: a broken OCR page must not kill ingestion
                        # of the whole document; keep whatever native text exists.
                        logger.warning(
                            "OCR fallback failed; keeping native page text",
                            page=i + 1,
                            error=str(exc),
                        )
                pages.append(
                    {
                        "page": i + 1,
                        "text": text,
                        "ocr_used": ocr_used,
                        "ocr_confidence": ocr_confidence,
                    }
                )
            return pages
    except Exception as exc:
        raise IngestionError("Failed to parse PDF document", detail=str(exc)) from exc


def parse_docx(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse a Microsoft Word (.docx) file extracting paragraph text.
    Extracts XML from the ZIP container without requiring external C libraries.
    """
    # Check for zip bomb before parsing
    stream.seek(0, 2)
    compressed_size = stream.tell()
    stream.seek(0)
    try:
        with zipfile.ZipFile(stream) as docx_zip:
            # Check for zip bomb
            total_uncompressed = sum(info.file_size for info in docx_zip.infolist())
            check_decompression_bomb("document.docx", compressed_size, total_uncompressed)

            xml_content = docx_zip.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", namespaces):
                texts = [node.text for node in p.iterfind(".//w:t", namespaces) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            full_text = "\n\n".join(paragraphs)
            return [{"page": 1, "text": full_text.strip()}]
    except Exception as exc:
        raise IngestionError("Failed to parse DOCX document", detail=str(exc)) from exc


def parse_csv(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse a CSV document formatting each row as structured key-value lines.
    """
    raw_bytes = stream.read()
    if not raw_bytes:
        return [{"page": 1, "text": ""}]

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"

    try:
        text_content = raw_bytes.decode(encoding, errors="replace")
        reader = csv.reader(io.StringIO(text_content))
        rows = list(reader)
        if not rows:
            return [{"page": 1, "text": ""}]

        header = [h.strip() for h in rows[0]]
        lines = []
        for row in rows[1:]:
            if not any(val.strip() for val in row):
                continue
            row_str = " | ".join(
                f"{header[i] if i < len(header) else f'Col{i + 1}'}: {val.strip()}"
                for i, val in enumerate(row)
            )
            lines.append(row_str)

        formatted_text = "\n".join(lines) if lines else " | ".join(header)
        return [{"page": 1, "text": formatted_text.strip()}]
    except Exception as exc:
        raise IngestionError("Failed to parse CSV document", detail=str(exc)) from exc


def parse_json(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse a JSON document into indented, semantic text representations.
    """
    raw_bytes = stream.read()
    if not raw_bytes:
        return [{"page": 1, "text": ""}]

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"

    try:
        text_content = raw_bytes.decode(encoding, errors="replace")
        data = json.loads(text_content)
        formatted_text = json.dumps(data, indent=2, ensure_ascii=False)
        return [{"page": 1, "text": formatted_text.strip()}]
    except Exception as exc:
        raise IngestionError("Failed to parse JSON document", detail=str(exc)) from exc


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.pieces: list[str] = []
        self.ignore = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "meta", "noscript"):
            self.ignore = True
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self.pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "meta", "noscript"):
            self.ignore = False

    def handle_data(self, data: str) -> None:
        if not self.ignore and data.strip():
            self.pieces.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self.pieces)


def parse_html(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse an HTML document, stripping tags and extracting readable text content.
    """
    raw_bytes = stream.read()
    if not raw_bytes:
        return [{"page": 1, "text": ""}]

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"

    try:
        content = raw_bytes.decode(encoding, errors="replace")
        extractor = _HTMLTextExtractor()
        extractor.feed(content)
        cleaned = extractor.get_text()
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return [{"page": 1, "text": cleaned.strip()}]
    except Exception as exc:
        raise IngestionError("Failed to parse HTML document", detail=str(exc)) from exc


def parse_txt_or_md(stream: BinaryIO) -> list[dict[str, Any]]:
    """
    Parse a text or Markdown file with automatic encoding detection.
    """
    raw_bytes = stream.read()
    if not raw_bytes:
        return [{"page": 1, "text": ""}]

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"

    try:
        text = raw_bytes.decode(encoding, errors="replace")
        return [{"page": 1, "text": text.strip()}]
    except Exception as exc:
        raise IngestionError(
            f"Failed to decode text file using encoding '{encoding}'", detail=str(exc)
        ) from exc


def parse_document(
    filename: str, stream: BinaryIO
) -> tuple[list[dict[str, Any]], datetime | None, datetime | None]:
    """
    Determine format and parse document bytes across all supported extensions.
    Extracts temporal validity metadata if present.
    """
    # Validate magic bytes before parsing
    validate_magic_bytes(filename, stream)

    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if ext == ".pdf":
        pages = parse_pdf(stream)
    elif ext in (".txt", ".md"):
        pages = parse_txt_or_md(stream)
    elif ext == ".docx":
        pages = parse_docx(stream)
    elif ext == ".csv":
        pages = parse_csv(stream)
    elif ext == ".json":
        pages = parse_json(stream)
    elif ext in (".html", ".htm"):
        pages = parse_html(stream)
    else:
        raise UnsupportedFormatError(f"Unsupported file format '{ext}' during ingestion")

    # Combine text snippet to extract dates
    full_text = "\n".join(p["text"] for p in pages)
    eff_from, eff_until = extract_dates(full_text)

    return pages, eff_from, eff_until
