"""
Unit tests for the per-page OCR fallback (RapidOCR-ONNX).

The real engine is NEVER initialized here (no model downloads): all engine
interactions go through app.ingestion.ocr._load_engine / ocr_image_bytes
seams and are patched. Real PyMuPDF is used to build tiny in-memory PDFs.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest
from bson import ObjectId

from app.ingestion import ocr as ocr_module
from app.ingestion.chunker import chunk_text
from app.ingestion.ocr import OCRPageResult, ocr_image_bytes, should_ocr_page
from app.ingestion.parser import parse_pdf


def _pdf_bytes(*page_texts: str):
    """Build an in-memory PDF; one page per text (empty string = blank/scanned)."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    return doc.tobytes()


def _fake_engine(result):
    engine = MagicMock()
    engine.return_value = (result, [0.01])
    return engine


# ── Routing gate ─────────────────────────────────────────────────────────────


def test_should_ocr_page_density_gate():
    assert should_ocr_page("", 50) is True
    assert should_ocr_page("   ", 50) is True
    assert should_ocr_page("x" * 49, 50) is True
    assert should_ocr_page("x" * 50, 50) is False
    assert should_ocr_page("a" * 500, 50) is False


# ── Engine wrapper ───────────────────────────────────────────────────────────


def test_ocr_image_bytes_joins_lines_and_averages_confidence():
    fake = _fake_engine([[None, "Hello", 0.9], [None, "World", 0.8]])
    with patch.object(ocr_module, "_load_engine", return_value=fake):
        res = ocr_image_bytes(b"png-bytes")
    assert res == OCRPageResult(text="Hello\nWorld", confidence=pytest.approx(0.85), used=True)
    fake.assert_called_once_with(b"png-bytes")


def test_ocr_image_bytes_drops_low_confidence_text_but_marks_used():
    fake = _fake_engine([[None, "garbage~", 0.2], [None, "noise", 0.3]])
    with patch.object(ocr_module, "_load_engine", return_value=fake):
        res = ocr_image_bytes(b"png-bytes", min_confidence=0.5)
    assert res.text == ""
    assert res.confidence == pytest.approx(0.25)
    assert res.used is True


def test_ocr_image_bytes_empty_result():
    fake = _fake_engine(None)
    with patch.object(ocr_module, "_load_engine", return_value=fake):
        res = ocr_image_bytes(b"png-bytes")
    assert res == OCRPageResult(text="", confidence=None, used=True)


def test_ocr_image_bytes_skips_malformed_entries():
    fake = _fake_engine(["not-a-triplet", [None, "  ok  ", 0.9], [None, "", 0.9]])
    with patch.object(ocr_module, "_load_engine", return_value=fake):
        res = ocr_image_bytes(b"png-bytes")
    assert res.text == "ok"
    assert res.confidence == pytest.approx(0.9)


def test_ocr_image_bytes_propagates_missing_engine():
    with patch.object(ocr_module, "_load_engine", side_effect=RuntimeError("not installed")):
        with pytest.raises(RuntimeError, match="not installed"):
            ocr_image_bytes(b"png-bytes")


# ── parse_pdf routing ────────────────────────────────────────────────────────


def test_parse_pdf_native_page_skips_ocr():
    pages = parse_pdf(io.BytesIO(_pdf_bytes("Hello native world " * 10)))
    assert len(pages) == 1
    assert "hello native world" in pages[0]["text"].lower()
    assert pages[0]["ocr_used"] is False
    assert pages[0]["ocr_confidence"] is None


def test_parse_pdf_scanned_page_uses_ocr():
    ocr_result = OCRPageResult(text="scanned hello", confidence=0.9, used=True)
    with patch.object(ocr_module, "ocr_image_bytes", return_value=ocr_result) as mock_ocr:
        pages = parse_pdf(io.BytesIO(_pdf_bytes("")))
    assert len(pages) == 1
    assert pages[0]["text"] == "scanned hello"
    assert pages[0]["ocr_used"] is True
    assert pages[0]["ocr_confidence"] == 0.9
    mock_ocr.assert_called_once()


def test_parse_pdf_mixed_pages_route_independently():
    ocr_result = OCRPageResult(text="scanned hello", confidence=0.9, used=True)
    with patch.object(ocr_module, "ocr_image_bytes", return_value=ocr_result):
        pages = parse_pdf(io.BytesIO(_pdf_bytes("Hello native world " * 10, "")))
    assert [p["ocr_used"] for p in pages] == [False, True]
    assert pages[1]["text"] == "scanned hello"


def test_parse_pdf_ocr_failure_fails_open_to_native_text():
    with patch.object(ocr_module, "ocr_image_bytes", side_effect=Exception("engine down")):
        pages = parse_pdf(io.BytesIO(_pdf_bytes("ab")))
    assert pages[0]["text"] == "ab"
    assert pages[0]["ocr_used"] is False


def test_parse_pdf_dense_native_page_never_calls_engine():
    from app.core.config import get_model_config

    assert get_model_config().ocr_enabled is True  # default on; gate below is explicit
    with (
        patch.object(ocr_module, "ocr_image_bytes") as mock_ocr,
        patch.object(ocr_module, "should_ocr_page", return_value=False),
    ):
        pages = parse_pdf(io.BytesIO(_pdf_bytes("")))
    assert pages[0]["ocr_used"] is False
    mock_ocr.assert_not_called()


# ── Chunker + pipeline propagation ───────────────────────────────────────────


def test_chunker_propagates_ocr_provenance():
    pages = [
        {"page": 1, "text": "native " * 200, "ocr_used": False, "ocr_confidence": None},
        {"page": 2, "text": "scanned " * 200, "ocr_used": True, "ocr_confidence": 0.87},
    ]
    chunks = chunk_text(pages)
    by_page = {}
    for c in chunks:
        by_page.setdefault(c["page"], []).append(c)
    assert all(c["ocr_used"] is False and c["ocr_confidence"] is None for c in by_page[1])
    assert all(c["ocr_used"] is True and c["ocr_confidence"] == 0.87 for c in by_page[2])


def test_chunker_defaults_when_flags_absent():
    chunks = chunk_text([{"page": 1, "text": "legacy page " * 200}])
    assert chunks
    assert all(c["ocr_used"] is False and c["ocr_confidence"] is None for c in chunks)


@pytest.mark.asyncio
async def test_pipeline_payload_carries_ocr_provenance():
    from app.ingestion.pipeline import index_parsed_chunks

    mock_client = MagicMock()
    mock_client.upsert = AsyncMock()
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": ObjectId("64ee39d09c6292376e191983"),
            "user_id": ObjectId("64ee39d09c6292376e191981"),
        }
    )
    mock_collection.update_one = AsyncMock()
    mock_collection.insert_many = AsyncMock()
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 384])

    chunks = [
        {
            "text": "scanned chunk",
            "page": 2,
            "chunk_index": 0,
            "character_offset": 0,
            "zone": "body",
            "ocr_used": True,
            "ocr_confidence": 0.87,
        },
    ]
    with (
        patch("app.ingestion.pipeline.init_kb_collection", AsyncMock()),
        patch("app.ingestion.pipeline.get_embedding_model", return_value=mock_embeddings),
        patch("app.ingestion.pipeline.get_collection", return_value=mock_collection),
        patch("app.ingestion.pipeline.get_qdrant_client", AsyncMock(return_value=mock_client)),
    ):
        await index_parsed_chunks(
            doc_id_str="64ee39d09c6292376e191983",
            kb_id_str="64ee39d09c6292376e191982",
            chunks=chunks,
        )

    mongo_doc = mock_collection.insert_many.call_args.args[0][0]
    assert mongo_doc["ocr_used"] is True
    assert mongo_doc["ocr_confidence"] == 0.87
    point = mock_client.upsert.call_args.kwargs["points"][0]
    assert point.payload["ocr_used"] is True
    assert point.payload["ocr_confidence"] == 0.87


# ── Config defaults ──────────────────────────────────────────────────────────


def test_ocr_config_defaults():
    from app.core.config import get_model_config

    cfg = get_model_config()
    assert cfg.ocr_enabled is True
    assert cfg.ocr_min_native_chars == 50
    assert cfg.ocr_dpi == 300
    assert cfg.ocr_min_confidence == 0.5
    assert cfg.as_snapshot()["ocr_enabled"] is True
