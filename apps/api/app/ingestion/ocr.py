"""
TRUSTRAG — per-page OCR fallback (RapidOCR-ONNX).

Native text extraction stays the default. OCR runs ONLY for pages whose
native text is below a density threshold (scanned/image pages, mixed PDFs).
This keeps latency and error surface off the pages that don't need it.

Design notes:
- Engine import is lazy and the instance is process-wide: model weights load
  once on first OCR page, never at import time (keeps CLI/test startup fast
  and lets ingestion fail open when the optional dependency is missing).
- RapidOCR reuses the ONNX Runtime already shipped for BGE embeddings —
  no torch, no PaddlePaddle, no system binaries.
- Confidence is the mean line score. Pages below min_confidence contribute
  NO text (garbage must never become evidence) but still record ocr_used=True
  so the gap is auditable instead of silent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

_ENGINE = None
_ENGINE_ERROR: str | None = None
_ENGINE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class OCRPageResult:
    """Outcome of OCR-ing one rendered page image."""

    text: str
    confidence: float | None  # mean line score in 0..1; None when nothing recognized
    used: bool  # True whenever the engine ran, even on empty/low-confidence output


def should_ocr_page(native_text: str, min_native_chars: int) -> bool:
    """True when native extraction yielded too little text to trust."""
    return len((native_text or "").strip()) < min_native_chars


def _load_engine():
    """Import and instantiate RapidOCR (lazy; raises RuntimeError when unavailable)."""
    global _ENGINE, _ENGINE_ERROR
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _ENGINE_ERROR is not None:
            raise RuntimeError(f"OCR engine unavailable: {_ENGINE_ERROR}")
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:
            _ENGINE_ERROR = str(exc)
            raise RuntimeError(
                "rapidocr-onnxruntime is not installed; install it or disable ingestion.ocr.enabled"
            ) from exc
        try:
            _ENGINE = RapidOCR()
        except Exception as exc:
            _ENGINE_ERROR = str(exc)
            raise RuntimeError(f"Failed to initialize RapidOCR engine: {exc}") from exc
        return _ENGINE


def reset_engine_for_tests() -> None:
    """Clear the cached engine/error (tests only)."""
    global _ENGINE, _ENGINE_ERROR
    with _ENGINE_LOCK:
        _ENGINE = None
        _ENGINE_ERROR = None


def ocr_image_bytes(image_png: bytes, min_confidence: float = 0.5) -> OCRPageResult:
    """Run OCR over a rendered page image (PNG bytes).

    Returns joined line texts in reading order with mean confidence.
    Text below min_confidence is dropped (returned as "") — it must not
    become retrieval evidence — while used=True preserves auditability.
    """
    engine = _load_engine()
    result, _elapse = engine(image_png)
    if not result:
        return OCRPageResult(text="", confidence=None, used=True)

    lines: list[str] = []
    scores: list[float] = []
    for entry in result:
        try:
            text = str(entry[1]).strip()
            score = float(entry[2]) if len(entry) > 2 else None
        except (IndexError, TypeError, ValueError):
            continue
        if text:
            lines.append(text)
            if score is not None:
                scores.append(score)

    if not lines:
        return OCRPageResult(text="", confidence=None, used=True)

    confidence = sum(scores) / len(scores) if scores else None
    text = "\n".join(lines)
    if confidence is not None and confidence < min_confidence:
        logger.warning(
            "Dropping low-confidence OCR page text",
            confidence=round(confidence, 3),
            threshold=min_confidence,
        )
        return OCRPageResult(text="", confidence=confidence, used=True)
    return OCRPageResult(text=text, confidence=confidence, used=True)
