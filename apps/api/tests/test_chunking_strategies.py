"""
Unit tests for Phase-3 chunking repair.

Covers: newline preservation in normalization, strategy wiring defaults,
semantic section splits + true offsets, progressive gap-freedom,
layout table grouping + order, and OCR provenance passthrough.
All deterministic; no live services.
"""

from __future__ import annotations

import pytest

from app.ingestion import chunking_strategies as strategies
from app.ingestion.chunker import chunk_text
from app.ingestion.preprocessor import detect_chunk_zone, lexical_analyze, normalize_text

# ── Normalization ────────────────────────────────────────────────────────────


def test_normalize_preserves_paragraph_breaks():
    assert (
        normalize_text("Title here\n\nBody text  with   spaces")
        == "title here\n\nbody text with spaces"
    )
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_lexer_output_identical_for_newline_vs_space():
    assert lexical_analyze("refund\nwindow", stem=True) == lexical_analyze(
        "refund window", stem=True
    )


def test_header_zone_detects_markdown_heading_on_normalized_text():
    assert detect_chunk_zone("## refund policy\nsome body text here") == "header"


# ── Strategy selection ───────────────────────────────────────────────────────


def test_default_strategy_is_sliding_window_and_delegates():
    strategies.clear_chunking_strategy()
    strategy = strategies.get_chunking_strategy()
    assert isinstance(strategy, strategies.SlidingWindowStrategy)
    pages = [{"page": 1, "text": "word " * 300}]
    assert strategy.chunk(pages, chunk_size=200, chunk_overlap=20) == chunk_text(
        pages, chunk_size=200, chunk_overlap=20
    )


def test_unknown_strategy_name_falls_back_to_sliding():
    assert isinstance(strategies._create_strategy("nope"), strategies.SlidingWindowStrategy)
    assert isinstance(strategies._create_strategy("semantic"), strategies.SemanticChunkingStrategy)


# ── Semantic strategy ────────────────────────────────────────────────────────


def test_semantic_splits_markdown_sections_with_true_offsets():
    text = (
        "# Refunds\n\nFull refund within 30 days of delivery. "
        + ("Details follow here. " * 40)
        + "\n\n# Shipping\n\nOrders ship within 5 days. "
        + ("More shipping notes. " * 40)
    )
    pages = [{"page": 1, "text": text}]
    chunks = strategies.SemanticChunkingStrategy().chunk(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    offsets = [c["character_offset"] for c in chunks]
    # Strictly increasing, traceable offsets — the old code reset ALL to 0.
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert max(offsets) > 0
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["page"] == 1 for c in chunks)
    # Both sections are represented in the output.
    joined = " ".join(c["text"] for c in chunks)
    assert "refund within 30 days" in joined
    assert "ship within 5 days" in joined


def test_semantic_single_section_matches_sliding_boundaries():
    pages = [{"page": 2, "text": "plain prose without any headings. " * 60}]
    semantic = strategies.SemanticChunkingStrategy().chunk(pages, chunk_size=200, chunk_overlap=20)
    sliding = chunk_text(pages, chunk_size=200, chunk_overlap=20)
    assert [c["text"] for c in semantic] == [c["text"] for c in sliding]
    assert [c["character_offset"] for c in semantic] == [c["character_offset"] for c in sliding]


# ── Progressive strategy ─────────────────────────────────────────────────────


def test_progressive_covers_full_text_without_gaps():
    tokens = [f"w{i:04d}" for i in range(300)]
    pages = [{"page": 1, "text": " ".join(tokens)}]
    chunks = strategies.ProgressiveChunkingStrategy().chunk(pages, chunk_size=200, chunk_overlap=20)
    covered = set()
    for c in chunks:
        covered.update(c["text"].split())
    missing = [t for t in tokens if t not in covered]
    assert missing == [], f"{len(missing)} tokens silently dropped (e.g. {missing[:5]})"


def test_progressive_windows_grow():
    pages = [{"page": 1, "text": "word " * 500}]
    chunks = strategies.ProgressiveChunkingStrategy().chunk(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks[0]["text"]) < len(chunks[-1]["text"])


# ── Layout strategy ──────────────────────────────────────────────────────────


def _layout_page() -> list[dict]:
    return [
        {
            "page": 1,
            "text": (
                "Intro line one.\n"
                "Intro line two.\n"
                "| Plan | Price |\n"
                "| Pro | 25 dollars |\n"
                "| Team | 60 dollars |\n"
                "Closing remark here."
            ),
        }
    ]


def test_layout_groups_table_rows_and_preserves_order():
    chunks = strategies.LayoutAwareChunkingStrategy().chunk(
        _layout_page(), chunk_size=500, chunk_overlap=20
    )
    table_chunks = [c for c in chunks if c["zone"] == "table"]
    assert len(table_chunks) == 1  # consecutive rows chunked ONCE, not row-by-row
    table_text = table_chunks[0]["text"]
    assert "pro" in table_text and "team" in table_text and "25 dollars" in table_text
    # Page order preserved: intro → table → closing.
    positions = {c["text"][:12]: i for i, c in enumerate(chunks)}
    assert positions["intro line o"] < positions[table_chunks[0]["text"][:12]]
    assert positions[table_chunks[0]["text"][:12]] < positions["closing rema"]
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_layout_table_chunks_share_sequential_indices():
    chunks = strategies.LayoutAwareChunkingStrategy().chunk(
        _layout_page(), chunk_size=60, chunk_overlap=10
    )
    table_chunks = [c for c in chunks if c["zone"] == "table"]
    assert len(table_chunks) >= 1
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)


# ── OCR provenance passthrough ───────────────────────────────────────────────


@pytest.mark.parametrize("strategy_name", ["semantic", "progressive", "layout_aware"])
def test_strategies_propagate_ocr_flags(strategy_name):
    pages = [
        {
            "page": 3,
            "text": "scanned content here. " * 100,
            "ocr_used": True,
            "ocr_confidence": 0.87,
        }
    ]
    strategy = strategies._create_strategy(strategy_name)
    chunks = strategy.chunk(pages, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(c["ocr_used"] is True for c in chunks)
    assert all(c["ocr_confidence"] == 0.87 for c in chunks)
