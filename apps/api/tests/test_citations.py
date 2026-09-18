"""
Unit tests for inline segment citations (Phase 4).

RED: extract_citations / strip_invalid_citations do not exist yet and the
grounding prompt has no citation rule — every test here must fail first.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.generator import (
    GROUNDING_SYSTEM_PROMPT,
    extract_citations,
    generate_grounded_answer,
    strip_invalid_citations,
)


def test_prompt_requires_inline_segment_citations():
    assert "[Segment" in GROUNDING_SYSTEM_PROMPT
    assert "never invent" in GROUNDING_SYSTEM_PROMPT.lower()


def test_extract_citations_finds_refs_in_order():
    assert extract_citations("Refunds take ages [Segment 2]. See also [Segment 1].") == [2, 1]
    assert extract_citations("No citations here.") == []
    assert extract_citations("") == []
    # Malformed refs are not citations.
    assert extract_citations("Segment 2 states this without brackets.") == []
    assert extract_citations("See [Segment x] for details.") == []


def test_strip_invalid_citations_keeps_valid_refs():
    answer = "Refunds are fast [Segment 1] and tracked [Segment 2]."
    cleaned, dropped = strip_invalid_citations(answer, valid_segments=2)
    assert cleaned == answer  # byte-identical when nothing is invalid
    assert dropped == []


def test_strip_invalid_citations_removes_unknown_segments():
    answer = "Refunds are fast [Segment 1] and free [Segment 9]."
    cleaned, dropped = strip_invalid_citations(answer, valid_segments=2)
    assert "[Segment 9]" not in cleaned
    assert "[Segment 1]" in cleaned
    assert dropped == [9]


def test_strip_invalid_citations_rejects_zero_and_tidies_spacing():
    cleaned, dropped = strip_invalid_citations("A fact [Segment 0] here.", valid_segments=3)
    assert "[Segment 0]" not in cleaned
    assert "  " not in cleaned
    assert dropped == [0]


def test_strip_invalid_citations_no_refs_is_noop():
    answer = "Plain answer without any references."
    assert strip_invalid_citations(answer, valid_segments=1) == (answer, [])


@pytest.mark.asyncio
async def test_generation_strips_hallucinated_segment_refs():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Refunds are fast [Segment 1] and free [Segment 5]."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    chunks = [{"filename": "doc.txt", "page": 1, "text": "Factual segment content."}]
    with patch("app.generation.generator.get_llm", return_value=mock_llm):
        answer = await generate_grounded_answer("Is there matching info?", chunks)
    assert "[Segment 5]" not in answer
    assert "[Segment 1]" in answer


@pytest.mark.asyncio
async def test_generation_keeps_valid_citations_untouched():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Refunds are fast [Segment 1]."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    chunks = [{"filename": "doc.txt", "page": 1, "text": "Factual segment content."}]
    with patch("app.generation.generator.get_llm", return_value=mock_llm):
        answer = await generate_grounded_answer("Is there matching info?", chunks)
    assert answer == "Refunds are fast [Segment 1]."
