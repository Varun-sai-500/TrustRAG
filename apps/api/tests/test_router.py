"""
Unit tests for the deterministic query router + fan-out merge (Phase 6).

RED: app.agent.router does not exist — every import here fails first.
Router contract (no LLM anywhere on this path):
- SIMPLE: today's single hybrid call, byte-identical kwargs.
- TEMPORAL: single call + explicit reference_time when the query names a year.
- COMPARISON: "A vs B" → two parallel retrievals, merged.
- COMPLEX: multi-"?" → deterministic split, capped at max_sub_queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from app.agent.router import (
    QueryRoute,
    fanout_retrieve,
    merge_fanout_results,
    route_query,
)


def _chunk(id_: str, rrf: float, text: str = "evidence text") -> dict:
    return {"id": id_, "rrf_score": rrf, "text": text}


# ── Classification ───────────────────────────────────────────────────────────


def test_simple_factual_query_takes_todays_path():
    routed = route_query("How long do I have to request a refund?")
    assert routed.route == QueryRoute.SIMPLE
    assert routed.sub_queries == ["How long do I have to request a refund?"]
    assert routed.reference_time is None


def test_single_question_mark_stays_simple():
    routed = route_query("What is the Pro plan price?")
    assert routed.route == QueryRoute.SIMPLE


def test_temporal_year_sets_reference_time():
    routed = route_query("What was the Pro plan price in 2025?")
    assert routed.route == QueryRoute.TEMPORAL
    assert routed.sub_queries == ["What was the Pro plan price in 2025?"]
    assert routed.reference_time == datetime(2025, 7, 1, tzinfo=UTC)


def test_temporal_keyword_without_year_keeps_caller_time():
    routed = route_query("What is the current Pro plan price?")
    assert routed.route == QueryRoute.TEMPORAL
    assert routed.reference_time is None


def test_comparison_vs_splits_into_two_sub_queries():
    routed = route_query("How much is the Pro vs Team plan?")
    assert routed.route == QueryRoute.COMPARISON
    assert routed.sub_queries == ["How much is the Pro", "Team plan"]


def test_comparison_difference_between_splits():
    routed = route_query("What is the difference between Pro and Team storage?")
    assert routed.route == QueryRoute.COMPARISON
    assert routed.sub_queries == ["Pro", "Team storage"]


def test_unsubstantiated_comparison_falls_back_to_simple():
    routed = route_query("Compare these plans for me")
    assert routed.route == QueryRoute.SIMPLE
    assert routed.sub_queries == ["Compare these plans for me"]


def test_multi_question_splits_and_caps_sub_queries():
    routed = route_query(
        "What is the refund window? How fast are refunds paid? Where do I send the request? "
        "Is express delivery available?",
        max_sub_queries=3,
    )
    assert routed.route == QueryRoute.COMPLEX
    assert len(routed.sub_queries) == 3
    assert routed.sub_queries[0] == "What is the refund window"


# ── Merge ────────────────────────────────────────────────────────────────────


def test_merge_dedupes_by_id_keeping_best_rrf_and_sorts_desc():
    first = [_chunk("a", 0.03), _chunk("b", 0.02)]
    second = [_chunk("b", 0.05), _chunk("c", 0.01)]
    merged = merge_fanout_results([first, second])
    assert [c["id"] for c in merged] == ["b", "a", "c"]
    assert [c["rrf_score"] for c in merged] == [0.05, 0.03, 0.01]
    assert merge_fanout_results([]) == []
    assert merge_fanout_results([[], []]) == []


# ── Fan-out execution ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fanout_runs_sub_queries_concurrently_and_merges():
    async def fake_hybrid(query, **kwargs):
        return [_chunk(f"hit-{query}", 0.02, text=f"text for {query}")]

    with patch("app.agent.router.retrieve_hybrid_chunks", new=AsyncMock(side_effect=fake_hybrid)):
        merged = await fanout_retrieve(
            ["Pro plan", "Team plan"], {"kb_id": "kb1", "top_k_override": None}
        )
    assert len(merged) == 2
    assert {c["id"] for c in merged} == {"hit-Pro plan", "hit-Team plan"}


@pytest.mark.asyncio
async def test_fanout_degrades_to_healthy_branch_on_partial_outage():
    from app.core.exceptions import RetrievalOutageError

    async def flaky(query, **kwargs):
        if query == "bad side":
            raise RetrievalOutageError("sparse branch down")
        return [_chunk("good-hit", 0.02)]

    with patch("app.agent.router.retrieve_hybrid_chunks", new=AsyncMock(side_effect=flaky)):
        merged = await fanout_retrieve(["bad side", "good side"], {"kb_id": "kb1"})
    assert [c["id"] for c in merged] == ["good-hit"]


@pytest.mark.asyncio
async def test_fanout_total_outage_stays_an_outage():
    from app.core.exceptions import RetrievalOutageError

    with patch(
        "app.agent.router.retrieve_hybrid_chunks",
        new=AsyncMock(side_effect=RetrievalOutageError("all down")),
    ):
        with pytest.raises(RetrievalOutageError):
            await fanout_retrieve(["a side", "b side"], {"kb_id": "kb1"})


# ── Node wiring ──────────────────────────────────────────────────────────────


@patch("app.agent.graph.add_trace_event", AsyncMock())
@patch("app.agent.graph.audit_evidence_integrity")
@patch("app.agent.graph.rerank_candidate_chunks")
@patch("app.agent.router.retrieve_hybrid_chunks")
@patch("app.agent.graph.get_collection")
@pytest.mark.asyncio
async def test_retrieval_node_fans_out_comparison_query(
    mock_collection, mock_retrieve, mock_rerank, mock_audit
):
    from app.agent.graph import retrieval_node

    async def per_side(query, **kwargs):
        return [
            {
                "id": f"chunk-{query}",
                "text": f"segment for {query}",
                "document_id": "64ee39d09c6292376e191981",
                "rrf_score": 0.02,
            }
        ]

    mock_retrieve.side_effect = per_side
    mock_rerank.side_effect = lambda query, chunks, **kw: chunks
    mock_audit.side_effect = lambda chunks: [{**c, "integrity_status": "VERIFIED"} for c in chunks]
    mock_db = MagicMock()
    mock_db.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=ObjectId("64ee39d09c6292376e191985"))
    )
    mock_collection.return_value = mock_db

    state = {
        "analysis_id": "64ee39d09c6292376e191983",
        "kb_id": "64ee39d09c6292376e191982",
        "query": "Pro vs Team?",
        "current_query": "Pro vs Team?",
        "answer": None,
        "chunks": [],
        "evidence_ids": [],
        "attempts": 0,
        "verdict_status": "FAIL",
        "recovery_strategy": None,
    }
    res = await retrieval_node(state)
    assert mock_retrieve.await_count == 2
    assert len(res["chunks"]) == 2
    assert len(res["evidence_ids"]) == 2
