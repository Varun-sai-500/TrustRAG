"""
Unit tests for the hybrid dense + sparse retriever.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import RetrievalOutageError
from app.retrieval.retriever import (
    apply_temporal_filtering,
    dense_search,
    reciprocal_rank_fusion,
    retrieve_hybrid_chunks,
    sparse_search,
)


def test_reciprocal_rank_fusion_logic():
    # Mock ScoredPoint structures from Qdrant
    dense_point = MagicMock()
    dense_point.id = "point-1"
    dense_point.score = 0.9
    dense_point.payload = {"text": "dense text content", "document_id": "doc-1"}

    sparse_point_1 = MagicMock()
    sparse_point_1.id = "point-1"
    sparse_point_1.score = 0.5
    sparse_point_1.payload = {"text": "dense text content", "document_id": "doc-1"}

    sparse_point_2 = MagicMock()
    sparse_point_2.id = "point-2"
    sparse_point_2.score = 0.8
    sparse_point_2.payload = {"text": "sparse text content", "document_id": "doc-2"}

    dense_results = [dense_point]
    sparse_results = [sparse_point_1, sparse_point_2]

    # Fused with k=60
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    assert len(fused) == 2
    # point-1 was ranked 1st in dense and 1st in sparse, so it must be first
    assert fused[0]["id"] == "point-1"
    # RRF score calculation: 1/(1+60) + 1/(1+60) = 2/61 = ~0.0327
    assert abs(fused[0]["rrf_score"] - (2.0 / 61.0)) < 1e-5

    # point-2 was ranked 2nd in sparse and not found in dense
    assert fused[1]["id"] == "point-2"
    # RRF score calculation: 1/(2+60) = 1/62 = ~0.0161
    assert abs(fused[1]["rrf_score"] - (1.0 / 62.0)) < 1e-5


@pytest.mark.asyncio
async def test_collection_dimension_is_cached_per_collection():
    import app.retrieval.retriever as retriever

    collection_name = "kb_dimension_cache_test"
    retriever._collection_dimension_cache.clear()
    vectors = SimpleNamespace(size=384)
    col_info = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))
    client = SimpleNamespace(get_collection=AsyncMock(return_value=col_info))

    first = await retriever._get_collection_dimension(client, collection_name)
    second = await retriever._get_collection_dimension(client, collection_name)

    assert first == 384
    assert second == 384
    client.get_collection.assert_awaited_once_with(collection_name)


@pytest.mark.asyncio
async def test_temporal_validity_filtering():
    results = [
        {"document_id": "64ee39d09c6292376e191981", "text": "active chunk"},
        {"document_id": "64ee39d09c6292376e191982", "text": "expired chunk"},
        {"document_id": "64ee39d09c6292376e191983", "text": "future chunk"},
    ]

    ref_time = datetime(2026, 8, 1, tzinfo=UTC)

    # Document mock outputs from MongoDB
    mock_docs = [
        {
            "_id": "64ee39d09c6292376e191981",
            "filename": "active.txt",
            "effective_from": datetime(2026, 7, 1, tzinfo=UTC),
            "effective_until": datetime(2026, 9, 1, tzinfo=UTC),
        },
        {
            "_id": "64ee39d09c6292376e191982",
            "filename": "expired.txt",
            "effective_from": datetime(2026, 6, 1, tzinfo=UTC),
            "effective_until": datetime(2026, 7, 15, tzinfo=UTC),
        },
        {
            "_id": "64ee39d09c6292376e191983",
            "filename": "future.txt",
            "effective_from": datetime(2026, 8, 15, tzinfo=UTC),
            "effective_until": datetime(2026, 9, 15, tzinfo=UTC),
        },
    ]

    # Mock cursor
    mock_cursor = MagicMock()

    async def mock_async_gen():
        for d in mock_docs:
            yield d

    mock_cursor.__aiter__ = MagicMock(side_effect=mock_async_gen)

    mock_collection = MagicMock()
    mock_collection.find = MagicMock(return_value=mock_cursor)

    with patch("app.retrieval.retriever.get_collection", return_value=mock_collection):
        filtered = await apply_temporal_filtering(results, ref_time)

        # Only doc-active fits (2026-08-01 lies between 2026-07-01 and 2026-09-01)
        assert len(filtered) == 1
        assert filtered[0]["document_id"] == "64ee39d09c6292376e191981"


@pytest.mark.asyncio
async def test_dense_search_raises_outage_when_qdrant_unavailable():
    import app.retrieval.retriever as retriever

    retriever._query_cache._cache.clear()
    with patch(
        "app.retrieval.retriever.get_qdrant_client", side_effect=Exception("connection refused")
    ):
        with pytest.raises(RetrievalOutageError, match="Vector store unavailable"):
            await dense_search("outage probe query alpha", "kb_outage_1")


@pytest.mark.asyncio
async def test_dense_search_returns_empty_for_genuine_no_evidence():
    import app.retrieval.retriever as retriever

    retriever._query_cache._cache.clear()
    mock_client = SimpleNamespace(query_points=AsyncMock(return_value=SimpleNamespace(points=[])))
    mock_embed = MagicMock()
    mock_embed.embed_query = MagicMock(return_value=[0.1] * 8)

    with (
        patch("app.retrieval.retriever.get_qdrant_client", return_value=mock_client),
        patch("app.retrieval.retriever.get_embedding_model", return_value=mock_embed),
        patch(
            "app.retrieval.retriever._get_collection_dimension",
            AsyncMock(return_value=8),
        ),
    ):
        result = await dense_search("no evidence query beta", "kb_empty_1")
        assert result == []


@pytest.mark.asyncio
async def test_dense_search_raises_outage_when_query_fails():
    import app.retrieval.retriever as retriever

    retriever._query_cache._cache.clear()
    mock_client = SimpleNamespace(query_points=AsyncMock(side_effect=Exception("connection reset")))
    mock_embed = MagicMock()
    mock_embed.embed_query = MagicMock(return_value=[0.1] * 8)

    with (
        patch("app.retrieval.retriever.get_qdrant_client", return_value=mock_client),
        patch("app.retrieval.retriever.get_embedding_model", return_value=mock_embed),
        patch(
            "app.retrieval.retriever._get_collection_dimension",
            AsyncMock(return_value=8),
        ),
    ):
        with pytest.raises(RetrievalOutageError, match="query failed"):
            await dense_search("outage probe query gamma", "kb_outage_2")


@pytest.mark.asyncio
async def test_sparse_search_returns_empty_when_no_indexable_tokens():
    mock_client = SimpleNamespace(query_points=AsyncMock())
    with (
        patch("app.retrieval.retriever.get_qdrant_client", return_value=mock_client),
        patch(
            "app.retrieval.retriever.generate_sparse_vector",
            return_value={"indices": [], "values": []},
        ),
    ):
        result = await sparse_search("the and or", "kb_empty_2")
        assert result == []
        mock_client.query_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_sparse_search_raises_outage_when_query_fails():
    mock_client = SimpleNamespace(query_points=AsyncMock(side_effect=Exception("Qdrant timed out")))
    with (
        patch("app.retrieval.retriever.get_qdrant_client", return_value=mock_client),
        patch(
            "app.retrieval.retriever.generate_sparse_vector",
            return_value={"indices": [7], "values": [0.5]},
        ),
    ):
        with pytest.raises(RetrievalOutageError, match="query failed"):
            await sparse_search("outage probe query delta", "kb_outage_3")


@pytest.mark.asyncio
async def test_retrieve_hybrid_chunks_propagates_outage_not_empty():
    with (
        patch(
            "app.retrieval.retriever.dense_search",
            AsyncMock(side_effect=RetrievalOutageError("Vector store unavailable: down")),
        ),
        patch("app.retrieval.retriever.sparse_search", AsyncMock(return_value=[])),
    ):
        with pytest.raises(RetrievalOutageError):
            await retrieve_hybrid_chunks("outage probe query epsilon", "kb_outage_4")


@pytest.mark.asyncio
async def test_hybrid_degrades_to_healthy_branch_on_timeout(monkeypatch):
    """One hung branch must not discard the healthy branch's results."""
    from app.retrieval import retriever

    monkeypatch.setattr(retriever, "RETRIEVAL_BRANCH_TIMEOUT", 0.05)

    async def slow_dense(*args, **kwargs):
        await asyncio.sleep(60)
        return []

    sparse_point = MagicMock()
    sparse_point.id = "sparse-1"
    sparse_point.score = 0.7
    sparse_point.payload = {"text": "sparse hit"}
    with (
        patch("app.retrieval.retriever.dense_search", AsyncMock(side_effect=slow_dense)),
        patch("app.retrieval.retriever.sparse_search", AsyncMock(return_value=[sparse_point])),
    ):
        res = await retrieve_hybrid_chunks("degraded branch probe", "kb_degraded_1")

    assert len(res) == 1
    assert res[0]["text"] == "sparse hit"


@pytest.mark.asyncio
async def test_hybrid_both_branches_timeout_is_outage(monkeypatch):
    """Both branches hung → hard outage, never silent 'no evidence'."""
    from app.retrieval import retriever

    monkeypatch.setattr(retriever, "RETRIEVAL_BRANCH_TIMEOUT", 0.05)

    async def slow(*args, **kwargs):
        await asyncio.sleep(60)
        return []

    with (
        patch("app.retrieval.retriever.dense_search", AsyncMock(side_effect=slow)),
        patch("app.retrieval.retriever.sparse_search", AsyncMock(side_effect=slow)),
    ):
        with pytest.raises(RetrievalOutageError, match="both"):
            await retrieve_hybrid_chunks("outage probe query zeta", "kb_outage_5")


@pytest.mark.asyncio
async def test_hybrid_enforces_fusion_top_k():
    """Fused candidates are truncated to retrieval.fusion_top_k (default 20)."""
    from app.core.config import get_model_config

    fusion_top_k = get_model_config().fusion_top_k
    assert fusion_top_k == 20

    def make_points(prefix: str, count: int) -> list:
        points = []
        for i in range(count):
            point = MagicMock()
            point.id = f"{prefix}-{i}"
            point.score = 1.0 - (i * 0.01)
            point.payload = {"text": f"{prefix} hit {i}"}
            points.append(point)
        return points

    with (
        patch(
            "app.retrieval.retriever.dense_search",
            AsyncMock(return_value=make_points("d", 25)),
        ),
        patch(
            "app.retrieval.retriever.sparse_search",
            AsyncMock(return_value=make_points("s", 25)),
        ),
    ):
        res = await retrieve_hybrid_chunks("fusion bound probe", "kb_fusion_1")

    # 25 dense + 25 sparse (disjoint) fuse to 50; all single-list hits score
    # 1/(rank+60), so the top 20 are exactly ranks 1..10 of each branch.
    assert len(res) == fusion_top_k
    assert {r["id"] for r in res} == {f"d-{i}" for i in range(10)} | {f"s-{i}" for i in range(10)}
