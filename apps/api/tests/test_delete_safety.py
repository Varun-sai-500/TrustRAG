"""
Critical delete-path safety tests (fail-closed deletes, no orphaned vectors).

RED: delete_document swallows Qdrant errors (orphaned points keep serving
evidence for deleted docs); delete_kb deletes Mongo BEFORE Qdrant (a Qdrant
failure leaves metadata gone but vectors live); router cap < 2 silently
mismatches route label and executed query.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

USER_ID = "64ee39d09c6292376e191981"
KB_ID = "64ee39d09c6292376e191982"
DOC_ID = "64ee39d09c6292376e191999"


def _kb_doc():
    return {
        "_id": ObjectId(KB_ID),
        "name": "Refund Policies",
        "description": "d",
        "user_id": ObjectId(USER_ID),
        "created_at": "2026-08-27T10:00:00Z",
        "version": "1.0",
        "parent_kb_id": None,
        "is_snapshot": False,
    }


@pytest.mark.asyncio
async def test_delete_document_qdrant_failure_keeps_metadata():
    """A failed vector delete must NOT silently orphan servable points.

    Fail closed: the error propagates, the doc record (and its chunks, already
    integrity-excluded once Mongo chunks are gone) stays for a safe retry —
    never "deleted" metadata with live vectors behind it.
    """
    from app.services import kb_service

    mock_doc = {
        "_id": ObjectId(DOC_ID),
        "knowledge_base_id": ObjectId(KB_ID),
        "filename": "policy.pdf",
    }
    mock_coll = MagicMock()
    mock_coll.find_one = AsyncMock(side_effect=[mock_doc, _kb_doc()])
    mock_coll.count_documents = AsyncMock(return_value=1)
    mock_coll.delete_many = AsyncMock()
    mock_coll.delete_one = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete = AsyncMock(side_effect=Exception("Qdrant down"))

    with (
        patch.object(kb_service, "get_collection", return_value=mock_coll),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
    ):
        with pytest.raises(Exception, match="Qdrant down"):
            await kb_service.delete_document(DOC_ID, USER_ID)

    # The document record must survive for a retry — never metadata-gonepoints-live.
    mock_coll.delete_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_kb_drops_vectors_before_metadata():
    """Qdrant drop runs FIRST so a failure leaves metadata intact for retry.

    Mongo-first ordering would delete docs/chunks/records and then fail on
    vectors — leaving a live KB record whose Qdrant collection still serves
    evidence for "deleted" documents.
    """
    from app.services import kb_service

    order: list[str] = []
    mock_coll = MagicMock()
    mock_coll.find_one = AsyncMock(return_value=_kb_doc())
    mock_coll.count_documents = AsyncMock(return_value=1)

    async def record_delete_many(*args, **kwargs):
        order.append("mongo")

    async def record_delete_one(*args, **kwargs):
        order.append("mongo-record")

    mock_coll.delete_many = AsyncMock(side_effect=record_delete_many)
    mock_coll.delete_one = AsyncMock(side_effect=record_delete_one)

    async def record_drop(*args, **kwargs):
        order.append("qdrant")

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete_collection = AsyncMock(side_effect=record_drop)

    with (
        patch.object(kb_service, "get_collection", return_value=mock_coll),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
        patch("app.db.qdrant.get_qdrant_client", return_value=mock_qdrant),
        patch("app.core.semantic_cache.invalidate_semantic_cache"),
    ):
        await kb_service.delete_kb(KB_ID, USER_ID)

    assert order[0] == "qdrant"
    assert set(order) == {"qdrant", "mongo", "mongo-record"}


@pytest.mark.asyncio
async def test_delete_kb_qdrant_failure_keeps_all_metadata():
    """If the vector drop fails, NOTHING metadata-side may be deleted."""
    from app.services import kb_service

    mock_coll = MagicMock()
    mock_coll.find_one = AsyncMock(return_value=_kb_doc())
    mock_coll.count_documents = AsyncMock(return_value=1)
    mock_coll.delete_many = AsyncMock()
    mock_coll.delete_one = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete_collection = AsyncMock(side_effect=Exception("Qdrant down"))

    with (
        patch.object(kb_service, "get_collection", return_value=mock_coll),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
        patch("app.db.qdrant.get_qdrant_client", return_value=mock_qdrant),
        patch("app.core.semantic_cache.invalidate_semantic_cache"),
    ):
        with pytest.raises(Exception, match="Failed to drop"):
            await kb_service.delete_kb(KB_ID, USER_ID)

    mock_coll.delete_many.assert_not_awaited()
    mock_coll.delete_one.assert_not_awaited()


def test_router_cap_below_two_falls_back_to_simple():
    """Capping fan-out below 2 is meaningless — run the full query, not half."""
    from app.agent.router import QueryRoute, route_query

    routed = route_query("Pro vs Team plan?", max_sub_queries=1)
    assert routed.route == QueryRoute.SIMPLE
    assert routed.sub_queries == ["Pro vs Team plan?"]
