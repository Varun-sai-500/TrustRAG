"""
Unit tests for index lifecycle: snapshots, rollback, delete purge (Phase 8).

RED: snapshot/rollback routes do not exist; rollback has no empty-snapshot
guard; snapshot chunk copies drop OCR provenance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.main import app

client = TestClient(app)

USER_ID = "64ee39d09c6292376e191981"
KB_ID = "64ee39d09c6292376e191982"
SNAP_ID = "64ee39d09c6292376e191983"


@pytest.fixture
def mock_user_doc():
    return {
        "_id": ObjectId(USER_ID),
        "email": "test@example.com",
        "hashed_password": "hashed-stuff",
        "full_name": "Test User",
        "is_active": True,
        "created_at": "2026-08-27T10:00:00Z",
    }


@pytest.fixture(autouse=True)
def setup_dependency_override(mock_user_doc):
    app.dependency_overrides[get_current_user] = lambda: mock_user_doc
    yield
    app.dependency_overrides.clear()


def _kb_doc(kb_id: str, **overrides):
    doc = {
        "_id": ObjectId(kb_id),
        "name": "Refund Policies",
        "description": "Standard refund schedules",
        "user_id": ObjectId(USER_ID),
        "created_at": "2026-08-27T10:00:00Z",
        "version": "1.0",
        "parent_kb_id": None,
        "is_snapshot": False,
    }
    doc.update(overrides)
    return doc


# ── Routes ───────────────────────────────────────────────────────────────────


@patch("app.db.mongodb.connect_db")
@patch("app.db.mongodb.create_indexes")
def test_snapshot_route_end_to_end_shape(mock_create_indexes, mock_connect, mock_user_doc):
    """POST /knowledge-bases/{id}/snapshots → 201 with the snapshot KB payload."""
    snap_doc = _kb_doc(SNAP_ID, is_snapshot=True, parent_kb_id=ObjectId(KB_ID))
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(
        side_effect=[
            _kb_doc(KB_ID),  # get_kb ownership check
            _kb_doc(KB_ID),  # current_kb fetch
            snap_doc,  # (unused by create path, guards ordering)
        ]
    )
    mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId(SNAP_ID)))
    mock_collection.update_one = AsyncMock()
    mock_collection.count_documents = AsyncMock(return_value=0)

    mock_collection.find = MagicMock(side_effect=lambda *a, **k: _FakeCursor([]))
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=False)

    with (
        patch("app.services.kb_service.get_collection", return_value=mock_collection),
        patch("app.services.kb_service.get_qdrant_client", return_value=mock_qdrant),
    ):
        response = client.post(f"/api/v1/knowledge-bases/{KB_ID}/snapshots")
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == SNAP_ID
        assert data["is_snapshot"] is True


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def to_list(self, n):
        return self._rows


@patch("app.db.mongodb.connect_db")
@patch("app.db.mongodb.create_indexes")
def test_rollback_route_returns_new_live_id(mock_create_indexes, mock_connect):
    """POST /knowledge-bases/{id}/rollback/{snap} → 200; id CHANGES to the snapshot's."""
    live_doc = _kb_doc(KB_ID)
    snap_doc = _kb_doc(SNAP_ID, is_snapshot=True, parent_kb_id=ObjectId(KB_ID))
    promoted_doc = _kb_doc(SNAP_ID, is_snapshot=False, parent_kb_id=None)

    mock_collection = MagicMock()
    # get_kb(live) → #1; get_kb(snapshot) → #2; final get_kb(promoted) → #3.
    mock_collection.find_one = AsyncMock(side_effect=[live_doc, snap_doc, promoted_doc])
    mock_collection.count_documents = AsyncMock(return_value=0)
    mock_collection.delete_many = AsyncMock()
    mock_collection.delete_one = AsyncMock()
    mock_collection.update_one = AsyncMock()

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete_collection = AsyncMock()
    mock_qdrant.count = AsyncMock(return_value=MagicMock(count=5))

    with (
        patch("app.services.kb_service.get_collection", return_value=mock_collection),
        patch("app.services.kb_service.get_qdrant_client", return_value=mock_qdrant),
        # delete_kb_collection() resolves the client from the qdrant module's
        # own namespace — patch it too or the test hits real Qdrant.
        patch("app.db.qdrant.get_qdrant_client", return_value=mock_qdrant),
    ):
        response = client.post(f"/api/v1/knowledge-bases/{KB_ID}/rollback/{SNAP_ID}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == SNAP_ID
        assert data["is_snapshot"] is False


@patch("app.db.mongodb.connect_db")
@patch("app.db.mongodb.create_indexes")
def test_rollback_route_rejects_foreign_snapshot_with_409(mock_create_indexes, mock_connect):
    live_doc = _kb_doc(KB_ID)
    foreign_snap = _kb_doc(
        "64ee39d09c6292376e191990",
        is_snapshot=True,
        parent_kb_id=ObjectId("64ee39d09c6292376e191900"),
    )
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(side_effect=[live_doc, foreign_snap])
    mock_collection.count_documents = AsyncMock(return_value=0)

    with patch("app.services.kb_service.get_collection", return_value=mock_collection):
        response = client.post(
            "/api/v1/knowledge-bases/64ee39d09c6292376e191982/rollback/64ee39d09c6292376e191990"
        )
        assert response.status_code == 409


# ── Service: empty-snapshot guard ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_refuses_snapshot_without_vectors():
    """Pre-vector-copy snapshots roll back to an empty KB → 409, not silent loss."""
    from app.services import kb_service

    live_doc = _kb_doc(KB_ID)
    snap_doc = _kb_doc(SNAP_ID, is_snapshot=True, parent_kb_id=ObjectId(KB_ID))
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(side_effect=[live_doc, snap_doc])
    # Snapshot holds Mongo chunks (2) but ZERO Qdrant points.
    mock_collection.count_documents = AsyncMock(return_value=2)
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.count = AsyncMock(return_value=MagicMock(count=0))

    with (
        patch.object(kb_service, "get_collection", return_value=mock_collection),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
    ):
        from app.core.exceptions import ConflictError

        with pytest.raises(ConflictError, match="no searchable vectors"):
            await kb_service.rollback_kb_to_snapshot(KB_ID, SNAP_ID, USER_ID)


# ── Service: snapshot preserves OCR provenance ───────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_chunk_copies_keep_ocr_provenance():
    """Chunk copies must carry ocr_used/ocr_confidence or snapshots lose the chain."""
    from app.services import kb_service

    live_doc = _kb_doc(KB_ID)
    ocr_chunk = {
        "_id": ObjectId(),
        "document_id": ObjectId("64ee39d09c6292376e191999"),
        "knowledge_base_id": ObjectId(KB_ID),
        "user_id": ObjectId(USER_ID),
        "chunk_index": 0,
        "text": "scanned refund text",
        "page": 2,
        "character_offset": 0,
        "zone": "body",
        "text_hash": "abc",
        "ocr_used": True,
        "ocr_confidence": 0.87,
    }
    live_text_doc = {
        "_id": ocr_chunk["document_id"],
        "user_id": ObjectId(USER_ID),
        "knowledge_base_id": ObjectId(KB_ID),
        "filename": "scan.pdf",
        "file_size": 10,
        "content_hash": "def",
        "ingestion_status": "completed",
    }
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=live_doc)
    mock_collection.find = MagicMock(
        side_effect=[_FakeCursor([live_text_doc]), _FakeCursor([ocr_chunk])]
    )
    mock_collection.count_documents = AsyncMock(return_value=1)
    inserted: list[dict] = []

    async def capture_insert(doc):
        inserted.append(doc)
        return MagicMock(inserted_id=ObjectId())

    mock_collection.insert_one = AsyncMock(side_effect=capture_insert)
    mock_collection.update_one = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=False)

    with (
        patch.object(kb_service, "get_collection", return_value=mock_collection),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
    ):
        await kb_service.create_kb_snapshot(KB_ID, USER_ID, version="1.1")

    chunk_copies = [d for d in inserted if d.get("text") == "scanned refund text"]
    assert len(chunk_copies) == 1
    assert chunk_copies[0]["ocr_used"] is True
    assert chunk_copies[0]["ocr_confidence"] == 0.87


# ── Delete purges vectors (no-orphan proof) ──────────────────────────────────


@pytest.mark.asyncio
async def test_delete_document_purges_qdrant_points_by_document_id():
    """Document delete must remove its Qdrant points or stale evidence is served."""
    from app.services import kb_service

    doc_id = "64ee39d09c6292376e191999"
    mock_doc = {
        "_id": ObjectId(doc_id),
        "knowledge_base_id": ObjectId(KB_ID),
        "filename": "policy.pdf",
    }
    mock_kb = _kb_doc(KB_ID)
    mock_coll = MagicMock()
    mock_coll.find_one = AsyncMock(side_effect=[mock_doc, mock_kb])
    mock_coll.count_documents = AsyncMock(return_value=1)
    mock_coll.delete_many = AsyncMock()
    mock_coll.delete_one = AsyncMock()

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists = AsyncMock(return_value=True)
    mock_qdrant.delete = AsyncMock()

    with (
        patch.object(kb_service, "get_collection", return_value=mock_coll),
        patch.object(kb_service, "get_qdrant_client", return_value=mock_qdrant),
    ):
        await kb_service.delete_document(doc_id, USER_ID)

    mock_qdrant.delete.assert_awaited_once()
    selector = mock_qdrant.delete.call_args.kwargs["points_selector"]
    matched_values = [
        cond.match.value for cond in selector.filter.must if cond.key == "document_id"
    ]
    assert matched_values == [doc_id]
    mock_coll.delete_many.assert_awaited_once()
    mock_coll.delete_one.assert_awaited_once()
