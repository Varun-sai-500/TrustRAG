"""
TRUSTRAG — Knowledge Base and Document management business logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from qdrant_client.http import models

from app.api.v1.schemas.kb import DocResponse, KBCreate, KBResponse
from app.core.exceptions import AuthorizationError, NotFoundError
from app.core.logging import get_logger
from app.db.mongodb import Collections, get_collection
from app.db.qdrant import delete_kb_collection, get_collection_name, get_qdrant_client

logger = get_logger(__name__)


def serialize_kb(kb_doc: Mapping[str, Any], doc_count: int = 0) -> KBResponse:
    """Helper to convert MongoDB KB document to Pydantic KBResponse."""
    # Extract version info
    version = kb_doc.get("version", "1.0")
    parent_kb_id = kb_doc.get("parent_kb_id")
    is_snapshot = kb_doc.get("is_snapshot", False)

    return KBResponse(
        id=str(kb_doc["_id"]),
        name=kb_doc["name"],
        description=kb_doc.get("description", ""),
        user_id=str(kb_doc["user_id"]),
        document_count=doc_count,
        created_at=kb_doc["created_at"],
        version=version,
        parent_kb_id=str(parent_kb_id) if parent_kb_id else None,
        is_snapshot=is_snapshot,
        embedding_model=kb_doc.get("embedding_model"),
        embedding_provider=kb_doc.get("embedding_provider"),
        embedding_dim=kb_doc.get("embedding_dim"),
    )


def serialize_doc(doc_doc: Mapping[str, Any]) -> DocResponse:
    """Helper to convert MongoDB Document document to Pydantic DocResponse."""
    return DocResponse(
        id=str(doc_doc["_id"]),
        knowledge_base_id=str(doc_doc["knowledge_base_id"]),
        filename=doc_doc["filename"],
        file_size=doc_doc["file_size"],
        content_hash=doc_doc["content_hash"],
        ingestion_status=doc_doc.get("ingestion_status", "pending"),
        error_message=doc_doc.get("error_message"),
        effective_from=doc_doc.get("effective_from"),
        effective_until=doc_doc.get("effective_until"),
        created_at=doc_doc["created_at"],
    )


async def create_kb(schema: KBCreate, user_id: str) -> KBResponse:
    """Create a new knowledge base linked to user."""
    kb_coll = get_collection(Collections.KNOWLEDGE_BASES)
    kb_doc = {
        "name": schema.name.strip(),
        "description": schema.description.strip(),
        "user_id": ObjectId(user_id),
        "created_at": datetime.now(UTC),
        "version": "1.0",
        "parent_kb_id": None,
        "is_snapshot": False,
    }
    result = await kb_coll.insert_one(kb_doc)
    kb_doc["_id"] = result.inserted_id
    return serialize_kb(kb_doc, 0)


async def get_kb(kb_id_str: str, user_id_str: str) -> KBResponse:
    """
    Retrieve knowledge base by ID and verify ownership.

    Raises NotFoundError or AuthorizationError on security check failure.
    """
    try:
        kb_id = ObjectId(kb_id_str)
    except Exception as exc:
        raise NotFoundError("Knowledge Base not found", detail=str(exc)) from exc

    kb = await get_collection(Collections.KNOWLEDGE_BASES).find_one({"_id": kb_id})
    if not kb:
        raise NotFoundError("Knowledge Base not found")

    if str(kb["user_id"]) != user_id_str:
        raise AuthorizationError("Access denied", detail="You do not own this knowledge base")

    # Count documents
    doc_count = await get_collection(Collections.DOCUMENTS).count_documents(
        {"knowledge_base_id": kb_id}
    )

    return serialize_kb(kb, doc_count)


async def list_kbs(user_id_str: str) -> list[KBResponse]:
    """List all knowledge bases owned by user with document counts in a single aggregation."""
    kb_coll = get_collection(Collections.KNOWLEDGE_BASES)
    doc_coll = get_collection(Collections.DOCUMENTS)

    # Fetch all KBs owned by user in one query
    user_obj_id = ObjectId(user_id_str)
    kbs_raw = await kb_coll.find({"user_id": user_obj_id}).sort("created_at", -1).to_list(500)

    if not kbs_raw:
        return []

    # Batch-fetch document counts using a single aggregation (avoids N+1 per-KB count queries)
    kb_ids = [kb["_id"] for kb in kbs_raw]
    count_pipeline = [
        {"$match": {"knowledge_base_id": {"$in": kb_ids}}},
        {"$group": {"_id": "$knowledge_base_id", "count": {"$sum": 1}}},
    ]
    count_map: dict = {}
    async for row in doc_coll.aggregate(count_pipeline):
        count_map[row["_id"]] = row["count"]

    return [serialize_kb(kb, count_map.get(kb["_id"], 0)) for kb in kbs_raw]


async def delete_kb(kb_id_str: str, user_id_str: str) -> None:
    """
    Delete a knowledge base and all associated documents.

    Verifies ownership before deleting.
    This is an irreversible hard delete: Mongo documents/chunks, the Qdrant
    vector collection, and the KB record are all removed. Snapshots follow the
    same path via their own ids.
    """
    # Ensure KB exists and belongs to the user
    kb = await get_kb(kb_id_str, user_id_str)
    kb_id = ObjectId(kb_id_str)

    # Vectors FIRST: a failed drop must leave all metadata intact for retry.
    # Mongo-first ordering would delete records and then fail on vectors,
    # leaving a live KB whose collection still serves "deleted" evidence.
    # (Snapshots follow the same path via their own ids.)
    await delete_kb_collection(kb_id_str)

    # 1. Delete associated documents in MongoDB
    await get_collection(Collections.DOCUMENTS).delete_many({"knowledge_base_id": kb_id})

    # 2. Delete associated document chunks in MongoDB
    await get_collection(Collections.DOCUMENT_CHUNKS).delete_many({"knowledge_base_id": kb_id})

    # 3. Delete the KB record itself
    await get_collection(Collections.KNOWLEDGE_BASES).delete_one({"_id": kb_id})

    # Cached answers must never outlive the evidence that produced them.
    from app.core.semantic_cache import invalidate_semantic_cache

    invalidate_semantic_cache(kb_id_str)
    logger.info(
        "KB permanently deleted with all associated data",
        kb_id=kb_id_str,
        was_snapshot=kb.is_snapshot,
    )


async def add_document(
    kb_id_str: str,
    filename: str,
    file_size: int,
    content_hash: str,
    user_id_str: str,
    effective_from: datetime | None = None,
    effective_until: datetime | None = None,
    version: str = "1.0",
    is_snapshot: bool = False,
) -> DocResponse:
    """
    Add a document metadata record. Verifies KB ownership first.
    Returns 409 Conflict if a document with the same content_hash already exists in the KB.
    """
    import pymongo.errors

    # Verify owner
    await get_kb(kb_id_str, user_id_str)

    doc_coll = get_collection(Collections.DOCUMENTS)
    doc_doc = {
        "user_id": ObjectId(user_id_str),
        "knowledge_base_id": ObjectId(kb_id_str),
        "filename": filename,
        "file_size": file_size,
        "content_hash": content_hash,
        "ingestion_status": "pending",
        "error_message": None,
        "effective_from": effective_from,
        "effective_until": effective_until,
        "created_at": datetime.now(UTC),
        "version": version,
        "is_snapshot": is_snapshot,
    }

    try:
        result = await doc_coll.insert_one(doc_doc)
        doc_doc["_id"] = result.inserted_id

        # New evidence can change the best answer for an already cached query.
        from app.core.semantic_cache import invalidate_semantic_cache

        invalidate_semantic_cache(kb_id_str)
        return serialize_doc(doc_doc)
    except pymongo.errors.DuplicateKeyError as exc:
        if "doc_kb_content_hash_unique" in str(exc):
            from app.core.exceptions import ConflictError

            raise ConflictError(
                "Document with identical content already exists in this knowledge base",
                detail=f"content_hash: {content_hash}",
            ) from exc
        raise


async def list_kb_documents(kb_id_str: str, user_id_str: str) -> list[DocResponse]:
    """List all documents registered in a knowledge base."""
    # Verify owner
    await get_kb(kb_id_str, user_id_str)

    doc_coll = get_collection(Collections.DOCUMENTS)
    docs = []
    async for d in doc_coll.find({"knowledge_base_id": ObjectId(kb_id_str)}).sort("created_at", -1):
        docs.append(serialize_doc(d))
    return docs


async def create_kb_snapshot(kb_id_str: str, user_id_str: str, version: str = "1.0") -> KBResponse:
    """Create a snapshot/version of a knowledge base for rollback capability.

    Creates a new KB that is a copy of the current state, with version tracking.
    The original KB remains unchanged, and the snapshot can be used for rollback.
    """
    # Verify owner
    await get_kb(kb_id_str, user_id_str)

    # Get current KB details
    kb_coll = get_collection(Collections.KNOWLEDGE_BASES)
    current_kb = await kb_coll.find_one({"_id": ObjectId(kb_id_str)})

    if not current_kb:
        from app.core.exceptions import NotFoundError

        raise NotFoundError("Knowledge Base not found")

    # Create snapshot KB with incremented version
    snapshot_name = f"{current_kb['name']} - Snapshot {version}"
    snapshot_desc = f"{current_kb.get('description', '')} (Snapshot: {version})"

    snapshot_doc = {
        "name": snapshot_name,
        "description": snapshot_desc,
        "user_id": current_kb["user_id"],
        "created_at": datetime.now(UTC),
        "version": version,
        "parent_kb_id": ObjectId(kb_id_str),
        "is_snapshot": True,
    }

    result = await kb_coll.insert_one(snapshot_doc)
    snapshot_doc["_id"] = result.inserted_id
    snapshot_id_str = str(result.inserted_id)

    # Live doc id -> snapshot doc id, so chunk/vector references stay valid
    # after rollback (snapshot docs are re-inserted with fresh ObjectIds).
    doc_id_map: dict[str, str] = {}

    # Also snapshot the documents (copy document records with new IDs)
    doc_coll = get_collection(Collections.DOCUMENTS)
    existing_docs = await doc_coll.find(
        {"knowledge_base_id": ObjectId(kb_id_str), "is_snapshot": {"$ne": True}}
    ).to_list(10000)

    for existing_doc in existing_docs:
        # Create a copy of the document with a new ID, linked to the snapshot KB
        doc_copy = {
            "user_id": current_kb["user_id"],
            "knowledge_base_id": ObjectId(result.inserted_id),
            "filename": existing_doc["filename"],
            "file_size": existing_doc["file_size"],
            "content_hash": existing_doc["content_hash"],
            "ingestion_status": existing_doc.get("ingestion_status", "pending"),
            "error_message": existing_doc.get("error_message"),
            "effective_from": existing_doc.get("effective_from"),
            "effective_until": existing_doc.get("effective_until"),
            "created_at": datetime.now(UTC),
            "version": version,
            "is_snapshot": True,
        }
        doc_res = await doc_coll.insert_one(doc_copy)
        # Map live doc id -> snapshot doc id so chunk + vector references
        # can be remapped (snapshot docs get fresh ObjectIds).
        doc_id_map[str(existing_doc["_id"])] = str(doc_res.inserted_id)

    # Also copy document chunks
    chunks_coll = get_collection(Collections.DOCUMENT_CHUNKS)
    existing_chunks = await chunks_coll.find(
        {"knowledge_base_id": ObjectId(kb_id_str), "is_snapshot": {"$ne": True}}
    ).to_list(10000)

    for chunk in existing_chunks:
        remapped_doc_id = doc_id_map.get(str(chunk["document_id"]), str(chunk["document_id"]))
        chunk_copy = {
            "document_id": ObjectId(remapped_doc_id),
            "knowledge_base_id": ObjectId(result.inserted_id),
            "user_id": current_kb["user_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "page": chunk["page"],
            "character_offset": chunk["character_offset"],
            "zone": chunk.get("zone", "body"),
            "text_hash": chunk.get("text_hash"),
            "ocr_used": bool(chunk.get("ocr_used", False)),
            "ocr_confidence": chunk.get("ocr_confidence"),
            "is_snapshot": True,
        }
        await chunks_coll.insert_one(chunk_copy)

    # Copy Qdrant vectors into the snapshot collection so a later rollback
    # restores searchable state (Mongo copies alone would leave retrieval empty).
    await _copy_kb_vectors(kb_id_str, snapshot_id_str, doc_id_map)

    # Carry over the embedding-space pin so the guard keeps working post-rollback.
    if current_kb.get("embedding_model"):
        await kb_coll.update_one(
            {"_id": ObjectId(snapshot_id_str)},
            {
                "$set": {
                    "embedding_model": current_kb.get("embedding_model"),
                    "embedding_provider": current_kb.get("embedding_provider"),
                    "embedding_dim": current_kb.get("embedding_dim"),
                }
            },
        )

    return serialize_kb(snapshot_doc)


async def _copy_kb_vectors(source_kb_id: str, dest_kb_id: str, doc_id_map: dict[str, str]) -> None:
    """Duplicate all Qdrant points from one KB collection into another.

    Payloads are rewritten to the destination KB (and remapped snapshot doc
    ids); point ids are recomputed deterministically from the new doc ids so
    re-snapshotting stays idempotent.
    """
    from app.db.qdrant import get_collection_name, get_qdrant_client, init_kb_collection
    from app.ingestion.pipeline import hashlib_qdrant_id

    client = await get_qdrant_client()
    source_name = get_collection_name(source_kb_id)
    try:
        if not await client.collection_exists(source_name):
            return
    except Exception as exc:
        logger.warning("Snapshot vector copy skipped (collection check failed)", error=str(exc))
        return

    await init_kb_collection(dest_kb_id)
    dest_name = get_collection_name(dest_kb_id)

    offset: object = None
    total = 0
    while True:
        points, offset = await client.scroll(
            collection_name=source_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        batch = []
        for pt in points:
            payload = dict(pt.payload or {})
            live_doc_id = str(payload.get("document_id", ""))
            new_doc_id = doc_id_map.get(live_doc_id, live_doc_id)
            chunk_index = int(payload.get("chunk_index", 0))
            payload["document_id"] = new_doc_id
            payload["knowledge_base_id"] = dest_kb_id
            batch.append(
                models.PointStruct(
                    id=hashlib_qdrant_id(new_doc_id, chunk_index),
                    vector=pt.vector,
                    payload=payload,
                )
            )
        if batch:
            await client.upsert(collection_name=dest_name, points=batch)
            total += len(batch)
        if offset is None:
            break
    logger.info("Snapshot vector copy completed", points=total, dest=dest_name)


async def rollback_kb_to_snapshot(
    kb_id_str: str, snapshot_kb_id_str: str, user_id_str: str
) -> KBResponse:
    """Rollback a knowledge base to a previous snapshot version.

    The snapshot KB record survives and becomes the live KB (keeping its own
    ObjectId); the pre-rollback live data is discarded. Documents and chunks
    already point at the snapshot record, so no repointing is required.

    NOTE: the restored KB keeps the *snapshot's* id, not the original live id.
    No API route exposes this yet; when one is added it must return the new id
    and clients must swap it. Snapshots taken before the vector-copy fix have
    no Qdrant points — rolling back to those yields an empty (re-upload) state.
    """
    # Verify owner of current KB and capture its identity before deleting it.
    current_kb = await get_kb(kb_id_str, user_id_str)

    # Get snapshot KB details
    snapshot_kb = await get_kb(snapshot_kb_id_str, user_id_str)

    if not snapshot_kb.is_snapshot:
        from app.core.exceptions import ConflictError

        raise ConflictError("Rollback target is not a snapshot of this knowledge base")

    if str(snapshot_kb.parent_kb_id) != kb_id_str:
        from app.core.exceptions import ConflictError

        raise ConflictError("Snapshot does not belong to this knowledge base")

    kb_id = ObjectId(kb_id_str)
    snapshot_kb_id = ObjectId(snapshot_kb_id_str)

    # 0. Refuse snapshots with no searchable vectors (taken before the
    # vector-copy fix): restoring those silently yields an empty KB while the
    # Mongo copies pretend everything is fine. Re-upload instead.
    snap_client = await get_qdrant_client()
    snap_points = 0
    try:
        snap_count = await snap_client.count(get_collection_name(snapshot_kb_id_str), exact=True)
        snap_points = snap_count.count
    except Exception as exc:
        logger.warning(
            "Snapshot vector check failed; proceeding without the guard",
            snapshot_id=snapshot_kb_id_str,
            error=str(exc),
        )
        snap_points = -1
    if snap_points == 0:
        snap_chunks = await get_collection(Collections.DOCUMENT_CHUNKS).count_documents(
            {"knowledge_base_id": snapshot_kb_id}
        )
        if snap_chunks > 0:
            from app.core.exceptions import ConflictError

            raise ConflictError(
                "Snapshot has no searchable vectors (predates the vector-copy fix); "
                "re-upload the documents instead of rolling back",
            )

    # 1. Delete current (live) KB data
    await get_collection(Collections.DOCUMENTS).delete_many({"knowledge_base_id": kb_id})
    await get_collection(Collections.DOCUMENT_CHUNKS).delete_many({"knowledge_base_id": kb_id})
    await delete_kb_collection(kb_id_str)
    await get_collection(Collections.KNOWLEDGE_BASES).delete_one({"_id": kb_id})

    # 2. Promote the snapshot record to live, restoring the original name.
    snapshot_kb_coll = get_collection(Collections.KNOWLEDGE_BASES)
    await snapshot_kb_coll.update_one(
        {"_id": snapshot_kb_id},
        {
            "$set": {
                "name": current_kb.name,
                "description": current_kb.description,
                "is_snapshot": False,
                "parent_kb_id": None,
                "version": current_kb.version,
            }
        },
    )

    # 3. Cached answers for both identities are stale after a rollback.
    from app.core.semantic_cache import invalidate_semantic_cache

    invalidate_semantic_cache(kb_id_str)
    invalidate_semantic_cache(snapshot_kb_id_str)

    # 4. Return the restored (formerly snapshot) KB.
    return await get_kb(snapshot_kb_id_str, user_id_str)


async def delete_document(doc_id_str: str, user_id_str: str) -> None:
    """
    Delete a document, all its chunks, and associated Qdrant vector points.

    Verifies ownership of the parent KB before deleting.
    """
    from app.core.exceptions import NotFoundError

    try:
        doc_id = ObjectId(doc_id_str)
    except Exception as exc:
        raise NotFoundError("Document not found", detail=str(exc)) from exc

    doc_coll = get_collection(Collections.DOCUMENTS)
    doc = await doc_coll.find_one({"_id": doc_id})
    if not doc:
        raise NotFoundError("Document not found")

    # Verify ownership
    kb_id_str = str(doc["knowledge_base_id"])
    await get_kb(kb_id_str, user_id_str)

    # 1. Delete chunks in MongoDB
    await get_collection(Collections.DOCUMENT_CHUNKS).delete_many({"document_id": doc_id})

    # 2. Delete points from Qdrant collection. Fail CLOSED on vector-store
    # errors: swallowing them here would delete the metadata below while the
    # points keep serving evidence for a "deleted" document. The record below
    # is only removed after vectors are gone, so a failure stays retry-safe.
    client = await get_qdrant_client()
    collection_name = get_collection_name(kb_id_str)
    if await client.collection_exists(collection_name):
        try:
            await client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=doc_id_str),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.error(
                "Qdrant point delete failed; document record kept for retry",
                doc_id=doc_id_str,
                error=str(exc),
            )
            raise

    # 3. Delete document record itself
    await doc_coll.delete_one({"_id": doc_id})

    from app.core.semantic_cache import invalidate_semantic_cache

    invalidate_semantic_cache(kb_id_str)
    logger.info("Document deleted successfully", doc_id=doc_id_str, kb_id=kb_id_str)
