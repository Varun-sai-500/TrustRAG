"""
TRUSTRAG — Ingestion pipeline coordinator.

Generates dense and sparse embeddings, indexes points to Qdrant,
and updates document ingestion status in MongoDB.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from qdrant_client.http import models

from app.core.config import get_model_config
from app.core.logging import get_logger
from app.core.model_registry import get_embedding_model
from app.db.mongodb import Collections, get_collection
from app.db.qdrant import get_collection_name, get_qdrant_client, init_kb_collection
from app.ingestion.chunking_strategies import ChunkingStrategy
from app.ingestion.sparse_vector import generate_sparse_vector

logger = get_logger(__name__)

# Ingestion can run several CPU/embedding-heavy background jobs at once. Keep
# it serialized per event loop so uploads cannot starve the local model or API.
_INGESTION_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _get_ingestion_semaphore() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    semaphore = _INGESTION_SEMAPHORES.get(loop_id)
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
        _INGESTION_SEMAPHORES[loop_id] = semaphore
    return semaphore


async def _index_parsed_chunks(
    doc_id_str: str,
    kb_id_str: str,
    chunks: list[dict[str, Any]] | None = None,
    strategy: ChunkingStrategy | None = None,
) -> None:
    """
    Background task to generate embeddings and index chunks to Qdrant.

    Stages:
      1. Fetch document record, update status to 'processing'
      2. Ensure Qdrant collection 'kb_{kb_id}' exists
      3. For each chunk:
          - Generate dense embedding (sentence-transformers/all-MiniLM-L6-v2)
          - Generate sparse keyword weights
          - Construct Qdrant point
      4. Upsert points into Qdrant
      5. Update document status to 'completed'
    """
    doc_id = ObjectId(doc_id_str)
    doc_coll = get_collection(Collections.DOCUMENTS)

    # 1. Update status to processing
    await doc_coll.update_one(
        {"_id": doc_id},
        {"$set": {"ingestion_status": "processing", "updated_at": datetime.now(UTC)}},
    )

    try:
        if not chunks:
            await doc_coll.update_one({"_id": doc_id}, {"$set": {"ingestion_status": "completed"}})
            logger.info("Ingestion completed: document has no text chunks", doc_id=doc_id_str)
            return
        user_id = None
        doc_filename = "Document"
        doc = await doc_coll.find_one({"_id": doc_id})
        if doc:
            user_id = doc.get("user_id")
            doc_filename = doc.get("filename", "Document")

        # NOTE: chunking happens at upload time (knowledge_bases.py selects the
        # configured strategy via get_chunking_strategy()). The `strategy`
        # parameter is kept for backward compatibility and ignored here —
        # this stage only embeds and indexes the chunks it receives.

        # Store chunks in MongoDB for future integrity audits
        import hashlib

        chunks_coll = get_collection(Collections.DOCUMENT_CHUNKS)
        mongo_chunks = []
        for c in chunks:
            mongo_chunks.append(
                {
                    "document_id": doc_id,
                    "knowledge_base_id": ObjectId(kb_id_str),
                    "user_id": user_id,
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                    "page": c["page"],
                    "character_offset": c["character_offset"],
                    "zone": c.get("zone", "body"),
                    "text_hash": hashlib.sha256(c["text"].encode("utf-8")).hexdigest(),
                    "ocr_used": bool(c.get("ocr_used", False)),
                    "ocr_confidence": c.get("ocr_confidence"),
                }
            )
        if mongo_chunks:
            await chunks_coll.insert_many(mongo_chunks)

        # 2. Ensure Qdrant collection is initialized
        await init_kb_collection(kb_id_str)

        # 3. Load embedding model (cached)
        embed_model = get_embedding_model()
        cfg = get_model_config()

        # Zero-Cost Contextual Prefixing (Anthropic SOTA pattern):
        # Prepend document filename and zone to resolve chunk ambiguity without extra LLM cost
        contextual_texts = [
            f"[{doc_filename} | {c.get('zone', 'body').upper()}] {c['text']}" for c in chunks
        ]
        logger.info("Generating dense embeddings", doc_id=doc_id_str, count=len(contextual_texts))

        # Use async batch embedding (aembed_documents) for 2.87x speedup
        # The CachedEmbeddingsWrapper handles disk cache lookup and batching internally
        from app.core.hardware import get_ingest_embed_batch_size

        embed_batch_size = get_ingest_embed_batch_size()
        dense_vectors = []
        for offset in range(0, len(contextual_texts), embed_batch_size):
            batch_slice = contextual_texts[offset : offset + embed_batch_size]
            batch_vecs: list[list[float]] | None = None
            last_batch_err: Exception | None = None
            for attempt in range(5):
                try:
                    batch_vecs = await embed_model.aembed_documents(batch_slice)
                    break
                except Exception as batch_err:
                    last_batch_err = batch_err
                    err_msg = str(batch_err)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < 4:
                        wait_seconds = 32 if attempt >= 1 else 15
                        logger.warning(
                            "Embedding rate limit reached, waiting for quota reset",
                            attempt=attempt + 1,
                            wait_seconds=wait_seconds,
                        )
                        await asyncio.sleep(wait_seconds)
                    else:
                        raise batch_err
            if batch_vecs is None:
                # All retries exhausted on rate limits — fail loudly instead of
                # falling through with a short vector list (which would cause
                # a misleading IndexError below).
                raise last_batch_err or RuntimeError("Embedding batch failed without error")
            dense_vectors.extend(batch_vecs)

        qdrant_client = await get_qdrant_client()
        collection_name = get_collection_name(kb_id_str)

        # 4. Construct Qdrant points
        points = []
        for i, chunk in enumerate(chunks):
            # Compute sparse TF vector with zone weighting over contextual text
            chunk_zone = chunk.get("zone", "body")
            sparse_vec = generate_sparse_vector(contextual_texts[i], zone=chunk_zone)

            # Unique deterministic ID for Qdrant point (based on doc ID and chunk index)
            point_id = hashlib_qdrant_id(doc_id_str, chunk["chunk_index"])

            # Payload contains metadata + text + zone + OCR provenance
            payload = {
                "document_id": doc_id_str,
                "knowledge_base_id": kb_id_str,
                "user_id": str(user_id) if user_id else "",
                "chunk_index": chunk["chunk_index"],
                "page": chunk["page"],
                "character_offset": chunk["character_offset"],
                "zone": chunk_zone,
                "text": chunk["text"],
                "ocr_used": bool(chunk.get("ocr_used", False)),
                "ocr_confidence": chunk.get("ocr_confidence"),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        # Named vector configurations
                        "": dense_vectors[i],  # Default/dense
                        "sparse-text": models.SparseVector(  # Sparse BM25
                            indices=sparse_vec["indices"], values=sparse_vec["values"]
                        ),
                    },
                    payload=payload,
                )
            )

        # Incremental indexing: upsert only new/updated points
        # The deterministic point IDs based on (doc_id, chunk_index) ensure
        # existing chunks are updated in place rather than duplicated.
        # Batch upsert to prevent network timeouts
        batch_size = 100
        for offset in range(0, len(points), batch_size):
            batch = points[offset : offset + batch_size]
            await qdrant_client.upsert(collection_name=collection_name, points=batch)

        logger.info("Incremental indexing completed", doc_id=doc_id_str, chunks=len(points))

        # 5. Mark document completed
        await doc_coll.update_one({"_id": doc_id}, {"$set": {"ingestion_status": "completed"}})
        logger.info("Ingestion completed successfully", doc_id=doc_id_str, chunks=len(points))

        # 6. Pin the embedding space on the KB record so future analyses can
        # NEVER silently query these vectors with a different embedding model.
        # (Cross-space queries return plausible-looking garbage → recovery spiral.)
        # Pin-once: re-uploading one doc after a provider change must NOT
        # silently re-pin while older vectors stay in the old space.
        if dense_vectors:
            kb_coll = get_collection(Collections.KNOWLEDGE_BASES)
            existing_kb = await kb_coll.find_one({"_id": ObjectId(kb_id_str)})
            if existing_kb and existing_kb.get("embedding_model"):
                if existing_kb.get("embedding_model") != cfg.embedding_model:
                    logger.warning(
                        "Ingest uses a different embedding model than the KB pin; "
                        "keeping the original pin — re-upload into a NEW KB to migrate",
                        kb_pin=existing_kb.get("embedding_model"),
                        current=cfg.embedding_model,
                    )
            else:
                await kb_coll.update_one(
                    {"_id": ObjectId(kb_id_str)},
                    {
                        "$set": {
                            "embedding_model": cfg.embedding_model,
                            "embedding_provider": cfg.embedding_provider,
                            "embedding_dim": len(dense_vectors[0]),
                            "embedding_pinned_at": datetime.now(UTC),
                        }
                    },
                )

    except Exception as exc:
        logger.error("Ingestion pipeline failed", doc_id=doc_id_str, error=str(exc))
        # Store a generic error type — NOT str(exc), which can leak internal details
        # (file paths, connection strings, stack info) to the client via DocResponse
        # (this field is returned as-is by the documents API).
        error_type = type(exc).__name__
        await doc_coll.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "ingestion_status": "failed",
                    "error_message": f"Ingestion error ({error_type}). See server logs.",
                }
            },
        )
    finally:
        from app.core.memory import trim_memory

        await asyncio.to_thread(trim_memory)


async def index_parsed_chunks(
    doc_id_str: str,
    kb_id_str: str,
    chunks: list[dict[str, Any]] | None = None,
    strategy: ChunkingStrategy | None = None,
) -> None:
    """Run one ingestion job at a time per API process."""
    async with _get_ingestion_semaphore():
        await _index_parsed_chunks(
            doc_id_str=doc_id_str,
            kb_id_str=kb_id_str,
            chunks=chunks,
            strategy=strategy,
        )


def hashlib_qdrant_id(doc_id_str: str, chunk_index: int) -> str:
    """Generate a consistent UUID string for Qdrant from doc_id and chunk_index."""
    import hashlib
    import uuid

    unique_str = f"{doc_id_str}_{chunk_index}"
    hash_bytes = hashlib.sha256(unique_str.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))
