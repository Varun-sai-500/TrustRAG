"""
TRUSTRAG — Qdrant vector database client (Async).

Handles collection initialization and drops.
Separates knowledge bases into independent Qdrant collections.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.core.config import get_model_config, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)

API_ROOT = Path(__file__).resolve().parents[2]
_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return an async Qdrant client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        logger.info("Initializing async Qdrant client", url=settings.qdrant_url)
        try:
            # Support embedded local Qdrant directly via pip qdrant-client rust engine
            if (
                settings.qdrant_url in ("local", ":memory:")
                or settings.qdrant_url.startswith("./")
                or settings.qdrant_url.startswith("/")
                or not settings.qdrant_url.startswith("http")
            ):
                if settings.qdrant_url == "local":
                    storage_path = API_ROOT / "data" / "qdrant"
                elif settings.qdrant_url == ":memory:":
                    storage_path = ":memory:"
                else:
                    storage_path = Path(settings.qdrant_url)

                if isinstance(storage_path, Path):
                    storage_path.mkdir(parents=True, exist_ok=True)
                    _client = AsyncQdrantClient(path=str(storage_path))
                else:
                    _client = AsyncQdrantClient(location=str(storage_path))
            elif settings.qdrant_api_key:
                _client = AsyncQdrantClient(
                    url=settings.qdrant_url, api_key=settings.qdrant_api_key
                )
            else:
                _client = AsyncQdrantClient(url=settings.qdrant_url)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize async Qdrant client", detail=str(exc)
            ) from exc
    return _client


def get_collection_name(kb_id: str) -> str:
    """Consistently format collection name per KB ID."""
    return f"kb_{kb_id}"


async def _sparse_idf_enabled(client: AsyncQdrantClient, collection_name: str) -> bool | None:
    """Check whether the collection's sparse-text index uses Modifier.IDF.

    Returns None when the config cannot be read (fail-open: keep old behavior).
    """
    try:
        info = await client.get_collection(collection_name)
        params = getattr(getattr(info, "config", None), "params", None)
        sparse_vectors = getattr(params, "sparse_vectors", None) or {}
        sparse_params = sparse_vectors.get("sparse-text")
        modifier = getattr(sparse_params, "modifier", None)
        return modifier is not None and str(modifier).lower() == "idf"
    except Exception as exc:
        logger.warning(
            "Could not verify sparse index config; keeping existing collection",
            collection=collection_name,
            error=str(exc),
        )
        return None


async def init_kb_collection(kb_id: str) -> None:
    """
    Initialize a vector collection for the given knowledge base ID.

    Creates a collection with:
      - Dense vector parameters: Cosine distance, 384 dimensions (HuggingFace)
      - Sparse vector parameters: BM25-style TF client vectors + server-side
        IDF (Modifier.IDF). Qdrant derives IDF from collection statistics.

    Collections created before the IDF sparse config are deleted and recreated
    empty — their TF-only vectors are scoring-incompatible with IDF weighting.
    Operators must re-upload the KB's documents afterwards (re-index).
    """
    client = await get_qdrant_client()
    collection_name = get_collection_name(kb_id)
    cfg = get_model_config()

    try:
        # Check if already exists
        exists = await client.collection_exists(collection_name)
        if exists:
            idf_enabled = await _sparse_idf_enabled(client, collection_name)
            if idf_enabled is False:
                logger.warning(
                    "Recreating Qdrant collection with IDF sparse config; "
                    "re-upload this KB's documents to re-index",
                    collection=collection_name,
                )
                await client.delete_collection(collection_name)
            else:
                logger.debug("Qdrant collection already exists", collection=collection_name)
                return

        await _create_kb_collection(client, collection_name, cfg)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to initialize Qdrant collection '{collection_name}'", detail=str(exc)
        ) from exc


async def _create_kb_collection(client: AsyncQdrantClient, collection_name: str, cfg) -> None:  # type: ignore[no-untyped-def]
    """Create the KB collection with dense + IDF-weighted sparse vectors."""
    logger.info(
        "Creating Qdrant collection",
        collection=collection_name,
        dense_dim=cfg.embedding_dimensionality,
    )

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=cfg.embedding_dimensionality,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        # Sparse leg: client sends BM25 TF-saturated values (see
        # app/ingestion/sparse_vector.py); Qdrant multiplies query-time IDF
        # from collection statistics (Modifier.IDF).
        sparse_vectors_config={
            "sparse-text": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
                modifier=models.Modifier.IDF,
            )
        },
        # Keep document payloads on disk using memory-mapped pages
        on_disk_payload=True,
        # Ultra-low RAM: Quantize float32 vectors to INT8 with on-disk storage
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            )
        ),
    )
    logger.info(
        "Qdrant collection created with on_disk and INT8 quantization",
        collection=collection_name,
    )


async def delete_kb_collection(kb_id: str) -> None:
    """Drop the Qdrant collection associated with this KB."""
    client = await get_qdrant_client()
    collection_name = get_collection_name(kb_id)

    try:
        if await client.collection_exists(collection_name):
            logger.info("Dropping Qdrant collection", collection=collection_name)
            await client.delete_collection(collection_name)
            logger.info("Qdrant collection dropped", collection=collection_name)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to drop Qdrant collection '{collection_name}'", detail=str(exc)
        ) from exc


async def health_check() -> bool:
    """Verify connectivity to Qdrant cluster."""
    try:
        client = await get_qdrant_client()
        await client.get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed", error=str(exc))
        return False
