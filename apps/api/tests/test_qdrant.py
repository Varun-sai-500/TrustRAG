"""
Unit tests for Qdrant collection init + IDF sparse migration (Phase 1).

All Qdrant I/O is mocked — no live services.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models

from app.db import qdrant as qdrant_module
from app.db.qdrant import init_kb_collection


def _mock_client(
    *,
    exists: bool,
    modifier,
    get_collection_raises: bool = False,
) -> MagicMock:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=exists)
    if get_collection_raises:
        client.get_collection = AsyncMock(side_effect=Exception("unreadable"))
    else:
        sparse_vectors = {"sparse-text": SimpleNamespace(modifier=modifier)} if exists else {}
        info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(sparse_vectors=sparse_vectors))
        )
        client.get_collection = AsyncMock(return_value=info)
    client.delete_collection = AsyncMock()
    client.create_collection = AsyncMock()
    return client


async def _run_init(client: MagicMock):
    # NOTE: the patch must stay active while the coroutine is AWAITED —
    # returning the un-awaited coroutine out of the `with` block would
    # run it after the patch is undone (→ real network calls).
    with patch.object(qdrant_module, "get_qdrant_client", AsyncMock(return_value=client)):
        return await init_kb_collection("kb_phase1_probe")


@pytest.mark.asyncio
async def test_create_new_collection_uses_idf_modifier():
    client = _mock_client(exists=False, modifier=None)
    await _run_init(client)
    client.delete_collection.assert_not_awaited()
    client.create_collection.assert_awaited_once()
    sparse_config = client.create_collection.call_args.kwargs["sparse_vectors_config"]
    assert sparse_config["sparse-text"].modifier == models.Modifier.IDF


@pytest.mark.asyncio
async def test_existing_idf_collection_is_kept():
    client = _mock_client(exists=True, modifier=models.Modifier.IDF)
    await _run_init(client)
    client.delete_collection.assert_not_awaited()
    client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_collection_is_recreated_with_idf():
    client = _mock_client(exists=True, modifier=None)
    await _run_init(client)
    client.delete_collection.assert_awaited_once()
    client.create_collection.assert_awaited_once()
    sparse_config = client.create_collection.call_args.kwargs["sparse_vectors_config"]
    assert sparse_config["sparse-text"].modifier == models.Modifier.IDF


@pytest.mark.asyncio
async def test_unreadable_config_fails_open():
    """Unverifiable sparse config keeps the collection (never delete blind)."""
    client = _mock_client(exists=True, modifier=None, get_collection_raises=True)
    await _run_init(client)
    client.delete_collection.assert_not_awaited()
    client.create_collection.assert_not_awaited()
