"""
Unit tests for the cross-encoder reranker (Phase 2).

The real CrossEncoder is NEVER loaded (no torch/model downloads): scoring is
driven by a FakeCrossEncoder through the get_reranker seam, and config by a
stub unless the test targets the real disabled default.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import _rerank_sync, rerank_candidate_chunks


def _stub_cfg(**overrides):
    base = {
        "reranker_enabled": True,
        "reranker_model": "fake-model",
        "reranker_top_k": 20,
        "fusion_top_k": 20,
        "max_context_chunks": 8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeCrossEncoder:
    """Deterministic stand-in: scores come from a text → score map."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.seen_pairs: list[list] = []

    def predict(self, pairs):
        self.seen_pairs.append(list(pairs))
        return [self._scores.get(doc, 0.0) for _, doc in pairs]


def _chunks(texts: list[str]) -> list[dict]:
    return [{"text": t, "dense_score": 0.5} for t in texts]


def _run(texts: list[str], scores: dict[str, float], **cfg_overrides):
    model = FakeCrossEncoder(scores)
    with (
        patch.object(reranker_module, "get_model_config", return_value=_stub_cfg(**cfg_overrides)),
        patch.object(reranker_module, "get_reranker", return_value=model),
    ):
        out = _rerank_sync("probe query", _chunks(texts))
    return out, model


def test_rerank_orders_by_cross_encoder_score():
    out, _ = _run(["a", "b", "c"], {"a": 0.1, "b": 0.9, "c": 0.5})
    assert [c["text"] for c in out] == ["b", "c", "a"]
    assert [c["rerank_score"] for c in out] == [0.9, 0.5, 0.1]


def test_rerank_adaptive_top4_on_confident_head():
    texts = [f"doc-{i}" for i in range(8)]
    scores = dict.fromkeys(texts, 0.1) | {"doc-3": 0.95}
    out, _ = _run(texts, scores)
    assert len(out) == 4
    assert out[0]["text"] == "doc-3"


def test_rerank_keeps_full_context_when_head_uncertain():
    texts = [f"doc-{i}" for i in range(8)]
    scores = {t: 0.4 + i * 0.01 for i, t in enumerate(texts)}  # top ≈ 0.47 < 0.80
    out, _ = _run(texts, scores)
    assert len(out) == 8
    assert out[0]["text"] == "doc-7"


def test_rerank_bounds_scoring_depth():
    texts = [f"doc-{i}" for i in range(25)]
    scores = {t: i / 25 for i, t in enumerate(texts)}
    out, model = _run(texts, scores)
    assert len(model.seen_pairs) == 1
    assert len(model.seen_pairs[0]) == 20  # reranker.top_k depth cap
    assert out[0]["text"] == "doc-19"  # best of the scored head
    assert len(out) == 8  # output still capped by max_context_chunks


def test_rerank_does_not_mutate_caller_order():
    chunks = _chunks(["a", "b", "c"])
    before = [c["text"] for c in chunks]
    model = FakeCrossEncoder({"a": 0.1, "b": 0.9, "c": 0.5})
    with (
        patch.object(reranker_module, "get_model_config", return_value=_stub_cfg()),
        patch.object(reranker_module, "get_reranker", return_value=model),
    ):
        _rerank_sync("probe query", chunks)
    assert [c["text"] for c in chunks] == before


def test_rerank_disabled_returns_rrf_slice_without_model():
    exploding = MagicMock(side_effect=AssertionError("model must not load when disabled"))
    with patch.object(reranker_module, "get_reranker", exploding):
        # Real config: reranker.enabled is false → model seam never touched.
        out = _rerank_sync("probe query", _chunks([f"doc-{i}" for i in range(10)]))
    exploding.assert_not_called()
    assert len(out) == 8


def test_rerank_none_model_falls_back_to_rrf_order():
    with (
        patch.object(reranker_module, "get_model_config", return_value=_stub_cfg()),
        patch.object(reranker_module, "get_reranker", return_value=None),
    ):
        out = _rerank_sync("probe query", _chunks([f"doc-{i}" for i in range(10)]))
    assert [c["text"] for c in out] == [f"doc-{i}" for i in range(8)]
    assert all("rerank_score" not in c for c in out)


def test_rerank_predict_failure_falls_back_to_rrf_order():
    model = MagicMock()
    model.predict.side_effect = Exception("CUDA OOM")
    with (
        patch.object(reranker_module, "get_model_config", return_value=_stub_cfg()),
        patch.object(reranker_module, "get_reranker", return_value=model),
    ):
        out = _rerank_sync("probe query", _chunks([f"doc-{i}" for i in range(10)]))
    assert [c["text"] for c in out] == [f"doc-{i}" for i in range(8)]


def test_rerank_empty_input():
    assert _rerank_sync("probe query", []) == []


@pytest.mark.asyncio
async def test_rerank_async_wrapper_respects_disabled_default():
    out = await rerank_candidate_chunks("probe query", _chunks([f"doc-{i}" for i in range(10)]))
    assert len(out) == 8
