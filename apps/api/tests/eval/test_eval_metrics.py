"""
Unit tests for the Phase-0 eval metric functions.

All expectations are hand-computed. These tests run in CI with no live
services (pure functions, no I/O, no randomness).
"""

from __future__ import annotations

from tests.eval import metrics


def test_recall_at_k_hand_computed():
    retrieved = ["a", "b", "c"]
    gold = {"a", "c", "z"}
    assert metrics.recall_at_k(retrieved, gold, k=2) == 1 / 3  # only "a" in top-2
    assert metrics.recall_at_k(retrieved, gold, k=3) == 2 / 3  # "a" + "c"
    assert metrics.recall_at_k(retrieved, set(), k=3) == 0.0
    assert metrics.recall_at_k([], gold, k=3) == 0.0
    assert metrics.recall_at_k(retrieved, gold, k=0) == 0.0


def test_hit_rate_at_k():
    assert metrics.hit_rate_at_k(["x", "b"], {"b"}, k=2) == 1.0
    assert metrics.hit_rate_at_k(["x", "y"], {"b"}, k=2) == 0.0
    assert metrics.hit_rate_at_k(["b"], set(), k=1) == 0.0


def test_reciprocal_rank_hand_computed():
    assert metrics.reciprocal_rank(["b"], {"b"}) == 1.0
    assert metrics.reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["x", "y"], {"b"}) == 0.0
    assert metrics.reciprocal_rank([], {"b"}) == 0.0


def test_ndcg_at_k_hand_computed():
    # retrieved ["a","x"], gold {"a","b"}: dcg=1/log2(2)=1.0,
    # idcg=1+1/log2(3)≈1.6309 → ndcg≈0.6131
    ndcg = metrics.ndcg_at_k(["a", "x"], {"a", "b"}, k=2)
    assert abs(ndcg - 0.613147) < 1e-5
    assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0  # perfect ranking
    assert metrics.ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), k=2) == 0.0


def test_resolve_snippets_case_insensitive():
    texts = ["FULL REFUND within 30 days of delivery", "Shipping is free"]
    resolved = metrics.resolve_snippets(texts, ["30 Days", "missing phrase"])
    assert resolved[0] == {"snippet": "30 Days", "found": True, "first_rank": 0}
    assert resolved[1] == {"snippet": "missing phrase", "found": False, "first_rank": None}
    assert metrics.resolve_snippets(texts, []) == []


def test_snippet_recall_and_mrr():
    resolved = [
        {"snippet": "a", "found": True, "first_rank": 0},  # 1/1
        {"snippet": "b", "found": True, "first_rank": 2},  # 1/3
        {"snippet": "c", "found": False, "first_rank": None},  # 0
    ]
    assert metrics.snippet_recall(resolved) == 2 / 3
    assert abs(metrics.snippet_mrr(resolved) - (1.0 + 1 / 3) / 3) < 1e-9
    assert metrics.snippet_recall([]) == 0.0
    assert metrics.snippet_mrr([]) == 0.0


def test_claim_summary_mirrors_verdict_formula():
    summary = metrics.claim_summary(["SUPPORTED", "supported", "NEUTRAL", "CONTRADICTED", "bogus"])
    assert summary["supported"] == 2
    assert summary["contradicted"] == 1
    assert summary["neutral"] == 2  # NEUTRAL + unknown "bogus" (fail-safe)
    assert summary["total"] == 5
    assert summary["coverage"] == 0.4
    assert summary["contradiction_rate"] == 0.2
    assert abs(summary["reliability_score"] - 0.4 * 0.8) < 1e-9  # coverage*(1-contra)
    empty = metrics.claim_summary([])
    assert empty["total"] == 0 and empty["reliability_score"] == 0.0


def test_citation_correctness_none_when_no_citations():
    assert metrics.citation_correctness([True, True, False]) == 2 / 3
    assert metrics.citation_correctness([]) is None  # abstentions must not score 0


def test_abstention_detection():
    assert metrics.is_abstained("abstained") is True
    assert metrics.is_abstained("ABSTAIN") is True
    assert metrics.is_abstained("completed") is False
    assert metrics.is_abstained(None) is False
    assert metrics.abstention_rate(["completed", "ABSTAINED"]) == 0.5
    assert metrics.abstention_rate([]) == 0.0


def test_percentile_hand_computed():
    assert metrics.percentile([1, 2, 3, 4], 50) == 2.5
    assert abs(metrics.percentile([1, 2, 3, 4], 95) - 3.85) < 1e-9
    assert metrics.percentile([5.0], 95) == 5.0
    assert metrics.percentile([], 95) == 0.0


def test_latency_summary_hand_computed():
    summary = metrics.latency_summary([100.0, 200.0, 300.0])
    assert summary == {
        "count": 3,
        "min_ms": 100.0,
        "p50_ms": 200.0,
        "p95_ms": 290.0,
        "max_ms": 300.0,
    }
    assert metrics.latency_summary([])["count"] == 0


def test_score_query_keys_and_values():
    first = metrics.score_query(
        query_id="q001",
        query_class="factual",
        retrieved_ids=["e1", "e2"],
        gold_ids={"e1", "e9"},
        evidence_texts=["full refund within 30 days"],
        gold_snippets=["30 days"],
        claim_states=["SUPPORTED", "NEUTRAL"],
        citation_supporting=[True],
        status="completed",
        latency_ms=123.0,
    )
    assert first["recall_at_k"] == 0.5
    assert first["hit_rate_at_k"] == 1.0
    assert first["mrr"] == 1.0
    assert first["snippet_recall"] == 1.0
    assert first["evidence_coverage"] == 0.5
    assert first["citation_correctness"] == 1.0
    assert first["abstained"] is False
    # Deterministic: same input → identical output
    assert (
        metrics.score_query(
            query_id="q001",
            query_class="factual",
            retrieved_ids=["e1", "e2"],
            gold_ids={"e1", "e9"},
            evidence_texts=["full refund within 30 days"],
            gold_snippets=["30 days"],
            claim_states=["SUPPORTED", "NEUTRAL"],
            citation_supporting=[True],
            status="completed",
            latency_ms=123.0,
        )
        == first
    )


def test_aggregate_scores_skips_missing_citations():
    with_cite = metrics.score_query(
        query_id="a",
        query_class="factual",
        retrieved_ids=["e1"],
        gold_ids={"e1"},
        evidence_texts=["t"],
        gold_snippets=["t"],
        claim_states=["SUPPORTED"],
        citation_supporting=[True, False],
        status="completed",
        latency_ms=100.0,
    )
    abstained = metrics.score_query(
        query_id="b",
        query_class="missing_evidence",
        retrieved_ids=[],
        gold_ids=set(),
        evidence_texts=[],
        gold_snippets=[],
        claim_states=[],
        citation_supporting=[],
        status="abstained",
        latency_ms=50.0,
    )
    agg = metrics.aggregate_scores([with_cite, abstained])
    assert agg["n_queries"] == 2
    assert agg["by_class"] == {"factual": 1, "missing_evidence": 1}
    assert agg["recall_at_k"] == 0.5
    assert agg["citation_correctness"] == 0.5  # only the cited query counts
    assert agg["citations_evaluated"] == 1
    assert agg["abstention_rate"] == 0.5
    assert agg["claim_support_rate"] == 1.0  # 1 supported of 1 total claim
    assert agg["latency"]["count"] == 2
    assert metrics.aggregate_scores([])["n_queries"] == 0
