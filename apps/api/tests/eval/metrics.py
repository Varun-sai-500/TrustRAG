"""
TRUSTRAG Phase-0 eval harness — pure metric functions.

Dependency-free (stdlib only) so the same module is usable from pytest and
from the live runner script. All functions are deterministic: same input →
same output, no randomness, no I/O. Outputs are JSON-serializable.

Metric definitions follow docs/evaluation/methodology.md:
  recall@k / hit-rate / MRR / nDCG  → retrieval quality
  coverage / support / contradiction → claim verification (mirrors the
      reliability formula in app/verification/verdict.py: coverage * (1 - contra))
  citation correctness               → fraction of citations whose cited chunk
      actually contains supporting text
  abstention rate                    → fraction of ABSTAIN outcomes
  p50 / p95                          → latency distribution
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_TOP_K = 8  # matches retrieval.max_context_chunks in models.yaml


# ── Retrieval metrics (ranked id lists vs gold id set) ───────────────────────


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = DEFAULT_TOP_K) -> float:
    """Fraction of gold ids present in the top-k retrieved ids."""
    if not gold_ids or k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = DEFAULT_TOP_K) -> float:
    """1.0 if any gold id is in the top-k retrieved ids, else 0.0."""
    if not gold_ids or k <= 0:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """1/rank of the first retrieved id that is gold (1-indexed rank)."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = DEFAULT_TOP_K) -> float:
    """Binary-relevance nDCG@k over the ranked retrieved ids."""
    if not gold_ids or k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rid in enumerate(top, start=1) if rid in gold_ids)
    ideal_hits = min(len(gold_ids), len(top))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Snippet resolution (gold text → retrieved evidence texts) ────────────────


def resolve_snippets(evidence_texts: list[str], snippets: list[str]) -> list[dict[str, Any]]:
    """Map each gold snippet to the first evidence text containing it.

    Case-insensitive substring match. Returns one dict per snippet:
    {"snippet": str, "found": bool, "first_rank": int | None} with 0-indexed rank.
    An empty snippet list returns [] (caller decides what that means —
    e.g. missing_evidence queries expect abstention, not retrieval).
    """
    lowered = [t.lower() for t in evidence_texts]
    resolved: list[dict[str, Any]] = []
    for snippet in snippets:
        needle = snippet.lower()
        first_rank = next((i for i, text in enumerate(lowered) if needle in text), None)
        resolved.append(
            {"snippet": snippet, "found": first_rank is not None, "first_rank": first_rank}
        )
    return resolved


def snippet_recall(resolved: list[dict[str, Any]]) -> float:
    """Fraction of gold snippets found in the retrieved evidence texts."""
    if not resolved:
        return 0.0
    return sum(1 for r in resolved if r["found"]) / len(resolved)


def snippet_mrr(resolved: list[dict[str, Any]]) -> float:
    """MRR over snippet first-ranks (0-indexed rank → 1/(rank+1))."""
    if not resolved:
        return 0.0
    return sum(1.0 / (r["first_rank"] + 1) for r in resolved if r["found"]) / len(resolved)


# ── Claim verification summary ───────────────────────────────────────────────


def claim_summary(states: list[str]) -> dict[str, Any]:
    """Aggregate SUPPORTED/CONTRADICTED/NEUTRAL claim states.

    Unknown states are counted as NEUTRAL (fail-safe: unverified claims must
    not inflate coverage). Reliability mirrors verdict.py:
    score = coverage * (1 - contradiction_rate).
    """
    supported = contradicted = neutral = 0
    for raw in states:
        state = (raw or "").upper()
        if state == "SUPPORTED":
            supported += 1
        elif state == "CONTRADICTED":
            contradicted += 1
        else:
            neutral += 1
    total = supported + contradicted + neutral
    coverage = supported / total if total else 0.0
    contradiction_rate = contradicted / total if total else 0.0
    return {
        "supported": supported,
        "contradicted": contradicted,
        "neutral": neutral,
        "total": total,
        "coverage": coverage,
        "contradiction_rate": contradiction_rate,
        "reliability_score": max(0.0, min(1.0, coverage * (1.0 - contradiction_rate))),
    }


# ── Citations / abstention / latency ─────────────────────────────────────────


def citation_correctness(supporting_flags: list[bool]) -> float | None:
    """Fraction of citations whose cited chunk supports the cited claim.

    Returns None when there are no citations (e.g. ABSTAIN answers) so that
    aggregation skips them instead of penalizing abstention.
    """
    if not supporting_flags:
        return None
    return sum(1 for f in supporting_flags if f) / len(supporting_flags)


def is_abstained(status: str | None) -> bool:
    """True for ABSTAIN / ABSTAINED analysis outcomes (case-insensitive)."""
    return (status or "").upper() in {"ABSTAIN", "ABSTAINED"}


def abstention_rate(statuses: list[str]) -> float:
    """Fraction of analysis outcomes that abstained."""
    if not statuses:
        return 0.0
    return sum(1 for s in statuses if is_abstained(s)) / len(statuses)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in 0..100). Empty → 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    """Count/min/p50/p95/max over latency samples in milliseconds."""
    if not samples_ms:
        return {"count": 0, "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(samples_ms),
        "min_ms": float(min(samples_ms)),
        "p50_ms": percentile(samples_ms, 50),
        "p95_ms": percentile(samples_ms, 95),
        "max_ms": float(max(samples_ms)),
    }


# ── Per-query scoring + run aggregation ──────────────────────────────────────


def score_query(
    query_id: str,
    query_class: str,
    retrieved_ids: list[str],
    gold_ids: set[str],
    evidence_texts: list[str],
    gold_snippets: list[str],
    claim_states: list[str],
    citation_supporting: list[bool],
    status: str,
    latency_ms: float,
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Score one evaluated query. All inputs are plain data; no I/O."""
    resolved = resolve_snippets(evidence_texts, gold_snippets)
    claims = claim_summary(claim_states)
    return {
        "query_id": query_id,
        "query_class": query_class,
        "recall_at_k": recall_at_k(retrieved_ids, gold_ids, k),
        "hit_rate_at_k": hit_rate_at_k(retrieved_ids, gold_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, gold_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_ids, k),
        "snippet_recall": snippet_recall(resolved),
        "snippet_mrr": snippet_mrr(resolved),
        "claim_supported": claims["supported"],
        "claim_contradicted": claims["contradicted"],
        "claim_neutral": claims["neutral"],
        "claim_total": claims["total"],
        "evidence_coverage": claims["coverage"],
        "contradiction_rate": claims["contradiction_rate"],
        "reliability_score": claims["reliability_score"],
        "citation_correctness": citation_correctness(citation_supporting),
        "abstained": is_abstained(status),
        "status": status,
        "latency_ms": float(latency_ms),
        "k": k,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_scores(query_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query score dicts into run-level metrics."""
    n = len(query_scores)
    cites = [
        s["citation_correctness"] for s in query_scores if s["citation_correctness"] is not None
    ]
    latencies = [s["latency_ms"] for s in query_scores]
    by_class: dict[str, int] = {}
    for s in query_scores:
        by_class[s["query_class"]] = by_class.get(s["query_class"], 0) + 1
    total_claims = sum(s["claim_total"] for s in query_scores)
    return {
        "n_queries": n,
        "by_class": by_class,
        "recall_at_k": _mean([s["recall_at_k"] for s in query_scores]),
        "hit_rate_at_k": _mean([s["hit_rate_at_k"] for s in query_scores]),
        "mrr": _mean([s["mrr"] for s in query_scores]),
        "ndcg_at_k": _mean([s["ndcg_at_k"] for s in query_scores]),
        "snippet_recall": _mean([s["snippet_recall"] for s in query_scores]),
        "evidence_coverage": _mean([s["evidence_coverage"] for s in query_scores]),
        "claim_support_rate": (
            sum(s["claim_supported"] for s in query_scores) / total_claims if total_claims else 0.0
        ),
        "contradiction_rate": (
            sum(s["claim_contradicted"] for s in query_scores) / total_claims
            if total_claims
            else 0.0
        ),
        "citation_correctness": _mean(cites) if cites else None,
        "citations_evaluated": len(cites),
        "abstention_rate": _mean([1.0 if s["abstained"] else 0.0 for s in query_scores]),
        "latency": latency_summary(latencies),
    }
