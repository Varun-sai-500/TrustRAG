#!/usr/bin/env python3
"""TRUSTRAG Phase-0 baseline runner — live evaluation against a running API.

Reads the frozen dataset (apps/api/tests/eval/datasets/baseline_v1.jsonl),
runs one analysis per query, scores each with tests/eval/metrics.py, and
writes a results JSON (docs/evaluation/results/). Optionally records the
aggregate as an experiment via POST /api/v1/experiments.

Prerequisites (operator):
  1. API + MongoDB + Qdrant + an LLM provider running (see docker-compose.yml).
  2. A knowledge base containing the fixture corpus:
       apps/api/tests/eval/fixtures/corpus/*.txt  (upload each file as a document)
  3. A registered user (email + password).

Example:
  python scripts/run_baseline_eval.py \\
      --email ops@example.com --password '...' --kb-id <KB_ID> --post-experiment

Stdlib only. Respects the 10-analyses/minute rate limit via --delay-seconds.
Exit codes: 0 = results written, 1 = usage/service error, 2 = run incomplete
(some queries timed out; partial results still written).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))

from eval import metrics  # noqa: E402  (sys.path wired above)

DATASET_DEFAULT = REPO_ROOT / "apps" / "api" / "tests" / "eval" / "datasets" / "baseline_v1.jsonl"
RESULTS_DIR_DEFAULT = REPO_ROOT / "docs" / "evaluation" / "results"
TERMINAL_STATUSES = {"completed", "failed", "abstained"}


class ApiError(RuntimeError):
    pass


def api_request(method: str, base: str, path: str, token: str | None, payload: dict | None) -> dict:
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise ApiError(f"{method} {path} → HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"{method} {path} → connection failed: {exc}") from exc


def parse_trace_span_ms(trace: list[dict]) -> float | None:
    """End-to-end ms from first to last trace-event timestamp. None if unusable."""
    stamps: list[datetime] = []
    for event in trace:
        raw = event.get("timestamp")
        if not raw:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(stamps) < 2:
        return None
    return max(0.0, (max(stamps) - min(stamps)).total_seconds() * 1000.0)


def repo_config_version() -> str:
    """config_version from the repo's models.yaml (server snapshot preferred at runtime)."""
    text = (REPO_ROOT / "apps" / "api" / "config" / "models.yaml").read_text(encoding="utf-8")
    match = re.search(r'config_version:\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def run_query(args: argparse.Namespace, token: str, row: dict) -> dict:
    started = time.monotonic()
    created = api_request(
        "POST",
        args.api_url,
        "/api/v1/analyses",
        token,
        {"knowledge_base_id": args.kb_id, "query": row["query"]},
    )
    analysis_id = created["id"]
    deadline = time.monotonic() + args.poll_timeout
    analysis: dict = created
    while time.monotonic() < deadline:
        time.sleep(args.poll_interval)
        analysis = api_request("GET", args.api_url, f"/api/v1/analyses/{analysis_id}", token, None)
        if analysis.get("status") in TERMINAL_STATUSES:
            break
    wall_ms = (time.monotonic() - started) * 1000.0
    status = analysis.get("status", "unknown")
    timed_out = status not in TERMINAL_STATUSES

    claims: list[dict] = []
    evidence: list[dict] = []
    trace: list[dict] = []
    if not timed_out:
        detail = api_request(
            "GET", args.api_url, f"/api/v1/analyses/{analysis_id}/detail", token, None
        )
        claims = detail.get("claims", [])
        evidence = detail.get("evidence", [])
        trace = detail.get("trace", [])
        server_snapshot = (detail.get("analysis") or {}).get("config_snapshot") or {}
    else:
        server_snapshot = {}

    evidence_texts = [e.get("text", "") for e in evidence]
    retrieved_ids = [str(e.get("id", "")) for e in evidence]
    snippets = [ev["snippet"] for ev in row.get("gold_evidence", [])]
    resolved = metrics.resolve_snippets(evidence_texts, snippets)
    # Honest ID-space mapping: a gold id is "retrieved" iff some evidence chunk
    # containing that gold snippet was retrieved. Snippet text is the stable key
    # (chunk ids change across re-indexes; see methodology §Baseline dataset).
    gold_ids = {f"gold:{i}" for i, r in enumerate(resolved) if r["found"]}
    ranked_gold_hits = [
        f"gold:{i}"
        for rank, text in enumerate(evidence_texts)
        for i, snip in enumerate(snippets)
        if snip.lower() in text.lower()
    ]
    # De-duplicate preserving rank order for rank metrics.
    ranked_gold_hits = list(dict.fromkeys(ranked_gold_hits))
    _ = retrieved_ids  # raw chunk ids kept in the per-query record below, not scored

    claim_states = [c.get("state", "NEUTRAL") for c in claims]
    # Phase-4 existence check: every [Segment N] in the answer must name a served
    # segment (1..len(evidence) approximates the served range). This measures
    # provenance honesty, NOT entailment — the entailment check lands in Phase 5.
    # No refs (e.g. abstentions) → [] → citation None, never a penalty.
    answer_text = ((detail.get("analysis") or {}).get("answer")) or ""
    cited = [int(n) for n in re.findall(r"\[Segment\s+(\d+)\]", answer_text)]
    citation_supporting = [1 <= n <= len(evidence) for n in cited]
    linked = sum(1 for c in claims if c.get("evidence_ids"))
    trace_span_ms = parse_trace_span_ms(trace)

    scored = metrics.score_query(
        query_id=row["id"],
        query_class=row["query_class"],
        retrieved_ids=ranked_gold_hits,
        gold_ids=gold_ids,
        evidence_texts=evidence_texts,
        gold_snippets=snippets,
        claim_states=claim_states,
        citation_supporting=citation_supporting,
        status=status,
        latency_ms=trace_span_ms if trace_span_ms is not None else wall_ms,
        k=args.top_k,
    )
    scored["expected_outcome"] = row.get("expected_outcome")
    scored["cited_segments"] = cited
    scored["outcome_match"] = (status == "abstained") == (row.get("expected_outcome") == "abstained")
    scored["claims_with_evidence_rate"] = (linked / len(claims)) if claims else None
    scored["evidence_count"] = len(evidence)
    scored["wall_ms"] = wall_ms
    scored["trace_span_ms"] = trace_span_ms
    scored["timed_out"] = timed_out
    scored["analysis_id"] = analysis_id
    return scored, server_snapshot


def print_table(agg: dict) -> None:
    rows = [
        ("queries", agg["n_queries"]),
        ("recall@k", round(agg["recall_at_k"], 4)),
        ("hit_rate@k", round(agg["hit_rate_at_k"], 4)),
        ("mrr", round(agg["mrr"], 4)),
        ("nDCG@k", round(agg["ndcg_at_k"], 4)),
        ("snippet_recall", round(agg["snippet_recall"], 4)),
        ("evidence_coverage", round(agg["evidence_coverage"], 4)),
        ("claim_support_rate", round(agg["claim_support_rate"], 4)),
        ("contradiction_rate", round(agg["contradiction_rate"], 4)),
        ("citation_correctness", agg["citation_correctness"]),
        ("abstention_rate", round(agg["abstention_rate"], 4)),
        ("p50_ms", round(agg["latency"]["p50_ms"], 1)),
        ("p95_ms", round(agg["latency"]["p95_ms"], 1)),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"  {key:<{width}}  {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Phase-0 baseline evaluation.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--kb-id", required=True, help="KB containing the fixture corpus")
    parser.add_argument("--dataset", default=str(DATASET_DEFAULT))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR_DEFAULT))
    parser.add_argument("--config-name", default="trustrag_baseline")
    parser.add_argument("--description", default="Phase-0 frozen baseline (baseline_v1).")
    parser.add_argument("--top-k", type=int, default=metrics.DEFAULT_TOP_K)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--poll-timeout", type=float, default=600.0)
    parser.add_argument("--delay-seconds", type=float, default=7.0, help="gap between queries")
    parser.add_argument("--post-experiment", action="store_true")
    args = parser.parse_args()

    dataset_rows = [
        json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
    ]
    print(f"Dataset: {args.dataset} ({len(dataset_rows)} queries)")

    try:
        login = api_request(
            "POST",
            args.api_url,
            "/api/v1/auth/login",
            None,
            {"email": args.email, "password": args.password},
        )
    except ApiError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    token = login["access_token"]

    query_scores: list[dict] = []
    server_versions: set[str] = set()
    incomplete = 0
    for i, row in enumerate(dataset_rows):
        print(f"[{i + 1}/{len(dataset_rows)}] {row['id']} ({row['query_class']}) …", flush=True)
        try:
            scored, snapshot = run_query(args, token, row)
        except ApiError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            incomplete += 1
            continue
        if snapshot.get("config_version"):
            server_versions.add(str(snapshot["config_version"]))
        query_scores.append(scored)
        print(f"  status={scored['status']} recall={scored['recall_at_k']:.3f} "
              f"lat={scored['latency_ms']:.0f}ms")
        if i < len(dataset_rows) - 1:
            time.sleep(args.delay_seconds)

    agg = metrics.aggregate_scores(query_scores)
    agg["outcome_match_rate"] = (
        sum(1 for s in query_scores if s["outcome_match"]) / len(query_scores)
        if query_scores
        else 0.0
    )
    config_version = next(iter(server_versions), None) or repo_config_version()
    if len(server_versions) > 1:
        print(f"WARNING: server config_version changed mid-run: {sorted(server_versions)}",
              file=sys.stderr)

    results = {
        "config_name": args.config_name,
        "description": args.description,
        "dataset": Path(args.dataset).name,
        "config_version": config_version,
        "server_config_versions_seen": sorted(server_versions),
        "created_at": datetime.now(UTC).isoformat(),
        "kb_id": args.kb_id,
        "top_k": args.top_k,
        "aggregate": agg,
        "queries": query_scores,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{args.config_name}_{config_version}_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nAggregate:")
    print_table(agg)
    print(f"outcome_match_rate  {agg['outcome_match_rate']:.4f}")
    print(f"\nWrote {out_path}")

    if args.post_experiment:
        recorded = api_request(
            "POST",
            args.api_url,
            "/api/v1/experiments",
            token,
            {
                "config_name": args.config_name,
                "description": f"{args.description} (config_version={config_version})",
                "metrics": agg,
            },
        )
        print(f"Recorded experiment id={recorded.get('id')}")

    if incomplete or any(s["timed_out"] for s in query_scores):
        print(f"INCOMPLETE: {incomplete} errors, "
              f"{sum(1 for s in query_scores if s['timed_out'])} timeouts", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
