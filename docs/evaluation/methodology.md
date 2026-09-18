# TRUSTRAG — Evaluation Methodology

## Principles

1. No fabricated results. All metrics must come from actual system runs.
2. Measure improvement — not assumed improvement.
3. Ablations must isolate single variables.
4. Baselines must be fair comparisons (same data, same query set).

---

## Experiment Configurations (Phase 11)

| Configuration | Description |
|--------------|-------------|
| `baseline_rag` | Dense retrieval only + selected-LLM generation. No verification, no recovery. |
| `hybrid_rag` | Dense + BM25 hybrid/RRF + selected-LLM generation. No verification. |
| `hybrid_rerank` | Hybrid + cross-encoder reranking + selected-LLM generation. No verification. Reranker stays off by default (Docker lacks torch — enable only with the `local-models` extra); thresholds uncalibrated pending this ablation. |
| `verified_rag` | Hybrid + reranking + claim verification. No adaptive recovery. |
| `trustrag_full` | Full TRUSTRAG: hybrid + reranking + verification + diagnosis + recovery + abstention. |

---

## Metrics

| Metric | Definition | Source |
|--------|-----------|--------|
| `retrieval_hit_rate` | Fraction of queries where relevant evidence was retrieved in top-k | Human / automated labels |
| `evidence_coverage` | Fraction of claims with at least one supporting evidence chunk | TRUSTRAG claim verification |
| `claim_support_rate` | Fraction of claims verified as SUPPORTED | TRUSTRAG claim verification |
| `contradiction_rate` | Fraction of claims verified as CONTRADICTED | TRUSTRAG claim verification |
| `citation_correctness` | Fraction of citations that actually support the cited claim | TRUSTRAG citation check |
| `recovery_success_rate` | Fraction of initially-failing analyses that recovered successfully | Recovery logs |
| `abstention_rate` | Fraction of queries where the system abstained | Analysis status |
| `latency_p50` | Median end-to-end analysis latency (ms) | Trace events |
| `latency_p95` | 95th percentile analysis latency (ms) | Trace events |

---

## Ablation Plan

| Ablation | Variable Isolated |
|----------|-----------------|
| Dense vs Hybrid | Retrieval method (sparse BM25 off/on) |
| Hybrid vs Hybrid+Reranking | Reranking contribution |
| Verification off vs on | Claim verification value |
| Recovery off vs on | Adaptive recovery value |
| Integrity logic off vs on | Evidence integrity analysis value |

---

## Query Dataset

Minimum viable evaluation set (frozen as `baseline_v1`: 12 factual + 3 temporal +
3 conflicting + 2 missing-evidence + 5 adversarial = 25 queries over the 6-file
fixture corpus):
- Queries spanning: factual, temporal (outdated evidence), conflicting sources, missing evidence
- At least 5 adversarial: queries designed to trigger failures
- Fixtures predate the IDF + newline changes: re-index after any chunking/normalization
  change; gold snippets are verbatim fixture text (the stable key across re-indexes)

---

## Reporting

All experiment results are stored in the `experiments` MongoDB collection.
Results are displayed in the Experiments page of the UI.
Results must include: configuration, query, metrics, timestamps, config_version.

---

## Baseline dataset v1 (Phase 0 — frozen)

Do not edit `apps/api/tests/eval/datasets/baseline_v1.jsonl` in place.
To change the set, add `baseline_v2.jsonl` and keep v1 for comparability.

- Corpus fixtures: `apps/api/tests/eval/fixtures/corpus/*.txt` (6 docs: refund,
  shipping, pricing-2025 [stale], pricing-2026 [current], support, notice-board
  with embedded prompt-injection graffiti).
- 25 queries: 12 factual + 3 temporal + 3 conflicting + 2 missing-evidence
  (= 20 standard) + 5 adversarial (injection, stale-bait, false-premise,
  garbled, injection-suffix).
- Gold evidence = verbatim snippets from the fixtures (chunk ids change across
  re-indexes, snippet text is the stable key). `test_baseline_dataset.py` fails
  CI if any snippet stops occurring in its fixture.
- Metrics code: `apps/api/tests/eval/metrics.py` (pure, deterministic;
  unit-tested in `test_eval_metrics.py` with hand-computed values).

### Run procedure (live numbers)

1. Ingest the 6 fixture files as documents into a fresh knowledge base.
2. Run: `python scripts/run_baseline_eval.py --email ... --password ... \
   --kb-id <KB_ID> --post-experiment`
   (7s gap between queries respects the 10-analyses/min rate limit;
   ~25 queries take ~5–10 min plus LLM time.)
3. Results JSON lands in `docs/evaluation/results/`; aggregate is also POSTed
   to `/api/v1/experiments` with the server `config_version`.
4. Copy the aggregate row into the snapshot table below. Never hand-edit
   a results JSON — re-run instead.

### Baseline snapshot (measured runs only — no fabricated rows)

| Date | Config | Dataset | recall@k | hit_rate | MRR | nDCG | coverage | contra | citation | abstain | p50 | p95 |
|------|--------|---------|----------|----------|-----|------|----------|--------|----------|---------|-----|-----|
| _pending live run_ | trustrag_baseline | baseline_v1 | — | — | — | — | — | — | n/a¹ | — | — | — |

¹ Citation correctness is an existence check until Phase 5 (every `[Segment N]` names a
served segment — provenance honesty, not entailment); the runner records `claims_with_evidence_rate` alongside.
