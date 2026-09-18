# TRUSTRAG — Implementation Plan (verified against code)

> **Status 2026-09-16:** Phase 0 ✅ · Phase 1 ✅ · Phase 2 ✅ · OCR fallback ✅ · Phase 3 ✅ ·
> Phase 4 ✅ · Phase 5 ✅ · Phase 6 ✅ · Phase 8 ✅ (lifecycle: snapshot/rollback routes,
> empty-snapshot guard, OCR-preserving snapshots). Backend 322/322, ruff clean,
> `models.yaml` v1.13 (no value change this phase).
> Claim-level evidence linkage from inline cites deferred to Phase 5 (needs decomposition
> changes — out of the narrowest layer). Details: `docs/IMPLEMENTATION_STATUS_2026-09-11.md`
> §13–16. Live baseline + ablations pending operator run. Next: Phase 7 is done (OCR);
> remaining: provenance/versioning, adaptive recovery, security, prod, final eval.

> Source: `docs/TRUSTRAG-UPGRADE-PLAN.md` verified file-by-file against the
> actual repo. No application code was modified to produce this plan.
> Priority: **QUALITY > RELIABILITY > SPEED > COMPLEXITY.**
> Rule: one phase at a time, test it, then move on. Prefer fixing existing
> code over adding new components.

Key config reference: `apps/api/config/models.yaml` (v1.7). All tuning knobs
below live there unless stated otherwise.

---

## 1. Current → target architecture

### Current (as built)

```text
Upload (pdf/txt/md/docx/csv/json/html, PyMuPDF native only, NO ocr)
  ↓  preprocessor.normalize_text (lowercase+stem) → chunker.py (fixed 512ch/64ov)
  ↓  pipeline._index_parsed_chunks: BGE-small-en-v1.5 (384d, local torch/ONNX)
  ↓    + TF-only hashed sparse (no IDF/BM25) → Qdrant kb_{id} (dense+sparse-text, INT8, on-disk)
  ↓
Query → dense_top_k=20 + sparse_top_k=20 (concurrent, 45s/branch) → RRF(k=60)
  ↓  temporal filter (Mongo effective_from/until) → reranker DISABLED (models.yaml:86)
  ↓  top max_context_chunks=8 → integrity audit (SHA-256 vs Mongo; web chunks auto-VERIFIED)
  ↓  generator (llama_cpp LFM2.5-1.2B default; prompt has NO citation rule; no inline cites)
  ↓  verifier: fused decompose+verify LLM-judge (1 call, temp 0, ≤8 claims)
  ↓    against the SAME top-8 chunks (no claim-specific retrieval)
  ↓  verdict (coverage≥0.80, contradiction≤0.20) → recovery (max 1 round:
  ↓    query_rewrite → re_retrieve → regenerate) → PASS/FAIL/ABSTAIN
```

Graph: linear `retrieval → generation → verification → (recover|end)` in
`apps/api/app/agent/graph.py`. No query router, no decomposition, no multi-hop.
Eval: record-only (`experiment_service` CRUD + `methodology.md` with no dataset/harness).

### Target (minimal delta)

```text
Upload (+ per-page native/OCR routing, OCR flag+confidence in payload)
  ↓  wire existing chunking strategies into pipeline; fix zoning-lowercase bug;
  ↓    sentence-aware windows, table preservation (improve, don't rewrite)
  ↓  same BGE embeddings + REAL sparse weighting (IDF via Qdrant modifier or
  ↓    BM25-sidecar) → same Qdrant collections (migration-safe, dim-pinned)
  ↓
Query → light router (simple/temporal/comparison/complex; cheap heuristics first)
  ↓  hybrid 20/20 → RRF → enforce fusion_top_k → cross-encoder rerank (flag-gated)
  ↓  top-8 + inline [Segment N] citations REQUIRED by prompt, span-checked post-hoc
  ↓  verifier: same fused path + targeted per-claim retrieval ONLY for
  ↓    NEUTRAL/contradicted claims (bounded, not every claim)
  ↓  verdict (same thresholds until calibration) → adaptive recovery (≤2 rounds,
  ↓    diagnose-then-act, token/latency budget) → PASS/FAIL/ABSTAIN
  ↓
Provenance chain doc→version→page→chunk→evidence→claim→answer (hashes already exist;
  add version + OCR-image link). Eval harness + fixed query set gates every phase.
```

### What is deliberately NOT added

GraphRAG, multi-agent orchestration, extra vector DBs, extra LLMs/rerankers,
microservices, Celery/Redis (until SSE multi-worker is proven necessary),
SPLADE/ColBERT learned sparse (revisit only if TF-IDF BM25 + rerank ablation fails),
page-as-image vision retrieval. Each rejected item gets a promotion criterion in §8.

---

## 2. Problems ranked by impact

| Rank | Problem (verified) | Quality impact | Effort |
|------|--------------------|----------------|--------|
| P0 | **No baseline/eval harness.** `methodology.md` exists, zero dataset, zero runner, `experiment_service` record-only. Every later claim is unmeasurable. | Blocks all decisions | Easy |
| P1 | **Sparse is TF-only, no IDF/BM25** (`sparse_vector.py:50-61`). Rare/common terms weighted equally; the "hybrid" leg is crippled. Research (2026 benchmarks): hybrid+RRF beats either leg; BM25 alone often beats dense. | High (recall) | Medium |
| P2 | **Reranker disabled** (`models.yaml:86`), code path exists (`reranker.py`, `graph.py`). Literature: cross-encoder rerank is the single biggest MRR jump (+0.1–0.15). Currently dead code. | High (precision) | Easy |
| P3 | **No inline citations.** Prompt (`generator.py:21-54`) never asks for `[Segment N]`; linkage only in Mongo. Unverifiable answers = trust failure (FACTUM 2026: citation hallucination is the trust-breaking mode). | High (trust) | Easy–Med |
| P4 | **No claim-specific retrieval.** All claims verified against the same top-8 (`verifier.py:796`). NEUTRAL may mean "missing evidence", indistinguishable from "false". | High (verification precision) | Medium |
| P5 | **Chunking strategies dead + zoning bug.** `pipeline.py:85-89` logs strategy, never uses it; `chunker.py:55` lowercases before `detect_chunk_zone`, so the `header` regex (`preprocessor.py:311`, requires A–Z) never fires except via `#+`. Fixed 512-char windows split tables/sentences. | Med–High | Medium |
| P6 | **Dead/misleading configs.** `fusion_top_k:20` has no reader; `max_total_tokens_per_doc` unenforced; bigrams implemented but `include_bigrams=False` everywhere incl. sparse; header/title detectors tuned to course-specific keywords. | Medium (correctness/hygiene) | Easy |
| P7 | **No OCR.** Zero OCR deps (only `pymupdf`); scanned pages index as empty strings silently (`parser.py:145-146`). Impact scoped to scanned/mixed PDFs only — do AFTER retrieval/verification if corpus is mostly native. | Med (conditional) | Med–Hard |
| P8 | **Single recovery round** (`max_recovery_attempts:1`), no diagnosis (rewrite→widen→regen blind). Contradiction/conflict path can't converge. | Medium | Medium |
| P9 | **No router/decomposition.** Every query pays full price; no temporal/comparison/multi-hop handling. `AmbiguityDetector` exists, unused in hot path. Gated behind retrieval strength (per upgrade plan). | Med (efficiency + complex-Q) | Med–Hard |
| P10 | **Provenance gaps.** Hashes exist; missing: doc versioning (no route exposes `rollback_kb_to_snapshot`), stale-point orphans on shrink re-upload (deterministic IDs overwrite same indices only), OCR→image link (blocked on P7). | Med (reliability) | Medium |
| P11 | **Security residuals** (base is solid: JWT+revocation, SSRF guards, rate limits, SSE tickets). Open: service-token tenant binding (tracked M-2), no JWT aud/iss, no red-team suite for prompt-injection/poisoned docs, no AV scan. | Med (prodirim) | Med–Hard |
| P12 | **Prod/observability gaps.** `BackgroundTasks` (no worker queue), in-process SSE (multi-worker needs bus), no `/metrics`/Prometheus/Sentry/cost accounting, k6 covers health+KB reads only, token budget not enforced pre-request. | Med (ops) | Medium |

Corrections to the upgrade plan: (a) sparse is **not BM25** — the plan says
"verify sparse implementation" and verification confirms it is TF-only hashing;
(b) reranker is **already implemented, just disabled** — Phase 3 is mostly
enablement+calibration, not construction; (c) OCR is P7 not P1 — it only pays
off on scanned corpora and adds latency/error surface to every page if misapplied;
(d) `fusion_top_k`, token caps are configured but unread — hygiene before tuning.

---

## 3. Phase order (verified, with dependencies)

```text
Phase 0  Baseline + eval harness (gates EVERYTHING) ............. Easy
  ↓
Phase 1  Sparse fix (real IDF/BM25) + enforce fusion_top_k ....... Medium
  ↓
Phase 2  Enable + calibrate reranker (flag-gated, ablation) ..... Easy–Medium
  ↓
Phase 3  Chunking: wire strategies, fix zoning bug, sentence/table .. Medium
  ↓      (re-index required after Phase 1–3; pin embedding version)
Phase 4  Inline citations (prompt rule + span post-check) ........ Easy–Medium
  ↓
Phase 5  Targeted claim retrieval + verification hardening ....... Medium–Hard
  ↓
Phase 6  Query router (cheap) → decomposition/multi-hop (gated) . Medium→Hard
  ↓
Phase 7  OCR fallback (per-page routing + provenance) ............ Medium–Hard
  ↓
Phase 8  Provenance/versioning/lifecycle (rollback route, orphans)  Medium
  ↓
Phase 9  Adaptive recovery (diagnose-then-act, ≤2 rounds, budget)  Medium–Hard
  ↓
Phase 10 Security red-team + residuals ........................... Medium–Hard
  ↓
Phase 11 Perf/prod (metrics, budgets, load coverage) ............. Medium
  ↓
Phase 12 Final ablations + cleanup + deploy checklist ............ Medium
```

Dependency notes: Phases 1–3 change the index → one coordinated re-index with
embedding-version pin (`pipeline.py:220-242` already pins; bump only on model
change). Phase 5 needs Phase 4's segment IDs. Phase 6 needs Phase 2 strength
(per upgrade plan: multi-hop only after basic retrieval is strong). Phase 7 is
independent of 1–6 and can parallelize if scanned-corpus demand is proven.

---

## 4–7. Per-phase detail (files · changes · difficulty · impact)

### Phase 0 — Baseline + eval harness [Easy]

- **Files:** NEW `apps/api/tests/eval/` (harness + `datasets/baseline_v1.jsonl`);
  `app/services/experiment_service.py`, `app/api/v1/experiments.py` (runner writes here);
  `docs/evaluation/methodology.md` (add dataset spec + run procedure).
- **Changes:** (1) freeze ≥20 native queries + ≥5 adversarial (injection/conflict/stale)
  with gold chunk IDs; (2) runner script computing Recall@K, MRR, nDCG, coverage,
  support/contradiction rate, citation correctness, abstention, P50/P95 per stage;
  (3) record `config_version` per run. No deps.
- **Quality/latency:** no product change; unblocks all go/no-go decisions.
- **Tests/acceptance:** runner is deterministic on fixed data (seeded); CI job
  `eval-baseline` fails if metrics regress >2pp vs stored snapshot.

### Phase 1 — Real sparse weighting [Medium]

- **Files:** `app/ingestion/sparse_vector.py`, `app/ingestion/pipeline.py`
  (re-index path), `app/retrieval/retriever.py` (`sparse_search`, L249-291),
  `app/db/qdrant.py` (`init_kb_collection`, L69-125), `app/core/config.py`
  (wire `fusion_top_k`, L507), `tests/test_retrieval.py`, `tests/test_ingestion.py`.
- **Changes:** Option A (preferred, no new index code): set Qdrant `Modifier.IDF`
  on `sparse-text` + generate BM25-style TF vectors client-side (k1/b saturation,
  length norm; Qdrant applies IDF at query time — current Qdrant-native pattern).
  Option B: FastEmbed `Qdrant/bm25` server-side inference. Enforce `fusion_top_k`
  truncation post-RRF (currently ignored). Keep RRF k=60. Requires full re-index
  (sparse values change shape/statistics); old collections need migration or
  versioned `kb_{id}_vN` cutover since `init_kb_collection` returns early on exists.
- **Quality/latency:** expected +5–15pp Recall@K on keyword/rare-term queries
  (literature: hybrid≫single-leg); sparse query path +~1–5 ms. Re-index is offline cost.
- **Tests/acceptance:** unit: TF saturation bounds, empty-query → `[]` preserved,
  RRF math unchanged; ablation Dense vs Sparse vs Hybrid shows Hybrid ≥ both on
  Recall@10/nDCG@10; no `RetrievalOutageError` regression on branch timeout.

### Phase 2 — Reranker enablement + calibration [Easy–Medium]

- **Files:** `apps/api/config/models.yaml` (`reranker.enabled`, `top_k`),
  `app/retrieval/reranker.py` (thresholds L25-27,53-54,80-120),
  `app/agent/graph.py` (`retrieval_node` L163-573),
  `app/core/model_registry.py` (`get_reranker` L756-798), `tests/test_retrieval.py`.
- **Changes:** flag-gated enablement (`enabled:true` in staging only); calibrate
  early-exit (`processed≥16, top≥0.85, gap≥0.15`) and adaptive top-4/8 cutoffs
  against Phase-0 set — current thresholds are uncalibrated guesses. Add
  `sentence-transformers` to runtime image if missing (check Docker stage —
  `get_reranker` returns None when absent). No new model class.
- **Quality/latency:** expected biggest single MRR/nDCG lift (+0.1 typical);
  cost +40–400 ms/query depending on candidate depth (bound depth: rerank top-20
  max; MiniLM ≈43 ms/128 passages on A100 per 2026 BEIR study). Roll back to
  disabled if P95 budget blown — decision logged per query class.
- **Tests/acceptance:** reranker-on vs off ablation; acceptance: nDCG@8 ↑
  with P95 ≤ budget (define: e.g. +300 ms); early-exit never drops gold from top-8
  on baseline set (or thresholds tightened until true).

### Phase 3 — Chunking repair (wire + fix, not rewrite) [Medium]

- **Files:** `app/ingestion/pipeline.py` (actually APPLY strategy L82-89),
  `app/ingestion/chunker.py` (word-snap, page-boundary overlap),
  `app/ingestion/chunking_strategies.py` (offset fidelity, table-row join),
  `app/ingestion/preprocessor.py` (`detect_chunk_zone` L279-314 — fix: zone BEFORE
  lowercase, or case-insensitive regex), `apps/api/config/models.yaml`
  (`chunking_strategy` key already read L304-326), `tests/test_ingestion.py`,
  `test_preprocessor.py`.
- **Changes:** (1) pipeline uses `get_chunking_strategy()` output instead of
  always-`chunk_text`; (2) fix lowercase/zone ordering bug; (3) sentence-boundary
  snap option (cheap: split on sentence ends within window); (4) layout strategy:
  join consecutive table rows into ONE chunk with `zone=table` (current code
  fragments on every table line); (5) preserve true `character_offset` in semantic
  strategy (currently reset to 0). Keep 512/64 defaults until ablation says otherwise.
  Requires re-index.
- **Quality/latency:** fewer split-table/broken-sentence chunks → generation
  grounding + citation span match improve; index-time only cost.
- **Tests/acceptance:** old-vs-new chunking benchmark on baseline set
  (Recall@K + citation-span hit rate); unit: offsets monotonic, tables unsplit,
  zone distribution sane (headers/titles actually detected).

### Phase 4 — Inline citations [Easy–Medium]

- **Files:** `app/generation/generator.py` (prompt L21-54 + post-process
  L57-152), `app/verification/verifier.py` (segment-index coercion L199-232),
  `app/agent/graph.py` (context cap L544-554), `tests/test_generation.py`,
  `test_verification.py`.
- **Changes:** (1) add citation rule to `GROUNDING_SYSTEM_PROMPT`
  ("every factual sentence ends with `[Segment N]`; N from provided segments");
  (2) post-hoc span check: cited segment must exist in `format_context` output,
  else strip/repair citation and flag claim NEUTRAL; (3) persist cited segment IDs
  per claim (extends existing `context_chunk_indices` L947-956).
  Research basis: 2026 RAGTruth/FACTUM work — evidence quotation at each step is
  the prerequisite for verifiable spans.
- **Quality/latency:** citation correctness 0→measurable; small token overhead
  (+~5%); no extra LLM calls.
- **Tests/acceptance:** citation precision/recall on baseline (cited segment
  supports the sentence per NLI spot-check ≥90%); no uncited factual sentences
  in PASS answers; ABSTAIN path unchanged.

### Phase 5 — Targeted claim retrieval + verification hardening [Medium–Hard]

- **Files:** `app/verification/verifier.py` (all paths L561-984),
  `app/retrieval/retriever.py` (new `retrieve_for_claim` reusing hybrid+RRF),
  `app/agent/graph.py` (`verification_node`, `recovery_node`),
  `apps/api/config/models.yaml` (claim-retrieval budget knobs), tests.
- **Changes:** (1) keep fused single-call fast path; (2) for claims verdict
  NEUTRAL (or CONTRADICTED with low evidence), run ONE targeted retrieval per
  such claim (claim text as query, top-5, same KB filter) then re-verify that
  claim only — bounded: max 3 claim-retrievals/analysis; (3) keep ≤8-claim cap;
  (4) wire unused `citation_correctness_weight` into a composite or DELETE it
  (remove dead config either way). No dedicated NLI encoder yet — LLM-judge
  stays; promotion to DeBERTa-v3 NLI cross-encoder only if ablation shows
  judge-precision deficit (2025 SDP result: fine-tuned NLI encoders beat LLM
  prompting for coarse hallucination detection).
- **Quality/latency:** converts false-NEUTRALs to SUPPORTED (coverage ↑,
  abstention ↓ where evidence exists); worst-case +1–3 extra retrievals + 1 NLI
  call, only on failing claims; budget-capped.
- **Tests/acceptance:** synthetic "evidence-beyond-top-8" fixture: claim whose
  support ranks 15–30 → must flip NEUTRAL→SUPPORTED; no-regression on baseline
  contradiction precision; budget never exceeded (counter test).

### Phase 6 — Router → decomposition/multi-hop [Medium → Hard]

- **Files:** NEW `app/agent/router.py` (or node in `graph.py` L1121-1146);
  `app/retrieval/retriever.py` (`AmbiguityDetector` L36-101 — adopt or remove);
  `app/agent/graph.py` (add `decompose` branch + fan-out retrieval);
  `apps/api/config/models.yaml` (router + fan-out budgets).
- **Changes:** (1) router first: regex/heuristic `simple | temporal | comparison |
  complex` — simple path = today's chain, zero added cost; temporal reuses
  `apply_temporal_filtering`; comparison = 2 parallel retrievals (entity A/B
  split); complex = LLM sub-question split (max 3) → parallel retrieval →
  merged evidence → single generation+verification. (2) hard caps: sub-questions
  ≤3, extra retrieval ≤2× base budget. Gate: implement ONLY if Phase 0–2 show
  single-hop Recall@K ≥ bar but complex-query subset fails.
- **Quality/latency:** complex/comparison coverage ↑; simple queries unchanged
  cost (router is regex-first, LLM fallback only on ambiguity).
- **Tests/acceptance:** per-class fixtures (temporal/comparison/2-hop);
  simple-query P50 latency unchanged; complex-query support rate improves vs
  no-decomposition control; fan-out budget test.

### Phase 7 — OCR fallback [Medium–Hard]

- **Files:** `app/ingestion/parser.py` (L142-146 page loop), `pyproject.toml`
  + Docker (new dep), `app/db/qdrant.py` payload schema (add `ocr_used`,
  `ocr_confidence`), `tests/test_ingestion.py` (scanned-PDF fixtures).
- **Changes:** per-page routing: native text density check → OCR (ocrmypdf or
  Tesseract sidecar; decide at implementation: ocrmypdf preserves layout best
  for mixed PDFs) ONLY low-text pages; mixed-PDF support; store
  `ocr_used/confidence` + original page-image reference for the
  `Answer → OCR chunk → page → image` chain (§7 of upgrade plan). NEVER
  blanket-OCR native pages.
- **Quality/latency:** scanned recall 0→high; native path untouched (density
  gate cost ~ms); OCR pages pay seconds at INGEST (offline), zero query cost.
- **Tests/acceptance:** native vs OCR vs OCR+verification ablation (per upgrade
  plan §12); OCR confidence < threshold → chunk flagged, excluded from PASS
  evidence or down-weighted; error-injection test (garbled OCR must not verify).

### Phase 8 — Provenance / versioning / lifecycle [Medium]

- **Files:** `app/services/kb_service.py` (`rollback_kb_to_snapshot` L419-480 —
  expose route), `app/api/v1/documents.py` + `knowledge_bases.py` (version +
  delete endpoints), `app/ingestion/pipeline.py` (orphan-point purge on
  shrink/replace: delete stale `chunk_index` points), `app/db/mongodb.py`
  (version indexes), `app/verification/integrity.py` (version check in audit).
- **Changes:** (1) document version chain + `DELETE` purges Qdrant points AND
  Mongo chunks (verify no orphans); (2) expose rollback route (exists, unrouted);
  (3) pre-vector-copy snapshots (currently roll back empty — fix or remove);
  (4) `re-index` admin path with version bump; stale evidence can never serve
  (version filter at retrieval).
- **Quality/latency:** correctness/reliability only; zero hot-path cost
  (one payload filter).
- **Tests/acceptance:** upload→update→query returns NEW text only;
  delete→query returns nothing (Qdrant point count assertion);
  rollback→query returns pre-update text; corruption-injection → CORRUPTED excluded.

### Phase 9 — Adaptive recovery [Medium–Hard]

- **Files:** `app/agent/graph.py` (`recovery_node` L892-1099, `should_recover`
  L1105-1112), `apps/api/config/models.yaml` (`max_recovery_attempts`,
  `strategy_priority`), `tests/test_agent.py`.
- **Changes:** diagnose-then-act mapping (per upgrade plan §8):
  retrieval-failure → rewrite; low coverage → expand/decompose (needs Phase 6);
  conflict → widen + compare; unsupported claim → Phase-5 claim retrieval.
  Raise cap 1→2 with global token/latency budget; abstain when budget exhausted.
- **Quality/latency:** recovery success rate ↑ on diagnosable failures;
  bounded worst-case (+1 round vs today); budget counter enforced in state.
- **Tests/acceptance:** per-failure-class fixture recovers correctly;
  budget-exhaustion → ABSTAIN (not loop); without-vs-with recovery ablation.

### Phase 10 — Security [Medium–Hard]

- **Files:** `app/core/security.py`, `app/api/v1/*` (tenant checks),
  `app/ingestion/parser.py` (AV hook), NEW `tests/test_redteam.py`,
  `docs/security/*`.
- **Changes:** (1) red-team suite FIRST: prompt-injection docs, poisoned chunks,
  conflicting/outdated docs, OCR-garbled adversarial text — must-triage ABSTAIN
  or flag, never obey embedded instructions (prompt already labels context
  untrusted; prove it); (2) close tracked residuals: service-token tenant binding
  (M-2), JWT aud/iss, login lockout; (3) upload AV scan hook (ClamAV sidecar,
  flag-gated); (4) output-policy filter (pending per status doc) if red-team
  shows exfiltration path.
- **Quality/latency:** no quality change on clean data; red-team attack success
  rate → ~0; auth changes are zero-hot-path-cost.
- **Tests/acceptance:** red-team suite green in CI; injection phrases
  ("ignore previous instructions…") never alter verdict/behavior; tenant
  isolation test (cross-KB read → 403).

### Phase 11 — Performance + production engineering [Medium]

- **Files:** `app/core/tracing.py`, `app/api/v1/health.py`, `app/core/memory.py`,
  `app/agent/graph.py` (budgets), `load-test/smoke.js` (extend),
  `docker-compose.yml`, NEW `/metrics` endpoint.
- **Changes:** only AFTER quality phases (measure, don't guess): per-stage
  latency already logged — add Prometheus `/metrics`, token/cost accounting per
  analysis, pre-request token budget enforcement, extend k6 to generation +
  verification + ingest paths, connection reuse/batch-embedding audit
  (mostly present: pooled httpx, batched `aembed_documents`, LRU caches).
  Redis/Celery ONLY if multi-worker SSE or ingest-queue pain is measured.
- **Quality/latency:** P50/P95 ↓ or documented unchanged; zero quality regression
  (baseline suite gates).
- **Tests/acceptance:** load test P95/P99 budgets defined and met;
  budget-enforcement unit tests; no metric regression vs Phase-0 snapshot.

### Phase 12 — Final ablations + cleanup + deploy [Medium]

- Run the four upgrade-plan §12 ablations (retrieval / trust / OCR / recovery)
  on the frozen baseline set; fill the production checklist (§13 — most items
  exist: Docker, health, structured logs, env config; verify each);
  remove dead code found en route (`AmbiguityDetector` if still unused,
  dead experiment router `setup_experimentation_router`, stale D-10/D-12 comments);
  lint/type/test gates green.

---

## 8. Evaluation / ablations (runs throughout, finalized in Phase 12)

Fixed dataset (Phase 0, frozen): ≥20 standard + ≥5 adversarial queries, gold
chunk IDs + gold verdicts. Metrics per run: Recall@K, MRR, nDCG, evidence
coverage, claim support rate, contradiction rate, citation correctness,
abstention rate, recovery success, P50/P95 per stage + total, tokens/cost.

| Ablation (upgrade-plan §12) | Compares | Gate for |
|---|---|---|
| Retrieval | Dense vs Sparse vs Hybrid vs Hybrid+Rerank | Phases 1–2 |
| Chunking | old vs new | Phase 3 |
| Trust | Normal vs Verified vs full TRUSTRAG | Phases 4–5 |
| OCR | native vs OCR vs OCR+verification | Phase 7 |
| Recovery | without vs with | Phase 9 |
| Router | single-hop vs decomposed (complex subset) | Phase 6 |

Promotion criteria for rejected components: SPLADE/learned sparse only if
BM25+rerank still loses keyword queries by >5pp; dedicated NLI encoder only if
judge contradiction precision < bar on gold verdicts; ColBERT late-interaction
only if rerank-latency budget is unmeetable; Celery/Redis only if measured
queue/SSE pain. Each requires a failing ablation, not enthusiasm.

---

## 9. Production / security risks

1. **Re-index coordination (Ph. 1–3).** Sparse-value change + chunking change
   invalidate existing Qdrant points; `init_kb_collection` early-returns on
   existing collections with NO migration. Risk: mixed old/new points silently
   degrade ranking. Mitigation: versioned collection cutover (`kb_{id}_vN`) or
   offline rebuild with dim/stat assertions before swap.
2. **Embedding-space mismatch (existing).** `retriever.py:203-228` truncates/
   zero-pads foreign-dim vectors with a warning — garbage scores, not errors.
   Mitigation: keep KB pin guard; fail closed (422) on mismatch; never rely on
   pad/truncate across models.
3. **Dose-dependent latency (Ph. 2, 5, 6).** Rerank + claim retrieval + fan-out
   each add hundreds of ms worst-case. Mitigation: per-phase P95 budgets,
   flag-gated rollouts, cheap-path-first (heuristics before LLM).
4. **OCR error injection (Ph. 7).** Garbled text becomes confidently-indexed
   evidence. Mitigation: confidence threshold + flag, verification weight-down,
   adversarial OCR fixtures in CI.
5. **Verification judge over-trust (Ph. 5).** 1.2B local judge is the weakest
   link for contradiction detection. Mitigation: gold-verdict spot checks every
   phase; contradiction early-exit preserved; NLI-encoder promotion criterion.
6. **Stale/poisoned evidence (Ph. 8, 10).** Orphan points + unrouted rollback +
   no AV/red-team = stale or malicious chunks servable. Mitigation: lifecycle
   tests asserting point counts; red-team suite before prod exposure.
7. **Single-flight ingestion + in-process SSE (existing).** Throughput ceiling
   and multi-worker fan-out gap. Mitigation: acceptable until measured pain;
   no premature queue/bus work (Phase 11 measures first).
8. **Secret/config drift.** Tunables in `models.yaml`, secrets in `.env`,
   ports in `ports.yaml` + compose override (`host.docker.internal` Mongo vs
   Atlas-only decision D-03 — reconcile and document the supported topology).

---

## Appendix — Verification notes (plan vs code)

- Pipeline diagram confirmed: parser → 512/64 chunker → BGE-384 + hashed sparse
  → RRF → (disabled) reranker → llama_cpp default → fused LLM-judge verify →
  1-round recovery. Gemini appears only if selected; default is local LFM2.5-1.2B.
- "Sparse retrieval" is TF-only (`sparse_vector.py`), NOT BM25 — upgrade-plan
  "verify sparse implementation" resolves as CONFIRMED GAP.
- Chunking strategies (`sliding/semantic/progressive/layout`) exist but the
  pipeline ignores the selection (`pipeline.py:85-89`) — wire-up task, not greenfield.
- `fusion_top_k`, `max_total_tokens_per_doc`, bigram support, reliability
  composite weights are configured but unread — hygiene items folded into Ph. 1/3/5.
- Security baseline is stronger than the upgrade plan implies (JWT revocation,
  SSRF allowlist+DNS pinning, SSE tickets, rate limits) — Phase 10 is residuals +
  red-team, not greenfield auth.
- Research consulted (2026): hybrid+RRF+cross-encoder rerank dominance
  (T2-RAGBench: R@5 0.695→0.816 with rerank); RRF k=60 standard, no tuning needed;
  claim-level NLI (entail/contradict/neutral) with fine-tuned encoders beats LLM
  prompting for hallucination detection; evidence-quotation prerequisite for span
  verification (RAGTruth/RLSeek/RT4CHART); Qdrant native hybrid (prefetch + RRF
  fusion query, IDF modifier) matches the Phase-1 approach; rerank-everything is
  485–2051× retrieval cost — always bound candidate depth.
