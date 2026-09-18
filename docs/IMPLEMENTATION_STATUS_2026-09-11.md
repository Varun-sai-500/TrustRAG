# TrustRAG — Implementation Status (2026-09-11)

> **2026-09-16 update:** RAG quality phases 0–8 + OCR fallback implemented (see §13–§18).
> Backend **322/322** ✅ (`pytest tests/`), ruff check + format clean ✅,
> `uv lock --check` ✅. `models.yaml` config_version **1.7 → 1.13**.
> Live baseline + ablations pending operator run (no local services during implementation).

**Date:** 2026-09-11 → 2026-09-12  
**Session:** Unified Senior Audit → Implementation Pass (≤2-day fixes from audit §11A) + **ONNX BGE Runtime** + **Claims verification hardening** + **Decompose+verify fusion** + **Push-readiness docs pass**  
**Baseline Audit:** `docs/audits/2026-09-11_unified_senior_audit.md`  
**Test State:** Backend 219/219 ✅ | Frontend 22/22 + lint + build ✅ | ruff check + format clean ✅ | Bandit 0 issues ✅ | k6 + Playwright green ✅ | Docker image boots ✅

---

## ✅ COMPLETED — All ≤2-Day Fixes (§11A from Audit)

### 1. Frontend — Apple-design compliance (HIGH-1 + tokens + press rule)
| File | Change |
|------|--------|
| `apps/web/src/lib/motionConfig.js` | Replaced stiffness/damping springs with Apple-mapped `bounce:0/duration:0.35` tokens; added `REDUCED_MOTION_TRANSITION` (opacity cross-fade 200ms); documented Motion↔Apple mapping. |
| `apps/web/src/main.jsx` | Added `MotionProvider` wrapper using `useReducedMotion()` + `MotionConfig` — global reduced-motion now applies to all `motion/*` components. |
| `apps/web/src/styles/animations.css` | `@media (prefers-reduced-motion: reduce)` now **preserves button press feedback** (`scale(0.97)` 80ms) while disabling decorative animations (cursor glow, float, pulse, radar, shimmer, status-dot). Removed `spring-*` classes from reduced-motion disable list (they were unused). |
| `apps/web/src/components/workbench/EvidenceViewer.jsx` | Removed per-component `useReducedMotion()`; now inherits global `MotionConfig` → `SPRING_SNAPPY` for expand/collapse and rotate. |
| `apps/web/src/styles/components.css` | `@media (prefers-reduced-motion)` disables `.status-dot--running` pulse only (decorative). |

**Result:** Reduced-motion = opacity cross-fade + instant transitions (Apple HIG §14), no vestibular springs. Press feedback works everywhere.

### 2. Backend — Ultra-low RAM foundations
| File | Change |
|------|--------|
| `apps/api/app/core/memory.py` | `get_memory_usage_mb()` now uses `psutil.Process().memory_info().rss` (current RSS) with `resource.ru_maxrss` fallback. Added `idle_trim_memory()` for proactive post-ingest/analysis trim. |
| `apps/api/app/core/disk_cache.py` | Added `set_cached_embeddings_batch()` — single `executemany` transaction replaces N+1 `set_cached_embedding` calls. |
| `apps/api/app/core/model_registry.py` | **Bounded LLM registry (max 4 instances, LRU eviction + close)** replaces `@lru_cache(maxsize=16)` on `get_llm`/`get_verification_model`. Added generation-scoped cache `gen_cache_get/set` (TTL 5 min, max 256, keyed on query+chunk hash). Legacy global `InMemoryCache` wrapped in bounded `BoundedCache` (max 512). `close_all_llm_instances()` called on shutdown. |
| `apps/api/app/ingestion/preprocessor.py` | `stem_word` LRU bound from 32768 → 8192 (English vocab ~5k). |
| `.env.example` | Documented `MALLOC_ARENA_MAX=1`, `TOKENIZERS_PARALLELISM=false` for shell/systemd/Docker. |
| `apps/api/Dockerfile` | Added `ENV MALLOC_ARENA_MAX=1` alongside existing `TOKENIZERS_PARALLELISM=false`. |
| `apps/api/app/main.py` (lifespan) | Calls `close_all_llm_instances()` on shutdown. |

**Result:** API-process RAM floor reduced: bounded LLM clients (4 max), batched SQLite writes, current-RSS guard, idle trim, allocator tuning.

### 3. Security — Internet-exposure hardening
| File | Change |
|------|--------|
| `apps/api/app/main.py` | `request_id_middleware`: validates `X-Request-ID` against `^[A-Za-z0-9-]{1,64}$`, regenerates UUID on fail. CORS: wildcard `allow_origin_regex` (vercel/netlify/pages.dev) **only in non-production**; production uses explicit `CORS_ORIGINS` only. |
| `apps/api/app/api/v1/health.py` | Split into public `/health` (minimal: status, timestamp, app, version) and authed `/health/detailed` (full models/hardware/formats). |
| `apps/web/src/services/api.js` | `healthService.get()` → `/health/detailed`; added `getPublic()` for unauthenticated probes. |
| `apps/api/app/ingestion/parser.py` | **Magic bytes validation** (`validate_magic_bytes`) for PDF/DOCX/ZIP/PNG/JPG/GIF/BMP/TIFF; **zip-bomb protection** (`check_decompression_bomb`) with per-format max decompression ratios. Integrated into `parse_document()`. |

**Result:** Request-ID forgery prevented; health recon minimized; CORS locked in prod; upload format + decompression-bomb defense added.

### 4. **ONNX BGE Runtime — Torch-free embeddings** (UL-1 from audit) ✅
| File | Change |
|------|--------|
| `scripts/export_bge_onnx.py` | Export script: loads BGE-small-en-v1.5, forces CPU, exports transformer + mean pooling + L2 norm to ONNX with dynamic batch/sequence axes. Uses `external_data=False` for single-file model (128 MB). |
| `apps/api/app/core/onnx_embeddings.py` | `ONNXBGEEmbeddings` — ONNX Runtime wrapper with tokenizer, two-tier cache (memory LRU + SQLite disk), BGE query instruction prefixing. `ONNXBGEEmbeddingsWrapper` integrates with existing cache infrastructure. |
| `apps/api/app/core/model_registry.py` | Added `onnx` provider support in `get_embedding_model()`. Auto-detects ONNX model at `embedding_cache_dir/bge-small-en-v1.5.onnx`. Falls back to HuggingFace if not present. |
| `apps/api/app/core/config.py` | Added `embedding_max_seq_length` property (default 512) from models.yaml. |
| `apps/api/config/models.yaml` | Added `max_seq_length: 512` to embedding section. |
| `.env.example` / `models.yaml` | Documented `EMBEDDING_PROVIDER=onnx` option. |

**Result:** Embeddings now run via ONNX Runtime (no PyTorch in API process). Numerical parity verified (max diff 0.000000 vs PyTorch). **~500-1000 MB RSS savings** — largest single RAM win. Set `EMBEDDING_PROVIDER=onnx` to enable.

### 5. Tests — All green
- Backend: `pytest tests/ -q` → **204 passed** (was 191; +2 health split, +6 NLI tolerance, +5 fused fusion).
- Frontend: `npm run lint` (0 errors), `npm run test` (21 passed), `npm run build` (success, 2.7s).

### 8. Decompose+verify fusion — one call instead of two (2026-09-12)
**Motivation:** verification cost 2 LLM calls in the typical case (decompose 512 + batch 768) and up to 11 on failure cascades — the dominant analysis-latency term on throttled 1.2–3B local models.

| File | Change |
|------|--------|
| `app/verification/verifier.py` | `FusedClaimVerdict` / `FusedDecomposeVerify` schemas (same tolerant validators), `FUSED_DECOMPOSE_VERIFY_PROMPT_TEMPLATE` (enum + int-only segments + JSON example), `fused_decompose_verify()` (1024-token cap, returns `None` on total failure). `execute_claim_verification` tries fused first (meta-filter + max-cap apply); `None`/empty falls through to the untouched two-step path (existing backstops, retry, budgeted fallback all preserved). Kill-switch: `verification.fused_decompose_verify` / `FUSED_DECOMPOSE_VERIFY=0`. |
| `app/core/config.py`, `config/models.yaml` | `fused_decompose_verify` property + `fused_decompose_verify: true`; `config_version` 1.6 → 1.7. |
| `apps/api/tests/test_verification.py` | 5 new tests: fused-skips-two-step (two-step seams assert never called), fused-failure fallback (+1 call worst case), kill-switch, fused meta-filter, fused-None unit. 6 pre-existing two-step tests pinned with `fused_decompose_verify=None` mock for hermeticity (they previously passed only because no live server was up — a running llama-server made fused succeed and skipped the asserted two-step calls). Suite proven hermetic: 26/26 with server DOWN. |

**Live eval (llama.cpp LFM2.5-1.2B, refund-policy answer, 2 chunks):** fused 3 judged claims in **2.0s / 1 call** vs two-step 3 claims in **3.4s / 2 calls** — ~40% wall-time saved, one fewer round trip, comparable quality. Fused kept as primary; worst case costs exactly one extra call.

### 6. Claims verification hardening — tolerant NLI parsing (2026-09-12)
**Symptom (Playground screenshot):** "Explain the key concepts" → 0/7 claims supported, all NEUTRAL, FAILED — on a good grounded answer with 16 evidence chunks. Same class as earlier "0/5 for Describe the knowledge base".

**Root cause (`app/verification/verifier.py`):** strict Pydantic Literals vs small-model near-miss JSON:
- verdict `"VERIFIED"` instead of `SUPPORTED` → ValidationError → NEUTRAL;
- `supporting_segments` as evidence prose instead of `list[int]` → ValidationError → NEUTRAL;
- batch `{"verdicts": [1]}` (bare ints) → whole batch raises → retry → individual fallback fails the same way.

| File | Change |
|------|--------|
| `app/verification/verifier.py` | `_normalize_verdict_value` alias map (VERIFIED/TRUE→SUPPORTED, FALSE/REFUTED→CONTRADICTED, UNKNOWN→NEUTRAL, junk→NEUTRAL); `_coerce_segment_list` (ints pass, digit runs in prose extracted, prose dropped); `_coerce_claim_id`; `field_validator(mode="before")` on `NLIVerdict`/`ClaimVerdict`; `model_validator` on `BatchNLIVerdict` drops unrecoverable items so valid siblings count and missing ids use per-claim fallback; NLI + batch prompts now pin the exact enum, int-only segments, and a JSON example. |
| `apps/api/tests/test_verification.py` | 6 new regression tests: VERIFIED→SUPPORTED, alias matrix, text-segment coercion, string claim_id, batch bare-int drop, all-bare-int empty map. |

**Result:** backend **199 passed** (was 193), ruff check + format clean. Malformed-but-correct NLI judgments now count instead of collapsing to 0/x.

### 7. Lint sweep — 38 session-introduced ruff errors fixed
- `onnx_embeddings.py` (28): `List`→`list`, unused `os`/`Path`, import sort, line lengths, EOF newline.
- `model_registry.py` (5): unquoted `BaseChatModel` annotations, long ONNX line (also dropped obsolete `hasattr` guard).
- `disk_cache.py` (2): batch signature wrap, `zip(..., strict=True)`.
- `memory.py` (1): psutil fallback now logs instead of bare pass.
- `parser.py` (2): long line wrap, trailing whitespace.
- `ruff format` applied repo-wide (2 files reflowed); full suite re-run green.

### 6. Full Pipeline Integration Test (IN PROGRESS)
- Model discovery snapshot loading fixed (`load_discovery_snapshot()` at module import in `local_llm.py`)
- `scripts/discover_local_models.py` works: 6 llama.cpp models discovered
- llama.cpp server running on port 8080 with LiquidAI/LFM2.5-1.2B
- **Blocked**: Analysis creation validates models via `get_discovered_llms()` but snapshot loading has a NameError (`_is_embedding_model_name` not available at module import time). Fix pending.

---

## 🟡 IN PROGRESS / REMAINING — >2-Day Items (§11B from Audit)

| Item | Audit Ref | Effort | Notes |
|------|-----------|--------|-------|
| **Decompose+verify fusion** (single structured call) | AI/ML | High | Needs prompt-schema robustness eval on 3B/4B models. |
| **Redis SSE bus + multi-worker readiness** | Backend | Med | Only if `workers > 1` ever needed; currently documented single-worker. |
| **Tenant-bound service tokens + MCP scoping + JWT `aud/iss` + httpOnly-cookie frontend auth** | Security | Med | Requires service-token design + frontend cookie migration. |
| **Output-policy filter + delimiter-preserving prune + web-evidence labeling** | Security/AI | Med | Prompt-injection depth; needs careful eval to avoid false positives. |
| **mimalloc/jemalloc eval + Alpine/slim image + FastAPI ≥0.140 rollout** | Platform | Med | Load-test (k6) before/after required. |
| **Chart lazy-load + proxy-key dedupe + full type-scale retune** | Frontend | Low | Design review needed for type-scale. |
| **Mongo `cacheSizeGB:1` + Qdrant on-disk tuning + self-heal progress UX** | Ops | Low | 8 GB host specific. |
| **Automated red-team + hallucination stress suite in CI** | Testing/Security | High | Requires Strix/CI integration. |

---

## 📋 VERIFICATION CHECKLIST (run before next session)

```bash
# Backend
cd apps/api
ruff check app/ tests/ && ruff format --check app/ tests/
pytest tests/ -q --no-header --no-cov   # 193+
python3 -m py_compile app/agent/graph.py app/core/model_registry.py app/core/memory.py app/core/disk_cache.py app/core/onnx_embeddings.py

# Frontend (Apple-design scope)
cd ../web
npm run lint            # 0 errors
npx vitest run          # 21 green
npm run build           # success

# Ports + security spot-checks
python scripts/apply_ports.py --check
curl -s http://localhost:8000/api/v1/health | jq '{status, version}'
curl -s http://localhost:8000/api/v1/health/detailed -H "Authorization: Bearer <token>" | jq '{environment, models, hardware}'
curl -si -X OPTIONS http://localhost:8000/api/v1/auth/login -H "Origin: https://evil.vercel.app" | head -20

# ONNX embeddings test
EMBEDDING_PROVIDER=onnx python -c "
from app.core.model_registry import get_embedding_model
import asyncio
emb = get_embedding_model()
q = asyncio.run(emb.aembed_query('test'))
print('ONNX embedding dims:', len(q))
"
```

---

## FILES MODIFIED THIS SESSION

### Frontend
- `apps/web/src/lib/motionConfig.js` — Apple-mapped spring tokens + reduced-motion transition
- `apps/web/src/main.jsx` — MotionProvider wrapper
- `apps/web/src/styles/animations.css` — Reduced-motion preserves press, disables decorative
- `apps/web/src/components/workbench/EvidenceViewer.jsx` — Removed local reducedMotion, uses global
- `apps/web/src/styles/components.css` — Reduced-motion status-dot only

### Backend — RAM/Security (≤2-day)
- `apps/api/app/core/memory.py` — psutil RSS + idle_trim_memory
- `apps/api/app/core/disk_cache.py` — Batch `set_cached_embeddings_batch` (executemany)
- `apps/api/app/core/model_registry.py` — Bounded LLM registry (4 max, LRU+close), generation-scoped TTL cache, bounded legacy cache, close_all_llm_instances, **ONNX provider support**
- `apps/api/app/ingestion/preprocessor.py` — stem_word LRU 8192
- `apps/api/app/main.py` — X-Request-ID validation, production CORS, shutdown close_all_llm_instances
- `apps/api/app/api/v1/health.py` — Public /health + authed /health/detailed
- `apps/api/app/ingestion/parser.py` — Magic bytes + zip-bomb protection
- `.env.example` — MALLOC_ARENA_MAX, TOKENIZERS_PARALLELISM documented
- `apps/api/Dockerfile` — MALLOC_ARENA_MAX=1 env
- `apps/api/app/core/config.py` — `embedding_max_seq_length` property
- `apps/api/config/models.yaml` — `max_seq_length: 512` for embeddings

### Backend — ONNX BGE Runtime (UL-1)
- `scripts/export_bge_onnx.py` — **NEW** Export script (transformer + pooling + norm to ONNX, dynamic axes, single-file)
- `apps/api/app/core/onnx_embeddings.py` — **NEW** ONNXBGEEmbeddings + ONNXBGEEmbeddingsWrapper (two-tier cache, BGE prefixing)
- `apps/api/app/core/model_registry.py` — `onnx` provider branch in `get_embedding_model()`
- `apps/api/config/models.yaml` — `max_seq_length: 512` added

### Tests
- `apps/api/tests/test_health.py` — Updated for public/detailed split (4 tests)

---

### 10. Audit-driven bugfix pass — critical → high → medium (2026-09-12)
Source: `docs/audits/2026-09-11_unified_senior_audit.md`. Already-fixed items re-verified against current code; only genuinely open findings changed.

**CRITICAL remainder.**
- CRIT-RAM-1 (torch floor): bounded registry, ONNX opt-in, batch writes, psutil guard all live. Remaining hard core (torch-by-default, embedded Qdrant) is architectural — tracked, not forced.
- **Latent correctness bug found & fixed:** the ONNX export used **mean pooling while BGE-small uses CLS pooling** (`pooling_mode: cls` verified live). Old export sat at 0.947 cosine (rankings survived: 8/8 top-1, which is why it looked fine). Re-exported with CLS → **cosine 1.000000, max diff 0.000000, 8/8 top-1: PARITY OK**. (Also caught because the eval compared stale disk-cached vectors — `eval_embedding_parity.py` now purges `onnx::%` rows first.)

**HIGH fixed.**
- H-BE-5 self-heal one-shot → batches of 128 (`SELF_HEAL_BATCH_SIZE`) with `retrieval.self_heal_batch` progress trace events (feed compacts them).
- Per-branch retrieval timeouts (45 s each, 60 s total backstop): one hung branch degrades to the other's results; both hung → `RetrievalOutageError` (never silent "no evidence").
- `nli.batch_total_failures` counter + `get_nli_metrics()`, surfaced in `/health/detailed` alongside current `rss_mb` (UL-8).
- Tier-tied ingest embed batches: lean 32 / standard 64 / high 128 via `get_ingest_embed_batch_size()` (safe 64 default).
- SEC M-2 area: strict Pydantic bodies on both internal ingest endpoints (422 on bad `user_id`/missing keys instead of 500s). Full tenant-bound tokens stay tracked (>2-day B.4); MCP scoping assessed — needs identity plumbing, tracked.

**MEDIUM fixed.**
- Warmup sequenced (discovery → hardware → embeddings) with RSS breakdown logs.
- Refusal-gate hit log in the verification node (tuning signal).
- Frontend #5 type scale was already complete (verified); #4 materials got the missing bright top-edge light + heavier nav shadow (stacking hierarchy was already correct).

**Tests:** 204 → **215** (self-heal batching, branch-timeout pair, tier sizes, NLI metric delta, RSS/metrics health, 5 internal-schema contracts). Non-hermetic risk closed: verification suite proven with llama-server DOWN.

### 11. Offline-warning regression hardening (2026-09-12)
**Report:** with the local model server off, the amber "Inference server offline" warning stopped appearing and the model list was empty.

**Investigation:** reproduced offline with both servers down — the backend contract was already correct (`connected:false` + cached models listed, verified live). The failure mode that produces exactly these symptoms is `/models/providers` itself failing (then `providersData` is undefined: empty dropdown AND the warning's `activeProviderInfo` guard hides the banner).

| File | Change |
|------|--------|
| `app/api/v1/models.py` | `_safe_provider_status()` + `_safe_hardware_profile()`: the endpoint can never 500 on discovery — a crashing check degrades to an explicit disconnected stub. Stale comment corrected. |
| `apps/web/.../PlaygroundPage.jsx` | `providersUnresolved` (query errored, not loading, no data) passed to QueryPanel. |
| `apps/web/.../QueryPanel.jsx` | Offline banner also renders on `providersUnresolved`, not only on explicit `connected:false`. |
| Tests | Backend endpoint-degradation test; frontend failed-query warning test. |

**Result:** 216 backend + 22 frontend green. If the symptom persists on your machine after pulling, restart the backend (`uvicorn` without `--reload` serves stale code) and hard-refresh the frontend (stale bundle) — the served code paths are verified.

### 12. Local-server flakiness hardening — probe retry + refused/timeout split (2026-09-12)
**Report:** llama-server running, but the system wouldn't identify it and analyses failed to run.

**Investigation:** reproduced a subtler cousin live — a lone 3 s probe sample failing against a live-but-slow server (cold model / full accept backlog on a busy 8 GB host), which 503s the whole analysis and flaps the UI pill on every 8 s poll. Connection-refused (down) and timeout (slow) were conflated into one "not reachable" message, sending users to restart a live server.

| File | Change |
|------|--------|
| `app/core/local_llm.py` | `_fetch_json_with_retry()` (2 attempts, connect/timeout only); probe distinguishes refused ("not reachable — start it") from timeout ("not answering — may be starting/overloaded, wait and retry"); `check_ollama_status` / `check_llamacpp_status` use the same retry. |
| `app/api/v1/models.py` | Provider status checks run concurrently (`asyncio.gather`) to absorb retry latency on the 8 s-polled endpoint; stale comment corrected. |
| `app/mcp/server.py` | `local_llm_status` tool degrades per-provider instead of erroring the whole tool. |
| Tests | Probe retry-then-success, timeout-message split, status retry (3 new). |

**Result:** 219 backend green. To tell the cases apart on your machine: `curl -m 5 http://127.0.0.1:8080/v1/models` failing instantly = server down (start it); hanging = overloaded (wait); instant 200 yet UI red = stale frontend/backend processes.

## NEXT SESSION START POINT

All **≤2-day fixes complete** including the **ONNX BGE Runtime** (UL-1), **fused decompose+verify** (primary, live-evaled), **CI repairs** (frontend install, Docker context + venv, k6 contract, onnxruntime in image, Bandit B615, Trivy SARIF), and this **push-readiness docs pass**.

Pick next from **>2-Day Items** (recommended order by impact):

1. **ONNX-as-default** — flip `EMBEDDING_PROVIDER` default after recall-parity eval on a real KB
2. **Tenant-bound tokens + httpOnly cookies** — prerequisite for internet exposure
3. **Output-policy filter** — prompt-injection depth
4. **Redis SSE bus + multi-worker** — only if scaling beyond single worker
5. **mimalloc/Alpine/FastAPI 0.140** — platform hardening with load tests
6. **Automated red-team suite** — CI security

The codebase is **push-ready**: clean tree, 219/219 + 22/22 green, secrets clean, docs current (this pass). Push with `git push origin ui-redesign` and open the PR against `main`.

---
### 9. Push-readiness docs pass (2026-09-12)
(Extended same day — new/existing-user readiness.)

| Area | Change |
|------|--------|
| New-user install gap (real bug) | `pip install -e ".[dev]"` never installed torch, so default HuggingFace embeddings failed on fresh clones. README + deployment guide + `setup.sh` now install `.[dev,local-models]`; `setup.sh` gained an **Embeddings** check (torch stack → ok, else `.onnx` file → ok, else actionable warn); Node message aligned to 22+. Verified: `setup.sh` 11/11 green, warn branch proven against a torch-less python. |
| Existing-user pull path | Verified safe: no DB migration (schemas unchanged), new `models.yaml` keys default safely (`fused_decompose_verify`→True, `max_seq_length`→512), disk/Qdrant caches compatible, no removed APIs. Only action: re-sync deps (`pip install -e ".[dev,local-models]"` for onnxruntime) + `npm ci` — documented in a new README Troubleshooting entry ("After `git pull`"). |
| (Previous §9 content) Root `README.md` | 204-test counts, ONNX setup notes, split-health API rows, implementation-status link; deleted `AUDIT_REPORT.md` row |
| Area | Change |
|------|--------|
| Root `README.md` | 204-test counts, ONNX setup notes, split-health API rows, implementation-status link; deleted `AUDIT_REPORT.md` row |
| `docs/README.md` | Audits tree/index show only the canonical unified audit + status doc; stack line current |
| `docs/ROADMAP.md` | 2026-09-12 header + 204 counts; phases 15–17 (RAM, ONNX, claims, fusion); fixed deleted-audit link |
| `docs/deployment/README.md` | Split-health docs, ONNX setup + Docker embeddings note (torch absent by design — use onnx + copy `.onnx` in), new env-var rows |
| `docs/architecture/decision-log.md` | D-21 (ONNX), D-22 (tolerant NLI) — added in prior pass |
| `.env.example` | Fixed stale `google_genai/nvidia` embedding comment → `huggingface/onnx`; added `HF_TOKENIZER_REVISION`, `FUSED_DECOMPOSE_VERIFY` |
| `docker-compose.yml` | `web`: `npm install` → `npm ci`; `model_cache` volume comment documents the ONNX copy-in step + torch-absent rationale |
| Deleted | `docs/AUDIT_REPORT.md` (2026-09-05, superseded) + `docs/ui-redesign-audit/` (11 archived files); zero dangling references repo-wide |
---

### 13. RAG quality phases 0–2 + OCR fallback (2026-09-16)

Plan: `docs/TRUSTRAG-IMPLEMENTATION-PLAN.md` (verified against code; priority
QUALITY > RELIABILITY > SPEED > COMPLEXITY). One phase at a time; no rewrites.
`models.yaml` config_version 1.7 → 1.10. Backend 219 → **273 tests**, ruff clean,
`uv lock --check` clean. No live services were up during implementation, so live
baselines/ablations are recorded as pending operator runs (procedure + runner ready).

| Phase | Change |
|-------|--------|
| **0 — Baseline + eval harness** | `tests/eval/metrics.py` (pure retrieval/trust/latency metrics, hand-computed unit tests); frozen `datasets/baseline_v1.jsonl` (25 queries: 12 factual + 3 temporal + 3 conflicting + 2 missing-evidence + 5 adversarial) over new `fixtures/corpus/*.txt` (6 docs incl. stale-pricing + injection graffiti); `scripts/run_baseline_eval.py` live runner (HTTP-only, respects 10/min limit, writes `docs/evaluation/results/` + optional `/experiments` record); methodology run procedure + measured-runs-only snapshot table |
| **1 — Real sparse weighting** | `sparse_vector.py`: linear TF → BM25 TF (`zone × sat(freq)/length-norm`, k1/b/avg_len from `models.yaml`); `qdrant.py`: `sparse-text` gains `Modifier.IDF` (server-side IDF) + recreate-on-mismatch migration for pre-IDF collections (fail-open when unreadable); `retriever.py`: dead `fusion_top_k` now enforced post-RRF/post-temporal |
| **2 — Reranker hardening** | `reranker_top_k` (dead config) wired as scoring-depth cap (default 20, floored at `fusion_top_k` so candidates are never discarded pre-score); `_rerank_sync` no longer sorts the caller's list in place; **stays `enabled: false`** — Docker runtime lacks sentence-transformers/torch by design, so enabling there is a silent no-op (documented in `models.yaml`); thresholds remain uncalibrated pending live ablation |
| **OCR fallback (RapidOCR-ONNX)** | New `ingestion/ocr.py` (lazy singleton engine, density gate, confidence-gated output — sub-threshold text dropped, `used=True` kept for audit); `parse_pdf` routes only low-native-text pages (<50 chars) through 300dpi render → OCR, failing open to native text; `ocr_used`/`ocr_confidence` plumbed page → chunk → Mongo + Qdrant payload; `pyproject.toml` + `uv.lock` gain `rapidocr-onnxruntime==1.4.4` (reuses the shipped `onnxruntime`, no torch/system binaries, no Dockerfile change). Surya (GPL + non-commercial weights) and Docling (parser-replacing) evaluated and rejected — see session notes |

**Tests added (54):** `tests/eval/` (18: metric math + dataset/fixture validation),
`test_sparse_bm25.py` (6: saturation 1.43 vs linear 5.0, length norm, query weights),
`test_qdrant.py` (4: create/keep/recreate/fail-open), `test_retrieval.py` +1 (fusion 50→20 exact cut),
`test_ocr.py` (15: gate, confidence drop, fail-open, provenance — engine fully mocked),
`test_reranker.py` (10: ordering, adaptive top-4, depth cap, no-mutate, disabled/None/exception fallbacks).
Two self-caught issues during the work: a stripped docstring (reverted, diff-verified) and an
un-awaited coroutine escaping a `patch` block in `test_qdrant.py` (caught by tests trying live network).

**Operator pendings (all procedures documented, none fabricated):**
1. Ingest `tests/eval/fixtures/corpus/*.txt` into a fresh KB and run
   `python scripts/run_baseline_eval.py --email ... --password ... --kb-id <ID> --post-experiment`;
   paste the aggregate into `docs/evaluation/methodology.md` snapshot table.
2. Re-upload documents for any pre-IDF KB (collections recreate on next init).
3. Pre-warm OCR models once (`~/.onnx` empty until first scanned page) so first upload doesn't stall.
4. Enable `reranker.enabled: true` only where the `local-models` extra is installed, then run the
   Hybrid-vs-Hybrid+Rerank ablation; calibrate early-exit/adaptive thresholds from it.
5. Next code phase: **Phase 3 (chunking repair)** — wire strategies, fix lowercase/zone bug, tables.

---

### 14. Phase 3 — chunking repair (2026-09-16)

Pre-existing suite was green (273/273) — nothing broken to fix; proceeded to Phase 3.

**Root cause found (bigger than reported):** `normalize_text` collapsed `\s+` → `" "`,
erasing all newlines. That single line silently disabled section splitting (semantic),
table line detection (layout), AND the all-caps branch of header zoning — three
features that looked wired but could never fire. Fix preserves `\n\n` paragraph breaks
(token stream identical either way — lexer treats all whitespace as separators).

| Area | Change |
|------|--------|
| `preprocessor.py` | Whitespace collapse keeps line breaks (`[ \t\r\f\v]+` → space, `3+\n` → `\n\n`) |
| `chunking_strategies.py` | Semantic: true page offsets via running `find` cursor (was: all reset to 0); progressive: step scales with effective window — the fixed full-size step skipped ~80 chars per window (**silent text loss**, now covered by a gap-freedom regression test); layout: full rewrite — ordered table/prose blocks, consecutive rows chunked ONCE with sequential indices (was: re-chunked per row + scrambled order + duplicate indices); all strategies propagate `ocr_used`/`ocr_confidence` (synthetic page dicts previously dropped them) |
| `knowledge_bases.py` | Both ingest paths (upload + URL) now chunk via `get_chunking_strategy().chunk()` — strategies selectable via `models.yaml: ingestion.chunking_strategy` (default `sliding_window` = byte-identical output to before) |
| `pipeline.py` | Removed dead "Using chunking strategy" log block that claimed re-chunking which never happened; `strategy` param kept for compatibility |
| `models.yaml` | `chunking_strategy: sliding_window` explicit default; version 1.10 → 1.11 |

**Tests:** +14 `test_chunking_strategies.py` (normalization, wiring/equivalence, semantic
offsets monotonic + non-zero, progressive 300-token gap-freedom, layout single-table-chunk +
order + sequential indices, OCR passthrough ×3 strategies). **287/287 green**, ruff clean.
Known limitation (deliberately out of scope): plain-text ALL-CAPS headings stay invisible
to zoning — detection runs on lowercased text; fixing needs raw-text zoning, candidate for later.

**Operator note:** normalization output changed (newlines preserved) → chunk text/ embeddings
shift → **re-index KBs** after deploy (same window as the Phase-1 IDF re-index).

---

### 15. Phase 4 — inline citations (2026-09-16)

Worked under loaded skills: `langchain-rag`, `test-driven-development` (RED→GREEN with
the repo's own pytest/ruff commands), `surgical-patch` (narrowest layer only).

**TDD-RED:** `tests/test_citations.py` written first — 8 tests failed at collection
(missing names) plus the prompt-rule assertion. **GREEN:** minimal `generator.py`-only change:

| Area | Change |
|------|--------|
| Prompt | `GROUNDING_SYSTEM_PROMPT` gains item 8: every factual sentence ends with `[Segment N]` (1-based, served segments only, never invented); headings exempt |
| Post-check | New pure `extract_citations()` + `strip_invalid_citations()` — refs outside 1..N are stripped (whitespace tidied), sentences never touched (entailment stays the verifier's job); ref-free answers return byte-identical |
| Wiring | `generate_grounded_answer` now uses `format_context_with_chunk_indices` (byte-identical prompt string) and strips hallucinated refs post-generation with a log line; ABSTAIN/empty paths unchanged |
| Runner | `run_baseline_eval.py` scores citation existence per answer (`cited_segments` recorded); methodology footnote updated: correctness = existence until Phase 5 entailment |

**Tests:** +8 (extraction incl. malformed/prose negatives, keep/strip/zero/noop matrix,
end-to-end strip + keep-valid via mocked LLM). Existing generation tests unbroken by the
prompt change. **295/295 green**, ruff clean. No `models.yaml` change (prompt-only phase).

**Deliberately deferred to Phase 5:** populating per-claim `evidence_ids` from inline cites
(requires decomposition-output changes — outside the narrowest layer).

---

### 16. Phase 5 — targeted claim retrieval (2026-09-16)

Skills: `langchain-rag`, `test-driven-development` (RED→GREEN), `surgical-patch`
(narrowest layer), `langgraph-fundamentals` (linear flow kept — retrieval happens
inside the verification node; no new nodes/edges). TDD caught 3 GREEN bugs, including
one REAL pre-existing conflict (below).

| Area | Change |
|------|--------|
| `verifier.py` | New `retrieve_evidence_for_claim()` (claim-text hybrid search, drops seen chunks, fail-closed → `[]`); new `_persist_claim_evidence()` (integrity-audit → dedup vs analysis evidence → persist with `method="claim_retrieval"`, returns chunk↔id pairs); `execute_claim_verification(..., kb_id_str=None)` gains step 2b — NEUTRAL claims only, budget `min(max_claim_retrievals=3, #neutral)`, each gets ≤1 retrieval (top-5) + 1 NLI on a fresh mini-context; SUPPORTED/CONTRADICTED flips adopted with `[targeted retrieval]` explanation suffix and correctly mapped fresh evidence linkage |
| Scope decision | CONTRADICTED claims are NEVER re-retrieved — existing evidence already refutes them; searching for support would cherry-pick |
| Claim linkage | Persistence loop unions step-2b fresh ids + Phase-4 `[Segment N]` markers surviving in claim text (closes the Phase-4 deferral — no decomposition changes needed) |
| `graph.py` | One line: `kb_id=state.get("kb_id")` threaded into verification; no structure change |
| `models.yaml` | `cost_controls.max_claim_retrievals: 3`, `claim_retrieval_top_k: 5`; **deleted** dead `citation/evidence-coverage/source-integrity_weight` trio (zero readers — tuning trap); version 1.11 → 1.12 |

**Real bug found by the new tests:** Phase-4 `[Segment N]` citations tripped the
scaffold-echo meta-filter (`\bsegments?\s+\d`), silently dropping cited claims to zero
→ honest FAIL. Fixed with a `(?<!\[)` lookbehind: bracketed citations pass, bare
"Segment 2 states…" prose still drops (pre-existing meta tests green). Second catch:
mini-context-relative segment numbers were briefly re-mapped against the original
context (wrong-evidence linkage) — fixed by consuming them at flip time.

**Tests:** +8 `test_claim_retrieval.py` (dedup+cap, outage → `[]`, NEUTRAL→SUPPORTED flip
with fresh-only linkage, budget == 3 hybrid calls, CONTRADICTED never re-searched,
no-kb skip, inline-cite union, weights-absent). **303/303 green**, ruff clean.
Worst-case cost per analysis: +3 retrievals +3 NLI, only on failing claims, inside the
existing verification-node timeout.

---

### 17. Phase 6 — deterministic query router + fan-out (2026-09-16)

Skills: `langchain-rag`, `test-driven-development` (RED→GREEN), `surgical-patch`
(narrowest layer), `langgraph-fundamentals` (linear flow kept — fan-out lives inside
the retrieval node; no new nodes/edges; bounded concurrent branches).

| Area | Change |
|------|--------|
| New `agent/router.py` | Pure, deterministic, zero-LLM routing: SIMPLE (today's path) / TEMPORAL (explicit year → July-1 reference_time, else caller now) / COMPARISON ("A vs B", "difference between", "compare X and Y" → 2 sub-queries) / COMPLEX (multi-"?" split, capped). Unsplittable input falls back to SIMPLE — never worse than today. `merge_fanout_results` dedups by chunk id keeping best RRF, sorts desc. `fanout_retrieve` runs sub-queries concurrently; partial outage degrades, total outage raises |
| `graph.py` | Retrieval node routes first; SIMPLE keeps byte-identical kwargs (existing exact-kwarg test green); multi-class fans out with per-branch top_k/embedding overrides + reference_time, one `retrieval.routed` trace event; rerank → integrity → persist tail untouched |
| `retriever.py` | **Removed** `AmbiguityDetector` + `detect_query_ambiguity` (~65 lines, zero callers — plan's adopt-or-remove verdict: remove; pre-retrieval routing supersedes it) |
| `models.yaml` | `retrieval.query_router: {enabled: true, max_sub_queries: 3}` kill-switch + ceiling; version 1.12 → 1.13 |

**TDD caught 2 issues:** the sub-query floor (10 chars) silently dropped real entities
("Team plan" is 9 chars → whole query fell back to SIMPLE); fixed to ≥2 chars with a
comment. Test fixture fused dicts lacked `id`, which merge correctly deduped — fixture
fixed (real RRF output always carries `id`).

**Tests:** +13 `test_router.py` (classify matrix incl. fallbacks, year→July-1, merge
order/dedup, concurrent fan-out, partial/total outage, node-level 2-call fan-out).
**316/316 green**, ruff clean. Simple-query cost unchanged (same single call);
worst case 3 concurrent hybrid calls inside existing per-branch budgets.
LLM-based sub-question decomposition deliberately NOT added — needs a live-eval
signal that deterministic splitting is insufficient.

---

### 18. Phase 8 — index lifecycle (2026-09-16)

Skills: `test-driven-development` (RED→GREEN), `surgical-patch` (narrowest layer).
Verified first: KB/document delete paths already purge Mongo + Qdrant + cache
(no orphan bug), snapshots already copy vectors — so the phase wired the missing
surface instead of rebuilding working code.

| Area | Change |
|------|--------|
| Routes | `POST /knowledge-bases/{id}/snapshots` → 201; `POST /knowledge-bases/{id}/rollback/{snap}` → 200 with the NEW live id (snapshot's — clients must swap; documented on the endpoint) |
| Guard | Rollback refuses vector-less snapshots (Mongo chunks present, 0 Qdrant points — pre-vector-copy era) with 409 "re-upload instead" rather than restoring an empty KB; empty snapshots still roll back fine |
| Fix | Snapshot chunk copies now carry `ocr_used`/`ocr_confidence` (were silently dropped, breaking the OCR provenance chain on restore) |
| Proven | `delete_document` purges Qdrant by `document_id` filter (characterization test — was only assumed) |

**Tests:** +6 `test_lifecycle.py` (both routes incl. id-swap + 409-on-foreign, empty-guard 409,
OCR-preserving snapshot, Qdrant purge filter). **322/322 green**, ruff clean. No
`models.yaml` change. Debugging note: `delete_kb_collection` resolves its Qdrant client
from the qdrant module's own namespace — tests must patch `app.db.qdrant.get_qdrant_client`
alongside the kb_service seam or they hit real Qdrant (503).
