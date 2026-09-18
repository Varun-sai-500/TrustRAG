# TRUSTRAG — Project Roadmap & Remaining Steps

> Last updated: 2026-09-16  
> Current status: All 12 core phases complete + Post-Launch Quality & Audit Cycle complete (v1 → v4) + ultra-low RAM + ONNX + claims-hardening + fusion passes + RAG quality phases 0–8 (eval harness, BM25+IDF, reranker cap, OCR, chunking, citations, claim retrieval, router, lifecycle). **322 backend tests, 22 frontend Vitest, 2 Playwright E2E, k6 load smoke — 100% pass rate.** Active and upcoming work below.

---

## ✅ Completed Phases (1–12 + Post-Launch Refinements)

| Phase | Description | Key Deliverables & Verification | Status |
| :--- | :--- | :--- | :--- |
| **0** | Architecture, ADRs, Threat Model | System boundaries, LangGraph loop specs, STRIDE analysis (`docs/architecture/`, `docs/security/`) | ✅ COMPLETE |
| **1** | Monorepo, Config, Docker, CI/CD | Docker Compose, non-root containers, GitHub Actions CI/CD workflows (`.github/workflows/`) | ✅ COMPLETE |
| **2** | React 18 Workbench Frontend | Glassmorphism design system, 11 interactive pages, Lucide icons (`apps/web/src/`) | ✅ COMPLETE |
| **3** | FastAPI CRUD Routes & Data Layer | MongoDB storage, asynchronous endpoints, analysis lifecycle (`apps/api/app/api/`) | ✅ COMPLETE |
| **4** | Auth & Multi-Tenant Security | JWT authentication, bcrypt passwords, anti-IDOR checks, rate limiting (`apps/api/app/core/security.py`) | ✅ COMPLETE |
| **5** | Document Ingestion (8 Formats) | Native parsers for PDF, DOCX (defusedxml), CSV, JSON, HTML, HTM, TXT, MD (`apps/api/app/ingestion/`) | ✅ COMPLETE |
| **6** | Hybrid Retrieval & Reciprocal Rank Fusion | BGE-small-en-v1.5 (384d, local) + client BM25-TF saturation + server-side Qdrant IDF + RRF ($k=60$) with enforced `fusion_top_k`; deterministic query router (simple/temporal/comparison/complex) + bounded fan-out (`apps/api/app/retrieval/`, `apps/api/app/agent/router.py`) | ✅ COMPLETE |
| **7** | Atomic Claim Decomposition & NLI Verifier | Structured extraction, Open Knowledge triples `(S, P, O)`, Gemini NLI verdict (`apps/api/app/verification/`) | ✅ COMPLETE |
| **8** | Evidence Integrity & Provenance | Cryptographic SHA-256 hash auditing, temporal validity filters (`apps/api/app/verification/integrity.py`) | ✅ COMPLETE |
| **9** | LangGraph Adaptive Self-Healing Loop | StateGraph recovery loop, dynamic rewrites, context expansions (`apps/api/app/agent/graph.py`) | ✅ COMPLETE |
| **10** | Live Execution Trace Streaming | Server-Sent Events (SSE) trace streaming, MongoDB persistence (`apps/api/app/services/analysis_service.py`) | ✅ COMPLETE |
| **11** | Evaluation & Experimentation | Custom reliability metrics, benchmark dataset, ablation engine (`apps/api/app/evaluation/`) | ✅ COMPLETE |
| **12** | Production Hardening & Security SAST | Bandit SAST 0 issues, memory streaming upload guards, defensive HTTP headers (`apps/api/app/main.py`) | ✅ COMPLETE |
| **P-L** | **Multi-Tenant User Data Isolation** | Scoped `user_id` on all records, compound indexes (`claim_user_time`, `evidence_user_time`), cascade deletions | ✅ COMPLETE |
| **P-L** | **Open Knowledge & JSON-LD Export** | `(Subject, Predicate, Object)` claim triples, schema.org JSON-LD compliance export (`GET /analyses/{id}/export`) | ✅ COMPLETE |
| **P-L** | **Observability & Health Telemetry** | Full health monitoring (`mongodb` + `qdrant`), interactive diagnostics panel in `SettingsPage.jsx` | ✅ COMPLETE |
| **P-L** | **Master Quality Audit (52 Findings)** | 52 findings resolved across FE/BE/DB/SEC/PERF, documented in `docs/audits/2026-09-11_unified_senior_audit.md` (canonical; history in git) | ✅ COMPLETE |
| **13** | **SOTA UI Overhaul & Low-RAM Architecture** | Ultra-premium landing page, 75% Qdrant RAM compression (INT8 on-disk), 0 MB GPU RAM via Gemini 384d MRL, LRU embedding cache, and universal Model Context Protocol (MCP) server | ✅ COMPLETE |
| **14** | **Master SOTA Multi-Role Production Audit** | Deep Systems, Security, AI/ML, and QA audit suite (79/79 pytest, 0 lint warnings) documented in `docs/audits/` | ✅ COMPLETE |
| **15** | **Ultra-Low RAM + Security Hardening (2026-09-11)** | psutil RSS guard, batched SQLite writes, bounded LLM registry (4 max), X-Request-ID validation, production CORS lock, split health endpoints, magic-bytes + zip-bomb upload defense, `MALLOC_ARENA_MAX`, Apple-design reduced-motion | ✅ COMPLETE |
| **16** | **ONNX BGE Runtime — Torch-Free Embeddings (2026-09-11)** | `scripts/export_bge_onnx.py` + ONNX Runtime wrapper (`EMBEDDING_PROVIDER=onnx`), numerical parity verified, ~500–1000 MB RSS savings | ✅ COMPLETE |
| **17** | **Claims Verification Hardening (2026-09-12)** | Tolerant NLI parsing (VERIFIED→SUPPORTED aliasing, segment coercion, batch bare-int drop) fixing 0/x-supported on good answers; 6 new regression tests; 38-error lint sweep | ✅ COMPLETE |
| **18** | **Fused Decompose+Verify + CI Repairs (2026-09-12)** | Single-call fused NLI path with two-step fallback (live-evaled 2.0s vs 3.4s); fixed frontend-build (missing install), Docker context + empty-venv boot bug, stale k6 health contract; onnxruntime shipped in image; Bandit B615 revision pin; Trivy SARIF advisory | ✅ COMPLETE |
| **19** | **Offline-Warning + Probe Hardening (2026-09-12)** | `/models/providers` degrades instead of 500ing; UI warns on failed providers query too; probe retries once and splits refused (down) vs timeout (slow); concurrent provider checks; hermetic verification suite | ✅ COMPLETE |
| **20** | **RAG Quality Phases 0–8 (2026-09-16)** | Frozen 25-query baseline + metrics harness + live runner (`tests/eval/`, `scripts/run_baseline_eval.py`); BM25-TF + server IDF with collection migration; reranker depth cap, stays default-off; RapidOCR-ONNX per-page fallback with provenance; newline-preserving normalization + wired chunking strategies; inline `[Segment N]` citations + strip post-check; NEUTRAL-only targeted claim retrieval; deterministic router + fan-out; snapshot/rollback routes + empty-snapshot guard — 322 backend tests | ✅ COMPLETE |

---

## 🔴 IMMEDIATE (Pre-Deployment Checklist)

- [ ] **Set real values in `.env`** — copy `.env.example` → `.env` and fill in:
  - `JWT_SECRET` → `python -c "import secrets; print(secrets.token_hex(64))"`
  - `GEMINI_API_KEY` → Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey)
  - `MONGODB_URI` → Local MongoDB (`mongodb://localhost:27017`) or MongoDB Atlas URI
- [ ] **Rotate MongoDB Atlas credentials** if initial credentials were ever exposed or shared.

---

## 🟠 HIGH PRIORITY (Next Production Sprint)

### Deployment & Cloud Hosting
- [ ] **Backend Cloud Deployment** — Deploy FastAPI container to:
  - [Render.com](https://render.com) or [Railway.app](https://railway.app)
  - [Fly.io](https://fly.io) (`flyctl deploy`)
  - Self-hosted Docker instance via Coolify
- [ ] **Frontend CDN Deployment** — Deploy React Vite bundle to [Vercel](https://vercel.com) or [Netlify](https://netlify.com); configure `VITE_API_URL`.
- [ ] **Production Environment Flag** — Set `APP_ENV=production` (disables public Swagger UI, enforces API keys).
- [ ] **Managed Qdrant Cloud** — Provision cluster at https://cloud.qdrant.io; set `QDRANT_URL` and `QDRANT_API_KEY`.

### Testing & Verification Expansion
- [x] **End-to-End Browser Automation** — Playwright suite in `apps/web/e2e/auth.spec.js`: route guard, register, UI login, dashboard, **JWT revocation after logout**; CI job with live Mongo + API + Chromium
- [x] **Live Integration Tests** — E2E suite runs against live MongoDB and the real FastAPI app (no mocks); k6 smoke also hits live Mongo/Qdrant reads
- [x] **Rate Limiter Threshold Tests** — `apps/api/tests/test_rate_limit.py` asserts 429 when the per-minute auth ceiling is exceeded
- [x] **Load Testing in CI** — k6 smoke (`load-test/smoke.js`) runs in the `e2e` CI job: health + authenticated reads, ~3.3k requests/35s, p95<300ms gate

---

## 🟡 MEDIUM PRIORITY (Enhancements & Scalability)

### Architecture & Worker Queues
- [ ] **Background Task Worker** — Migrate from FastAPI `BackgroundTasks` to Celery / Redis or ARQ for persistent background job processing.
- [ ] **Streaming LLM Token Delivery** — Stream Gemini generation tokens via SSE directly to the Playground UI.
- [ ] **Token Budget Enforcement** — Count tokens client-side / pre-request using `tiktoken` or Gemini token counter before submission.
- [ ] **Named Qdrant Vectors** — Migrate collection schema to explicit `NamedVectorParams` for multi-vector scenarios.

### Observability & Monitoring
- [ ] **Structured Log Shipping** — Ship structlog JSON logs to Datadog, Grafana Loki, or Papertrail.
- [ ] **Application APM** — Integrate Sentry for exception tracking and performance tracing.
- [ ] **Prometheus Metrics** — Expose `/metrics` endpoint measuring retrieval latency, NLI verification duration, and claim support ratios.
- [ ] **Gemini Cost Accounting** — Track and aggregate token expenditures per analysis run and experiment.

### Advanced Security
- [ ] **Refresh Token Rotation** — Implement 15-minute access tokens with 7-day rolling refresh tokens.
- [ ] **Account Lockout Policy** — Temporarily lock accounts after 5 consecutive failed login attempts.
- [ ] **File Content Antivirus Scanning** — Integrate ClamAV scanning on uploaded document streams.

---

## 🟢 LOW PRIORITY (Future Innovation)

### RAG Quality & Advanced AI
- [ ] **Live baseline + ablations** — run `scripts/run_baseline_eval.py` against a live stack; fill `docs/evaluation/methodology.md` snapshot table (no fabricated rows); calibrate reranker thresholds from Hybrid-vs-Hybrid+Rerank.
- [ ] **KB re-upload window** — pre-IDF collections recreate on next init; newline change needs re-index; combine into one operator re-upload.
- [ ] **Query Expansion** — Implement Hypothetical Document Embeddings (HyDE) or multi-query expansion prior to retrieval.
- [ ] **Dense Embedding Upgrades** — BGE-small-en-v1.5 is the current default (MiniLM legacy/retired); benchmark larger models only with a measured recall gap.
- [ ] **Multilingual Support** — Incorporate language detection and multilingual embedding models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`).
- [ ] **Per-KB Configuration Profiles** — `ingestion.chunking_strategy` is currently global (`sliding_window` default); allow per-KB chunk sizes/overlap/strategy.

### Collaboration & Enterprise Features
- [ ] **Team & Organization Sharing** — Role-based access control (RBAC) enabling collaborative knowledge base access.
- [ ] **Automated Continuous Verification** — Scheduled background re-verification runs triggered when documents are updated.
- [ ] **Webhook Notifications** — Dispatch alerts on low-confidence analysis completions or detected contradictions.
- [ ] **Interactive Annotated PDF Export** — Export PDF reports featuring highlighted evidence spans and verification callouts.

---

## 📋 Quality Audit Sign-Off History

| Cycle | Date | Scope & Key Findings | Status |
| :--- | :--- | :--- | :--- |
| **Audit v1** | 2026-08-28 | Initial codebase review, linting, and imports | ✅ Resolved |
| **Audit v2** | 2026-08-29 | Static type checks, defensive headers, MongoDB index performance | ✅ Resolved |
| **Audit v3** | 2026-08-29 | Global route aggregation, frontend page connections, LLM model upgrade | ✅ Resolved |
| **Audit v4 (Master)** | 2026-08-29 | Multi-tenant isolation, cascade deletion, 8 document formats, XXE defense, live telemetry, 79 automated tests | ✅ **100% VERIFIED** |
| **Audit v5 (SOTA)** | 2026-08-31 | Port 8080 default, Model Context Protocol (MCP) live grounding, LangGraph self-healing loop fixes, dual-channel SSE fallback polling, GFM tables via remark-gfm, 86 automated tests | ✅ **100% VERIFIED** |
| **RAG quality (phases 0–8)** | 2026-09-16 | Eval harness, BM25+IDF, reranker cap, OCR, chunking, citations, claim retrieval, router, lifecycle; 322 backend tests | ✅ **100% VERIFIED** |

> All historical audit passes have been verified and consolidated into the audit files under [`docs/audits/`](docs/audits/).
