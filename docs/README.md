# TRUSTRAG Documentation Index

Welcome to the technical documentation for the TRUSTRAG AI Reliability Workbench. This directory is organized by domain: architecture, security, quality audits, evaluation methodology, and deployment.

---

## 📁 Documentation Structure

```
docs/
├── README.md                  # Master index (this file)
├── TRUSTRAG_specs.md          # Full product specification
├── ROADMAP.md                 # Product vision, milestones, phase tracking
├── IMPLEMENTATION_STATUS_2026-09-11.md  # Current implementation status & progress
├── architecture/
│   ├── architecture.md        # End-to-end system design, MCP tools, LangGraph loop, data flow
│   └── decision-log.md        # ADRs D-01 through D-30 (BM25+IDF, reranker cap, OCR, chunking, citations, claim retrieval, router, lifecycle)
├── audits/
│   └── 2026-09-11_unified_senior_audit.md  # Canonical audit (supersedes all prior audits; history in git)
├── security/
│   ├── security-controls.md   # Auth, anti-IDOR, SSRF, rate limits, headers
│   └── threat-model.md        # STRIDE threat model and mitigations
├── evaluation/
│   └── methodology.md         # Benchmark dataset and reliability metrics
├── deployment/
│   ├── DEPLOYMENT_GUIDE.md    # Cloudflare Pages + Google Cloud Run + MongoDB Atlas
│   └── README.md              # Docker Compose / local deployment runbook
```

---

## 📑 Core Sections

### 1. Architecture & Design
- [**System Architecture (`architecture/architecture.md`)**](architecture/architecture.md): LangGraph agent loop, MCP tools, hybrid retrieval (dense + sparse RRF), claim decomposition → NLI verification → verdict pipeline.
- [**Decision Log (`architecture/decision-log.md`)**](architecture/decision-log.md): Architectural Decision Records covering technology choices, storage layers, and code quality decisions (D-21 ONNX embeddings, D-22 tolerant NLI parsing, D-23–D-30: BM25+IDF, reranker cap, RapidOCR, chunking, citations, claim retrieval, router, lifecycle).
- [**Implementation Plan (`TRUSTRAG-IMPLEMENTATION-PLAN.md` + `TRUSTRAG-UPGRADE-PLAN.md`)](TRUSTRAG-IMPLEMENTATION-PLAN.md): phase-by-phase upgrade plan verified against code, with per-phase status (phases 0–8 done; recovery/security/prod/eval remaining).

### 2. Audits & Implementation Status
- [**2026-09-11 — Unified Senior Audit**](audits/2026-09-11_unified_senior_audit.md): **Canonical audit** — multi-role review (frontend, backend, AI/ML, security, optimization, testing) with Apple-design compliance, ultra-low RAM plan, and 2-day vs >2-day upgrade split. Supersedes all prior audits (history preserved in git).
- [**Implementation Status (`IMPLEMENTATION_STATUS_2026-09-11.md`)**](IMPLEMENTATION_STATUS_2026-09-11.md): What was fixed, current test state (backend 322/322, frontend 22/22), and remaining work. Covers the ≤2-day fixes, ONNX runtime, tolerant-NLI + fusion hardening, CI repairs, offline-warning and probe hardening, the push-readiness passes, and RAG quality phases 0–8 (§13–§18).

### 3. Security
- [**Security Controls (`security/security-controls.md`)**](security/security-controls.md): JWT, bcrypt, anti-IDOR, SSRF guards, rate limiting, defensive headers.
- [**Threat Model (`security/threat-model.md`)**](security/threat-model.md): STRIDE analysis and countermeasures.

### 4. Evaluation
- [**Methodology (`evaluation/methodology.md`)**](evaluation/methodology.md): datasets and reliability metrics.

### 5. Deployment
- [**Production Deployment (`deployment/DEPLOYMENT_GUIDE.md`)**](deployment/DEPLOYMENT_GUIDE.md): Cloudflare Pages, GCR, MongoDB Atlas, Qdrant Cloud.
- [**Local/Compose Deployment (`deployment/README.md`)**](deployment/README.md): Docker Compose and local run instructions.

---

## Current Stack (2026-09-16, `models.yaml` v1.13)

- **LLM**: llama.cpp (local, default: `LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M`), Ollama (default: `gemma3:1b`), Gemini, NVIDIA NIM — selectable per request.
- **Embeddings**: BGE-small-en-v1.5 (local, 384d) — HuggingFace/torch by default, **ONNX Runtime** (`EMBEDDING_PROVIDER=onnx`, export via `scripts/export_bge_onnx.py`) for torch-free ultra-low RAM.
- **Retrieval**: Qdrant (embedded local or cloud) + client BM25-TF sparse with server-side IDF (`Modifier.IDF`) + RRF with enforced `fusion_top_k`; deterministic query router (simple/temporal/comparison/complex, ≤3 fan-out); reranker off by default (`top_k: 20` depth cap; needs the `local-models` extra, absent from Docker).
- **Ingestion**: newline-preserving normalization; `chunking_strategy` (`sliding_window` default); RapidOCR-ONNX per-page fallback (default on; models download to `~/.onnx` on first scanned page — pre-warm to avoid first-upload stall).
- **Verification**: Batch NLI + per-claim fallback with tolerant parsing of small-model near-miss JSON (VERIFIED→SUPPORTED aliasing, segment coercion); inline `[Segment N]` citations with invalid-ref strip post-check; targeted NEUTRAL-only claim retrieval (≤3/analysis).
- **Lifecycle**: KB snapshots + rollback routes (rollback returns a NEW live id; 409 on vector-less snapshots); delete paths purge Mongo + Qdrant.
- **Local LLM server**: start via `./scripts/start_local_llm.sh` (auto GPU offload + KV budget).
- **Tests**: backend 322/322, frontend 22/22 + lint + build; ruff check + format clean.
- **Pending operator runs**: live baseline + ablations (`scripts/run_baseline_eval.py`); pre-IDF KBs need document re-upload; chunking/normalization change needs re-index; OCR models not pre-warmed.
