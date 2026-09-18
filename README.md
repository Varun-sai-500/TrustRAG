# TrustRAG

> **Your local-first RAG reliability workbench** — catch hallucinations, audit evidence, and self-heal low-confidence answers using an adaptive LangGraph loop. Everything runs on your machine. No API keys required.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Offline-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF_Server-orange)](https://github.com/ggerganov/llama.cpp)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Embeddings-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Tests](https://img.shields.io/badge/Backend%20Tests-326%20Passing-brightgreen)](apps/api/tests)
[![Tests](https://img.shields.io/badge/Frontend%20Tests-22%20Passing-brightgreen)](apps/web)
[![E2E](https://img.shields.io/badge/Playwright%20E2E-2%20Passing-brightgreen)](apps/web/e2e)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## What is this?

Standard RAG systems fail silently. They grab some context, generate an answer, and present it as fact — even when the answer is wrong. There's no audit trail, no verification, no way to know if you can trust the output.

TrustRAG fixes that. It's a full reliability pipeline that:

1. **Decomposes** responses into individual factual claims — and verifies each one against your documents in the same step (fused NLI, so small local models answer in one call instead of two).
2. **Validates** every claim against retrieved evidence, tolerating the quirky JSON small models emit, and falling back gracefully instead of failing silently.
3. **Audits** source integrity with SHA-256 hashes and temporal validity windows.
4. **Self-heals** when confidence is low — rewriting queries and expanding search via a LangGraph state machine, then either returning a grounded answer or safely abstaining.

Think of it as a fact-checking layer for RAG. It runs 100% locally on your machine with Ollama or llama.cpp — your documents never leave your laptop, there are no per-query bills, and no API keys are needed unless you want cloud models for the heavy lifting.

---

## Quick Links

| Service | URL | What it does |
|---|---|---|
| **Frontend Workbench** | [http://localhost:5173](http://localhost:5173) | The React UI — upload docs, ask questions, see verification results |
| **Backend API** | [http://localhost:8000](http://localhost:8000) | FastAPI engine — all the RAG, NLI, and LangGraph magic |
| **Interactive Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI — test every endpoint right in your browser |
| **Health Check** | [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health) | Public status (detailed view at `/health/detailed` with auth) |

---

## Table of Contents

- [What is this?](#what-is-this)
- [Quick Links](#quick-links)
- [How the self-healing loop works](#how-the-self-healing-loop-works)
- [Pipeline stages in detail](#pipeline-stages-in-detail)
- [Reliability, verdicts & recovery](#reliability-verdicts--recovery)
- [Getting Started](#getting-started)
  - [What you need](#what-you-need)
  - [Step 1 — Install platform tools](#step-1--install-platform-tools)
  - [Step 2 — Clone and configure](#step-2--clone-and-configure)
  - [Step 3 — Start services](#step-3--start-services)
  - [Step 4 — Open the UI](#step-4--open-the-ui)
- [Try it from the command line](#try-it-from-the-command-line)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Configuration reference](#configuration-reference)
- [Technology stack](#technology-stack)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Frontend pages](#frontend-pages)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [License](#license)

---

## How the self-healing loop works

```
                        Your Question
                             │
                             ▼
               ┌───────────────────────────┐
               │ 1. Text Normalization     │  clean up noise, fix hyphens, strip fluff
               │    & Document Zoning      │  weight titles/headers higher than body
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ 2. Hybrid Retrieval       │  Dense vectors (BGE, 384d) + BM25 keywords
               │    + Reciprocal Fusion    │  combined with RRF scoring
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ 3. Grounded Generation    │  LLM answer, strictly conditioned on evidence
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ 4. Claim Decomposition    │  break answer into atomic facts
               │    + Batch NLI Verify     │  SUPPORTED | CONTRADICTED | NEUTRAL
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ 5. SHA-256 Hash Audit     │  tamper detection on source chunks
               │    + Reliability Scoring  │  coverage vs contradiction thresholds
               └─────────────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     [Meets Thresholds]            [Below Threshold]
              │                             │
              ▼                             ▼
    ┌───────────────────┐        ┌────────────────────────────┐
    │  Grounded Answer  │        │ 6. Adaptive Recovery Loop  │
    │  + Evidence Cards │        │    rewrite query, expand   │
    │  + Citations      │        │    search, retry (1 round) │
    └───────────────────┘        └────────────┬───────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              [Recovered]      [Recovery Exhausted]
                                    │                   │
                                    ▼                   ▼
                          ┌───────────────┐   ┌───────────────┐
                          │  Grounded     │   │  Safe         │
                          │  Answer       │   │  ABSTAIN      │
                          └───────────────┘   └───────────────┘
```

When the system isn't confident in its answer, it doesn't guess. It either heals itself or tells you it doesn't know. That's the point.

---

## Pipeline stages in detail

Each stage below names the code that runs it and the `config/models.yaml` knob that tunes it. Env vars always win over YAML.

| # | Stage | What happens | Key knobs |
|---|---|---|---|
| 1 | **Normalize & zone** | Noise cleanup, hyphen repair, filler stripping (paragraph breaks preserved for section/table detection); text split into ~512-char chunks (64-char overlap, word-boundary snapped) with zone tags — titles/headers score higher than body. Chunking strategy selectable (`sliding_window` default, `semantic`, `progressive`, `layout_aware`). Scanned/image PDF pages fall back to **local RapidOCR-ONNX** — on by default, per-page under 50 native chars, 300 dpi render, sub-0.5-confidence text dropped — with `ocr_used`/`ocr_confidence` provenance on every chunk | `ingestion.chunk_size: 512`, `chunk_overlap: 64`, `chunking_strategy`, `ingestion.ocr.enabled/min_native_chars/dpi/min_confidence` |
| 2 | **Hybrid retrieval** | A deterministic router classifies each query first — simple (one hybrid call), temporal (explicit year → reference time), comparison (`A vs B` → two parallel retrievals), complex (multi-question split, capped). Dense vectors (`BAAI/bge-small-en-v1.5`, 384d — HuggingFace/torch or torch-free ONNX Runtime via `EMBEDDING_PROVIDER=onnx`) + BM25 sparse vectors (client TF-saturation, server-side IDF via Qdrant `Modifier.IDF`) fused with Reciprocal Rank Fusion (capped at `fusion_top_k`); embedding model is **pinned per KB at ingest** | `retrieval.dense_top_k: 20`, `sparse_top_k: 20`, `rrf_k: 60`, `fusion_top_k: 20`, `sparse_k1/b`, `query_router.max_sub_queries: 3` |
| 3 | **Rerank (optional)** | Cross-encoder rescoring of fused candidates (≤20 scored via `reranker.top_k` depth cap, ≤8 to context, adaptive top-4 on confident heads); **off by default** — enable only where `sentence-transformers` is installed (`local-models` extra; absent from the torch-free Docker runtime, where enabling is a silent no-op) | `reranker.enabled: false`, `model: cross-encoder/ms-marco-MiniLM-L-6-v2`, `top_k: 20` |
| 4 | **Integrity audit** | SHA-256 tamper check per chunk + temporal validity windows (`effective_from`/`effective_until`); corrupted segments are excluded before generation | — (always on) |
| 5 | **Grounded generation** | Answer strictly conditioned on ≤8 surviving chunks within a 3000-char context budget (fits small-model windows); every factual sentence carries inline `[Segment N]` citations, and refs to unserved segments are stripped post-generation; empty/insufficient context → `ABSTAIN`, never a guess | `retrieval.max_context_chunks: 8`, `llm.temperature: 0.2` |
| 6 | **Claim decomposition + NLI** | Answer split into ≤8 atomic, self-contained claims; each judged `SUPPORTED` / `CONTRADICTED` / `NEUTRAL` against the evidence in one batch call, with full per-claim fallback if the batch fails. NEUTRAL claims (missing evidence — never CONTRADICTED) get one bounded targeted-retrieval round each (claim text as query, top-5, ≤3/analysis) | `cost_controls.max_verification_claims: 8`, `max_individual_nli_fallback: 8`, `max_claim_retrievals: 3`, `claim_retrieval_top_k: 5`, `verification.temperature: 0.0` |
| 7 | **Verdict & recovery** | Coverage/contradiction scored against thresholds (see below); on FAIL one recovery round runs, then either a grounded answer or safe `ABSTAIN` | `reliability.*`, `recovery.max_recovery_attempts: 1` |

> **Single-document note:** with one short document, any query retrieves roughly the same chunks. If verification still fails 0/8, suspect the NLI judge or truncated context — not retrieval. Check the Claims tab explanations and the analysis trace.

---

## Reliability, verdicts & recovery

### Thresholds (`reliability` in `models.yaml`)

| Threshold | Default | Meaning |
|---|---|---|
| `minimum_evidence_coverage` | `0.80` | ≥80% of claims must be SUPPORTED (7 of 8) |
| `maximum_contradiction_rate` | `0.20` | ≤20% of claims may be CONTRADICTED |
| `abstain_below` | `0.50` | Scores below this → abstain instead of answering |

Reliability score = `coverage × (1 − contradiction_rate)`. These are engineering defaults, not calibrated probabilities.

### Verdicts

| Status | Meaning |
|---|---|
| `TRUSTED` | Passed coverage and contradiction thresholds |
| `UNCERTAIN` | Failed thresholds but score ≥ `abstain_below` — shown with warnings |
| `FAILED` | Failed thresholds and score < `abstain_below` |
| `ABSTAINED` | Model explicitly abstained — correct behavior on zero evidence, not an error |

Failure diagnoses: `RETRIEVAL_FAILURE` (no usable evidence), `RETRIEVAL_OUTAGE` (search infra down — distinct from "no evidence"), `EVIDENCE_CONFLICT` (too many contradictions), `LOW_COVERAGE` (too few supported claims).

### Recovery strategies (one round, in priority order)

| Strategy | What it does | When it wins |
|---|---|---|
| `query_rewrite` | LLM expands acronyms/synonyms targeting the unverified claims (5–12 words) | Missing-fact failures |
| `re_retrieve` | Doubles search width (`top_k`, context) for thin evidence | Genuinely thin evidence |
| `regenerate` | Retries generation on saved chunks with zero retrieval spend | Evidence already sufficient (auto-downgraded from `re_retrieve`) |

Configure via `recovery.strategy_priority`. The rewrite is sanitized — instruction echoes collapse back to the original query instead of polluting retrieval.

---

## Getting Started

Plan on about 15 minutes end to end: install the platform tools once, configure one file, start three terminals, and you'll be asking questions of your own documents. If anything misbehaves, `./scripts/setup.sh` diagnoses your machine and tells you the exact fix.

### What you need

| Tool | Version | Why |
|---|---|---|
| **Python** | 3.11+ | Backend runtime |
| **Node.js** | 22+ | Frontend build tools (see `engines` in `apps/web/package.json`) |
| **MongoDB** | 7.0+ | Document & metadata storage |
| **Ollama** or **llama.cpp** | Latest | Local LLM inference (zero API keys) |
| **Git** | Any recent | Clone the repo |

Optional (only if you want cloud features):
- Google Gemini API key — for cloud LLM reasoning
- NVIDIA NIM API key — for enterprise NIM models
- Tavily API key — for AI-powered web search (DuckDuckGo is free and works without a key)

**Optional (ultra-low RAM embeddings):**
- ONNX Runtime BGE-small — set `EMBEDDING_PROVIDER=onnx` in `.env` (no PyTorch in API process, ~500-1000 MB RSS savings). Requires one-time export: `python scripts/export_bge_onnx.py` and model at `apps/api/.model_cache/bge-small-en-v1.5.onnx`.

---

### Step 1 — Install platform tools

Pick your operating system and run the commands. This installs everything TrustRAG needs.

<details>
<summary><b>macOS (Apple Silicon or Intel)</b></summary>

```bash
# 1. Xcode command line tools (if not already installed)
xcode-select --install

# 2. Homebrew (the macOS package manager)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Core tools
brew install git python@3.11 node mongodb-community ollama

# 4. Start MongoDB (runs in background on boot)
brew services start mongodb-community

# 5. Start Ollama (runs in background)
brew services start ollama

# 6. Pull the default local LLM (~815MB)
ollama pull gemma3:1b

# 7. (Optional) llama.cpp — for GGUF models
#    Option A: Install via Homebrew
brew install llama.cpp

#    Option B: Build from source for latest features
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_NATIVE=on
cmake --build build -j$(sysctl -n hw.ncpu)
# The binary is at build/bin/llama-server — add to PATH or use the full path
```

</details>

<details>
<summary><b>Linux (Ubuntu / Debian)</b></summary>

```bash
# 1. System packages
sudo apt update && sudo apt install -y \
  git python3.11 python3.11-venv python3-pip \
  build-essential cmake curl jq

# 2. Node.js 22+ (via NodeSource — the frontend requires Node >= 22)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version    # must print v22.x or newer

# 3. MongoDB 7.0
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
  sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
  https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
sudo systemctl enable --now mongod

# 4. Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Start the daemon in background
ollama serve &
sleep 2

# 5. Pull the default local LLM (~815MB)
ollama pull gemma3:1b

# 6. (Optional) llama.cpp — build from source
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_NATIVE=on
cmake --build build -j$(nproc)
# Binary at build/bin/llama-server — add to PATH or use the full path
```

</details>

<details>
<summary><b>Windows 11 (PowerShell + winget)</b></summary>

```powershell
# 1. Install core tools via winget
winget install --id Git.Git -e --source winget
winget install --id OpenJS.NodeJS.LTS -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id MongoDB.CommunityServer -e --source winget
winget install --id Ollama.Ollama -e --source winget

# 2. Restart your terminal, then verify
python --version    # 3.11 or newer
node --version      # v22 or newer (if winget gave you an older LTS, grab "Node.js 22" from nodejs.org)
git --version

# 3. Start MongoDB
#    Open PowerShell as Administrator:
Get-Service MongoDB | Start-Service

# 4. Start Ollama (opens a background terminal)
ollama serve

# 5. In a NEW terminal, pull the default LLM (~815MB)
ollama pull gemma3:1b

# 6. (Optional) llama.cpp — download a prebuilt release
#    Go to: https://github.com/ggml-org/llama.cpp/releases
#    Download the latest Windows zip (e.g. llama-*-bin-win-x64.zip)
#    Extract it and add the folder to your system PATH
#    Verify: llama-server --help
```

> **Note for Windows users:** TrustRAG uses bash scripts (`scripts/start_local_llm.sh`, `scripts/setup.sh`). Install **Git for Windows** (which includes Git Bash) and run those scripts — plus every `curl` example in this guide — from **Git Bash, not PowerShell**. (PowerShell has its own `curl` alias that speaks a different dialect and will mangle the commands below; if you must stay in PowerShell, use `curl.exe`.) Python/Node/`winget` commands work in both shells.

</details>

---

### Step 2 — Clone and configure

```bash
# Clone the repo
git clone https://github.com/MaithreshVaddi-27/TrustRAG.git
cd TrustRAG

# Create your local environment file
cp .env.example .env

# Generate a JWT secret (required for authentication)
python3 -c "import secrets; print(secrets.token_hex(64))"
# Windows (no python3 alias): py -c "import secrets; print(secrets.token_hex(64))"

# Copy that output into your .env file as JWT_SECRET
# Open .env in your editor and paste it:
#   JWT_SECRET=<the output from above>
```

> **Tip:** Run the setup checker to see if you missed anything:
> ```bash
> ./scripts/setup.sh
> ```
> It'll tell you exactly what's missing and how to fix it.

---

### Step 3 — Start services

You have two options: **Native** (recommended for development — faster, lighter) or **Docker** (good for staging/production testing).

#### Option A: Native (recommended)

Open **three terminal tabs**:

**Terminal 1 — Local LLM server (pick one):**

```bash
# Using Ollama (easiest — just make sure it's running)
ollama serve    # if not already running via brew services / systemctl

# OR using llama.cpp (auto-detects Metal on Mac, CUDA on Linux;
# on Windows, run this from Git Bash)
./scripts/start_local_llm.sh
```

> **Which one?** Ollama is the smoothest start on all three OSes. Pick llama.cpp when you want explicit control over quantized GGUF models and KV-cache budgets on tight hardware (e.g. 8 GB unified memory). On Linux you can keep Ollama alive across reboots with `sudo systemctl enable --now ollama`.

**Terminal 2 — Backend API:**

```bash
cd apps/api

# Create virtual environment (first time only)
python3 -m venv .venv            # Windows: py -3.11 -m venv .venv  (or: python -m venv .venv)
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
# local-models = torch + sentence-transformers for the default HuggingFace
# embeddings (also needed once for the optional ONNX export below).
pip install -e ".[dev,local-models]"

# Optional: discover installed models so they show up in the UI immediately
python ../../scripts/discover_local_models.py

# Start the server (hot-reload enabled)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Frontend:**

```bash
cd apps/web
npm ci          # clean, reproducible install from package-lock.json
npm run dev     # open http://localhost:5173
```

#### Option B: Docker Compose

```bash
# Starts three containers: Qdrant (vector store), the FastAPI backend,
# and the React frontend. MongoDB is NOT containerized — it must already
# be running on your host (see Step 1), reached via host.docker.internal.
docker compose up -d --build

# Check health (public endpoint, no auth needed)
curl -s http://localhost:8000/api/v1/health | jq

# Follow the backend logs if something looks off
docker compose logs -f api

# Stop everything
docker compose down
```

> Prefer running the frontend locally (`npm run dev` in `apps/web`) while developing — hot-reload is instant and you can keep the backend in Docker. Just point it at the same API with `VITE_API_URL=http://localhost:8000`.

---

### Step 4 — Open the UI

Open **http://localhost:5173** in your browser. You'll see the TrustRAG workbench.

1. **Register** a new account (first time only — your credentials never leave `localhost`).
2. **Create a Knowledge Base** and upload some documents (.pdf, .txt, .md, .docx, .csv, .json, .html). Watch the trace stream while it ingests.
3. **Ask a question** — TrustRAG retrieves evidence, drafts a grounded answer, splits it into atomic claims, and checks every single one against your documents. Open the **Claims** tab to see each verdict with its citations, and **Trace** to watch the self-healing loop think.

The default local model is `gemma3:1b` via Ollama (or `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` via llama.cpp). Both run on your CPU — no GPU required.

---

## Try it from the command line

No UI needed — here's the full flow via `curl`. (On Windows, run these from **Git Bash** — PowerShell's built-in `curl` alias will mangle the quoting. On all three OSes, `/tmp/sample.txt` can be any scratch path with write access.)

```bash
BASE=http://localhost:8000/api/v1

# 1. Register
curl -s -X POST $BASE/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"Password123!","full_name":"Your Name"}'

# 2. Login (grab the token)
TOKEN=$(curl -s -X POST $BASE/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"Password123!"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

# 3. Create a knowledge base
KB_ID=$(curl -s -X POST $BASE/knowledge-bases \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Documents","description":"Test KB"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# 4. Upload a document
cat << 'EOF' > /tmp/sample.txt
Effective from: 2026-01-01
Effective until: 2026-12-31

# Refund Policy
Annual contract customers can get a full refund within 30 days.
Monthly subscriptions can be canceled anytime with immediate effect.
Data backups are retained for 90 days after deactivation.
EOF

curl -s -X POST $BASE/knowledge-bases/$KB_ID/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/sample.txt;type=text/plain"

# 5. Ask a question
sleep 2  # let indexing finish
ANALYSIS_ID=$(curl -s -X POST $BASE/analyses \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"knowledge_base_id\":\"$KB_ID\",\"query\":\"What is the refund policy?\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# 6. Stream the live execution trace. The JWT never goes in the URL —
# mint a short-lived single-use ticket first, then stream with it.
TICKET=$(curl -s -X POST $BASE/analyses/$ANALYSIS_ID/stream-ticket \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['ticket'])")
curl -N "$BASE/analyses/$ANALYSIS_ID/stream?ticket=$TICKET"

# 7. Get the final answer
curl -s $BASE/analyses/$ANALYSIS_ID -H "Authorization: Bearer $TOKEN" | jq
```

---

## Architecture

```
TrustRAG/
├── apps/
│   ├── api/                        # FastAPI backend
│   │   ├── app/
│   │   │   ├── agent/              # LangGraph state machine & recovery loop
│   │   │   ├── api/                # Routers, auth, Pydantic schemas
│   │   │   ├── core/               # Config, logging, security, model registry, ONNX embeddings, memory guard
│   │   │   ├── db/                 # MongoDB (async) & Qdrant clients
│   │   │   ├── generation/         # LLM prompts and grounded generation
│   │   │   ├── ingestion/          # PDF/DOCX/TXT/MD/CSV/JSON/HTML parsers, chunker, OCR fallback
│   │   │   ├── retrieval/          # Dense search, BM25, RRF fusion, reranker
│   │   │   ├── services/           # Business logic: KB, analysis, auth
│   │   │   └── verification/       # Batch NLI verifier & SHA-256 auditor
│   │   ├── config/models.yaml      # Model IDs, thresholds, tuning
│   │   └── tests/                  # 326 tests (all passing: eval, sparse_bm25, qdrant, ocr, reranker, router, citations, claim-retrieval, lifecycle, delete-safety + core suites)
│   │
│   └── web/                        # React 18 + Vite 6 frontend
│       ├── src/
│       │   ├── components/         # ClaimInspector, EvidenceViewer, ExecutionTrace
│       │   ├── layouts/            # AppLayout, Sidebar, AuthGuard
│       │   ├── pages/              # 13 lazy-loaded pages
│       │   └── lib/                # API client, auth store, SSE streaming
│       └── package.json
│
├── docs/                           # Project documentation
│   ├── TRUSTRAG_specs.md           # Full product specification
│   ├── architecture/               # System design, ADRs
│   ├── audits/                     # Unified senior audit (2026-09-11)
│   ├── deployment/                 # Deployment guide
│   ├── security/                   # Threat model, security controls
│   └── evaluation/                 # Methodology + frozen baseline (25 queries) + metrics harness + live runner
│
├── scripts/
│   ├── discover_local_models.py    # Pre-boot model discovery snapshot
│   ├── export_bge_onnx.py          # Export BGE-small to ONNX (torch-free embeddings)
│   ├── eval_embedding_parity.py    # Prove torch-vs-ONNX vector parity before switching
│   ├── start_local_llm.sh          # Hardware-aware llama-server launcher
│   ├── apply_ports.py              # Propagate port changes everywhere
│   ├── setup.sh                    # Prerequisite checker
│   └── clear_qdrant.py             # Qdrant collection purge utility
│
├── load-test/
│   └── smoke.js                    # k6 smoke test
│
├── config/ports.yaml               # Single source of truth for service ports
├── docker-compose.yml              # Qdrant + FastAPI backend + React frontend (MongoDB stays on the host)
└── .env.example                    # Environment template
```

---

## API reference

Base URL: `http://localhost:8000/api/v1`. Interactive docs at `/docs`. Auth is Bearer JWT (`POST /auth/login` → `Authorization: Bearer <token>`). Rate limits: 10 analyses/min, 20 auth/min, 10 uploads/min, 10 URL ingests/min.

### Auth (`/auth`)

| Method & path | Purpose |
|---|---|
| `POST /auth/register` | Create account |
| `POST /auth/login` | Verify credentials, return access JWT |
| `GET /auth/me` | Current user profile |
| `POST /auth/logout` | Revoke current token |

### Knowledge bases & documents

| Method & path | Purpose |
|---|---|
| `POST /knowledge-bases` | Create a KB |
| `GET /knowledge-bases` | List your KBs |
| `GET /knowledge-bases/{kb_id}` | KB metadata |
| `DELETE /knowledge-bases/{kb_id}` | Delete a KB (cascades; vectors dropped first so a failure keeps metadata for retry) |
| `POST /knowledge-bases/{kb_id}/snapshots` | Snapshot a KB for rollback (201) |
| `POST /knowledge-bases/{kb_id}/rollback/{snapshot_id}` | Roll back to a snapshot — returns the NEW live id, clients must swap (409 on vector-less snapshots) |
| `GET /knowledge-bases/{kb_id}/documents` | List documents in a KB |
| `POST /knowledge-bases/{kb_id}/documents` | Upload a document (.pdf/.txt/.md/.docx/.csv/.json/.html) |
| `POST /knowledge-bases/{kb_id}/documents/from-url` | Ingest a document from URL |
| `GET /documents/{doc_id}` | Document details |
| `DELETE /documents/{doc_id}` | Delete a document (fails closed on vector errors — record kept for retry, never orphaned points) |

### Analyses — the RAG pipeline

| Method & path | Purpose |
|---|---|
| `POST /analyses` | Start an analysis run (201) |
| `GET /analyses` | Paginated history (`limit` ≤ 200, `skip`) |
| `GET /analyses/{id}` | Run details + verdict |
| `GET /analyses/{id}/claims` | Atomic claims with NLI verdicts |
| `GET /analyses/{id}/evidence` | Retrieved segments with scores |
| `GET /analyses/{id}/trace` | Step-by-step execution timeline |
| `GET /analyses/{id}/detail` | Answer + claims + evidence + trace in one call |
| `GET /analyses/{id}/export` | Audit & compliance dossier |
| `POST /analyses/{id}/stream-ticket` | Short-lived SSE ticket |
| `GET /analyses/{id}/stream` | Live execution trace (Server-Sent Events) |

### Verification artifacts, experiments & ops

| Method & path | Purpose |
|---|---|
| `GET /claims`, `GET /evidence`, `GET /conflicts` | Cross-run claim/evidence/conflict listings |
| `POST /experiments`, `GET /experiments`, `GET /experiments/{exp_id}` | Experiment runs |
| `GET /experimentation/flags` | Feature flags |
| `GET /models/providers` | AI provider status + installed models |
| `GET /models/hardware` | Hardware acceleration & resource profile |
| `POST /models/memory/trim` | Heap compaction + GC |
| `GET /health` | Public health (status, version — for load balancers) |
| `GET /health/detailed` | Detailed health (services, models, hardware — requires auth) |
| `/internal/*` (`tokens`, `ingest/*`, `search`, `verify/claims`, `health`, `status`) | Service-to-service diagnostics — not for UI use |

---

## Configuration reference

Two files own all non-secret config. **Env vars always win** over both.

| File | Owns | Example knobs |
|---|---|---|
| `apps/api/config/models.yaml` (`config_version: 1.13`) | Model IDs, thresholds, tuning | LLM/embedding IDs, `retrieval.*` (incl. `sparse_k1/b`, `query_router.*`), `reranker.*` (off by default, `top_k: 20` depth cap), `ingestion.*` (incl. `chunking_strategy`, `ocr.*`), `reliability.*`, `recovery.*`, `cost_controls.*` (incl. `max_claim_retrievals`, `claim_retrieval_top_k`), `verification.*`, `optimization.*` |
| `config/ports.yaml` | Ports + derived base URLs | backend `8000`, frontend `5173`, Ollama `11434`, llama.cpp `8080`, MongoDB `27017`, Qdrant `6335:6333` host:container |

Change a port in `ports.yaml`, then run `python3 scripts/apply_ports.py` (CI enforces with `--check`).

**`.env` holds secrets + deploy overrides only** (see `.env.example`): `JWT_SECRET` (required, ≥32 chars), `MONGODB_URI`, `QDRANT_URL` (`local` = embedded, no server), `CORS_ORIGINS`, plus optional `AI_PROVIDER` / `EMBEDDING_PROVIDER` / model overrides and cloud keys (`GEMINI_API_KEY`, `NVIDIA_API_KEY`, `TAVILY_API_KEY`). Accepted aliases (e.g. `OLLAMA_HOST` for `OLLAMA_BASE_URL`) are listed in `.env.example` — note a globally-exported `OLLAMA_HOST` is picked up automatically.

**Embedding providers:** `EMBEDDING_PROVIDER=huggingface` (default, PyTorch) or `EMBEDDING_PROVIDER=onnx` (ONNX Runtime, torch-free, ultra-low RAM). `EMBEDDING_MODEL` must match KB pin.

> **Embedding pin:** the embedding model is pinned per knowledge base at ingest time. Switching `EMBEDDING_MODEL` afterwards requires re-creating the KB — old vectors won't match the new dimensionality.

---

## Technology stack

| Layer | What | Details |
|---|---|---|
| **Frontend** | React 18 + Vite 6 | Tailwind CSS, Motion springs, dark glassmorphic theme |
| **Backend** | FastAPI + Python 3.11 | Async REST API, Pydantic v2, SSE streaming |
| **Local LLMs** | Ollama / llama.cpp | `gemma3:1b` (Ollama) or `LiquidAI/LFM2.5-1.2B` (llama.cpp) |
| **Cloud LLMs** | Gemini / NVIDIA NIM | Optional — for when you want cloud-scale reasoning |
| **Embeddings** | BAAI/bge-small-en-v1.5 | 384d local vectors, zero API cost; **ONNX Runtime** or HuggingFace (PyTorch) |
| **Vector Store** | Qdrant | Embedded Rust engine, INT8 quantization, on-disk vectors |
| **Database** | MongoDB 7.0 | Metadata, chunks, claims, execution traces |
| **Agent Protocol** | MCP (JSON-RPC 2.0) | Universal tool interface for AI coding agents |
| **State Machine** | LangGraph | Multi-node adaptive recovery loop |
| **Auth** | JWT + Bcrypt | HS256 tokens, 12-round hashing, rate limiting |

---

## Testing

TrustRAG has 326 backend tests, 22 frontend tests, and 2 E2E tests — all passing.

**Backend (same on all three OSes — run from Git Bash on Windows):**
```bash
cd apps/api
source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate

# Run all tests (mocked — no MongoDB, LLM, or Qdrant needed)
pytest tests/ -q

# Lint + format check (CI enforces both)
ruff check app/ tests/
ruff format --check app/ tests/
```

**Frontend:**
```bash
cd apps/web

# Unit & component tests
npm run test

# E2E tests (starts backend + browser automatically)
npm run test:e2e

# Lint & type check
npm run lint

# Production build
npm run build
```

**Load testing (requires k6 — install it first):**
```bash
# macOS:
brew install k6

# Linux (Ubuntu/Debian):
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://dl.k6.io/key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/k6-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt update && sudo apt install k6

# Windows: choco install k6   (or grab the installer from the link below)
# Full instructions for every platform: https://grafana.com/docs/k6/set-up/install-k6/
```
```bash
# Backend must be running first (Terminal 2), then:
k6 run load-test/smoke.js
# Thresholds: <1% failures, p95 < 300ms, p99 < 500ms
```

### What each backend test file covers

| File | Covers |
|---|---|
| `test_agent.py` | LangGraph nodes: retrieval, generation, verification, recovery, query-rewrite sanitization |
| `test_analyses.py` / `test_auth.py` / `test_kb.py` / `test_experiments.py` / `test_health.py` | REST endpoints for analyses, auth, knowledge bases, experiments, health |
| `test_config.py` | `models.yaml`/`ports.yaml` loading, validation, snapshots |
| `test_ports.py` | Port-registry drift (`apply_ports.py --check` equivalent) |
| `test_generation.py` | Grounded answer generation, ABSTAIN rules, scaffold stripping |
| `test_verification.py` | Claim decomposition, fused + batch/individual NLI, tolerant near-miss parsing, fallback budget, verdict math |
| `test_integrity.py` | SHA-256 evidence audit, temporal windows |
| `test_retrieval.py` / `test_preprocessor.py` / `test_ingestion.py` | Hybrid retrieval, fusion_top_k bound, text normalization, chunking, ingestion pipeline |
| `tests/eval/` (`test_eval_metrics.py`, `test_baseline_dataset.py`) | Phase-0 harness: metric math (hand-computed), frozen 25-query dataset schema + fixture-snippet validation |
| `test_sparse_bm25.py` | BM25 TF saturation, length norm, query-side weights, zone ordering |
| `test_qdrant.py` | Collection init + IDF sparse migration (create/keep/recreate/fail-open) |
| `test_ocr.py` | OCR density gate, confidence drop, fail-open parsing, provenance plumbing (mocked engine) |
| `test_reranker.py` | Enabled-path scoring/adaptive top-4/depth cap + disabled/None/exception fallbacks (mocked CrossEncoder) |
| `test_chunking_strategies.py` | Newline-preserving normalization, strategy wiring, semantic true offsets, progressive gap-freedom, layout table grouping + order, OCR passthrough |
| `test_citations.py` | Inline `[Segment N]` extraction, invalid-ref stripping matrix, end-to-end generation wiring (mocked LLM) |
| `test_claim_retrieval.py` | Targeted per-claim retrieval (dedup, budget, outage), NEUTRAL→SUPPORTED flip with fresh linkage, CONTRADICTED exclusion, inline-cite union |
| `test_router.py` | Router classify/split/merge matrix, fan-out concurrency + outage degradation, node-level comparison fan-out |
| `test_lifecycle.py` | Snapshot/rollback routes, empty-snapshot 409 guard, OCR-preserving snapshots, document-delete vector purge |
| `test_delete_safety.py` | Fail-closed deletes (vectors-before-metadata ordering, no swallowed vector errors), router cap floor |
| `test_local_llm.py` / `test_hardware.py` | Ollama/llama.cpp clients, model registry, hardware profiles |
| `test_disk_cache.py` / `test_semantic_cache.py` | Embedding disk cache, semantic answer cache |
| `test_rate_limit.py` | Per-route rate limiting |
| `test_search_mcp.py` | MCP web-search tools (Tavily/DuckDuckGo/hybrid) |
| `test_internal.py` | Internal service endpoint input contracts (422s, not 500s) |

---

## CI/CD

Every push/PR to `main`, `develop`, or `ui-redesign` runs two workflows (least-privilege tokens, concurrency-cancelled):

**CI (`.github/workflows/ci.yml`)** — `backend-lint` (ruff + format + ports drift) → `backend-test` (pytest) → `backend-config-validate` (`models.yaml` schema + secret scan) → `frontend-lint` (eslint + vitest) → `frontend-build` → `e2e` (Playwright + k6 against MongoDB service + live backend) → `docker-build` (API image + advisory Trivy HIGH/CRITICAL SARIF to code scanning) → `ci-gate` (fails on any failure/cancel/skip).

**Security (`.github/workflows/security.yml`)** — weekly Monday scan plus every push: `python-audit` (`pip-audit`, strict), `npm-audit` (high+), `secret-scan` (rejects committed `.env`, scans `models.yaml`), `sast` (Bandit on `app/`).

---

## Frontend pages

All 13 pages are lazy-loaded and auth-guarded (public: landing/login/register only):

| Route | Page | Purpose |
|---|---|---|
| `/` | Landing | Product intro |
| `/login`, `/register` | Auth | Sign in / create account |
| `/dashboard` | Dashboard | Overview of KBs, runs, reliability |
| `/playground` | Playground | Ask questions, watch live verification + recovery |
| `/knowledge-bases` | Knowledge Bases | Create KBs, upload documents |
| `/evidence` | Evidence | Retrieved segments across runs |
| `/claims` | Claims | Atomic claims with NLI verdicts |
| `/conflicts` | Conflicts | Source & claim contradictions |
| `/experiments` | Experiments | Experiment runs |
| `/traces/:id` | Trace | Per-analysis execution timeline |
| `/settings` | Settings | Providers, models, preferences |
| `*` | NotFound | 404 |

---

## Troubleshooting

**MongoDB won't connect:**
```bash
# macOS
brew services start mongodb-community

# Linux
sudo systemctl enable --now mongod

# Windows (PowerShell as Admin)
Get-Service MongoDB | Start-Service

# Verify it's running
mongosh --eval "db.runCommand({ ping: 1 })"
```

**Ollama not responding:**
```bash
# Check if it's running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve    # or: brew services start ollama (macOS)
```

**No offline warning / empty model list:**
The Playground reads `/models/providers` — if that request fails you get an empty dropdown with no explanation. First check the backend is running **current** code (`uvicorn` without `--reload` serves stale code after `git pull`), then hard-refresh the browser (stale bundle). The endpoint degrades instead of 500ing, so a persistent empty list means the API itself is unreachable — see `useBackendHealth` / the API Online pill.

**llama-server running but analyses fail (503):**
Distinguish the two cases before restarting anything:
```bash
curl -m 5 http://127.0.0.1:8080/v1/models
```
- Fails instantly → server is down: `./scripts/start_local_llm.sh`
- Hangs → server is overloaded/starting: wait, then retry (the preflight probe retries once; a cold model on a busy host can miss the first sample)
- Instant 200 but UI still red → stale backend/frontend processes; restart them
- 503 says "not answering" (not "not reachable") → slow server, not a dead one — do not reinstall models for this

**Port already in use:**
```bash
# Find what's using the port
lsof -i :8000     # macOS/Linux
netstat -ano | findstr :8000    # Windows

# Kill it or change the port in config/ports.yaml
```

**Firewall blocks localhost (per OS):**
The first launch often triggers a firewall prompt — that's expected, not an error. Allow private-network access and reload the page:
- **macOS:** System Settings → Network → Firewall → allow incoming connections for `Python` / `node` when prompted.
- **Linux:** `sudo ufw allow 8000/tcp && sudo ufw allow 5173/tcp` — but only if `ufw` is active (check with `sudo ufw status` first).
- **Windows:** Windows Defender Firewall will prompt for Python and Node.js — tick **Private networks** (leave Public unchecked) and continue.

**Windows: `python3` not recognized / venv won't activate:**
Windows installs the launcher as `py`, not `python3`. Create the environment with `py -3.11 -m venv .venv`, then activate with `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell). If PowerShell refuses the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once — or simply do everything in Git Bash.

**After `git pull` (existing users):**
```bash
cd apps/api && source .venv/bin/activate && pip install -e ".[dev,local-models]"  # picks up new deps (e.g. onnxruntime)
cd ../web && npm ci
```
No database migration is needed (Mongo/Qdrant schemas unchanged; new `models.yaml` keys have safe defaults). If you modified `models.yaml` locally, `git` may ask you to resolve the conflict — keep your values and copy any new keys (e.g. `fused_decompose_verify`, `max_seq_length`) from `models.yaml` in the pull.

**Qdrant port confusion:**
When running via Docker, Qdrant maps host port `6335` to container port `6333`. From your host, use `http://localhost:6335`. Inside Docker, services talk directly to `http://qdrant:6333`.

**Embedding model download on first boot:**
The BGE embedding model (~120MB) downloads automatically from HuggingFace on the first API startup. It's cached at `~/.cache/huggingface` after that. If you hit rate limits, set `HF_TOKEN` in your `.env`.

**Wrong Ollama host picked up:**
`OLLAMA_HOST` is an accepted alias for `OLLAMA_BASE_URL` — if it's exported globally (Ollama sets it on some installs), the backend uses it silently. Unset it or set `OLLAMA_BASE_URL` explicitly in `.env` to override.

**Changed embedding model, retrieval looks off:**
Vectors are pinned per knowledge base at ingest. After switching `EMBEDDING_MODEL`/`EMBEDDING_DIM`, re-create the KB and re-upload — old vectors won't match.

**0% FAILED even with evidence present:**
Open the Claims tab and read the per-claim explanations: `NEUTRAL` with "Verification could not be completed" means the local NLI judge failed to emit a verdict (check the model server logs), while "fallback budget exhausted" would mean claims were never attempted. Verification judges the same ~3000-char context the answer was generated from — claims drawn from beyond it can't verify.

---

## Documentation

| Document | What's in it |
|---|---|
| [Architecture](docs/architecture/architecture.md) | Technical design of the LangGraph state machine, hybrid search, claim decomposition |
| [Decision Log](docs/architecture/decision-log.md) | 22 ADRs explaining technology choices and tradeoffs |
| [Security Controls](docs/security/security-controls.md) | JWT auth, anti-IDOR, SSRF defense, defensive headers |
| [Threat Model](docs/security/threat-model.md) | STRIDE analysis, attack surface, countermeasures |
| [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) | Production container setup, cloud hosting, env management |
| [System Audit](docs/audits/2026-09-11_unified_senior_audit.md) | Multi-role senior audit: frontend, backend, AI/ML, security, optimization, testing |
| [Implementation Status](docs/IMPLEMENTATION_STATUS_2026-09-11.md) | What was fixed, test state, remaining work |
| [Roadmap](docs/ROADMAP.md) | Milestones, completed phases, upcoming work |

---

## License

MIT — see [LICENSE](LICENSE).
