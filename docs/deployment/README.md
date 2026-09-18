# TRUSTRAG — Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Variables](#environment-variables)
3. [Local Development](#local-development)
4. [MongoDB Atlas Setup](#mongodb-atlas-setup)
5. [Qdrant Cloud Setup](#qdrant-cloud-setup)
6. [Gemini API Setup](#gemini-api-setup)
7. [Backend Deployment](#backend-deployment)
8. [Frontend Deployment](#frontend-deployment)
9. [CORS Configuration](#cors-configuration)
10. [Health Checks](#health-checks)
11. [Indexing & Re-indexing](#indexing--re-indexing)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker + Docker Compose (for local dev)
- Git

---

## Environment Variables

Copy `.env.example` to `.env` and fill in what you need (split responsibilities: secrets/endpoints here; model IDs and providers in `apps/api/config/models.yaml`):

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URI` | Yes | MongoDB Atlas or local connection string |
| `MONGODB_DATABASE` | Yes | Database name (default: `trustrag_db`) |
| `QDRANT_URL` | Yes | Qdrant URL: `local` (embedded) or `http://localhost:6333` |
| `QDRANT_API_KEY` | Prod only | Qdrant Cloud API key |
| `JWT_SECRET` | Yes | Min 32-char random secret |
| `CORS_ORIGINS` | Yes | Comma-separated allowed origins |
| `GEMINI_API_KEY` | Conditional | Only if models.yaml uses gemini |
| `NVIDIA_API_KEY` | Conditional | Only if models.yaml uses nvidia |
| `TAVILY_API_KEY` | No | Web search grounding (else free DuckDuckGo) |
| `HF_TOKEN` | No | Read-only token to avoid Hub rate-limits on embedding download |
| `APP_ENV` | No | `development` (default) + `staging`/`production` |
| `LOG_LEVEL` | No | `INFO` (default) |
| `JWT_EXPIRY_MINUTES` | No | JWT lifetime, default 60 |
| `TRUSTED_PROXY_IPS` | No | Proxy peers allowed to supply `X-Forwarded-For` (empty = direct) |
| `RATE_LIMIT_*_PER_MINUTE` | No | Per-client ceilings (analyses/auth/upload/url-ingest) |
| `CACHE_DIR` | No | SQLite embedding + semantic-cache directory |
| `LOCAL_LLM_MAX_CONCURRENCY` | No | Concurrent local generations, default 1 (raise only on parallel servers) |
| `AI_PROVIDER`, `EMBEDDING_PROVIDER` (`huggingface`\|`onnx`), `SEARCH_PROVIDER`, `*_MODEL`, `*_BASE_URL`, `EMBEDDING_DIM` | No | Per-deploy overrides; env wins over models.yaml/ports.yaml (see `.env.example`) |
| `FUSED_DECOMPOSE_VERIFY` | No | `1`/`0` kill-switch for the fused decompose+verify fast path |
| `MALLOC_ARENA_MAX`, `TOKENIZERS_PARALLELISM` | No | Allocator tuning (`1`, `false`) to cut glibc/tokenizer RAM overhead |
| `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_PARALLEL` | No | Ollama **server** memory tuning — set in the shell before `ollama serve`, not read by the backend |

Generate a strong JWT secret:
```bash
python -c "import secrets; print(secrets.token_hex(64))"
```

---

## Local Development

### Option 1: Docker Compose (recommended)

```bash
# 1. Set up environment
cp .env.example .env
# Fill in GEMINI_API_KEY (if using cloud) and MONGODB_URI in .env
# QDRANT_URL will be overridden to http://qdrant:6333 by docker-compose

# 2. Start services
docker compose up

# 3. Verify
curl http://localhost:8000/api/v1/health
# Frontend: http://localhost:5173
```

### Option 2: Manual (local-first lean setup)

```bash
# Local LLM (start before or after backend — either order is fine)
./scripts/start_local_llm.sh   # Metal/CUDA auto-detect + GPU offload
# (or: ollama serve, if AI_PROVIDER=ollama)

# Backend
cd apps/api
python -m venv .venv
source .venv/bin/activate
# local-models = torch for default HuggingFace embeddings (also needed once
# for the optional ONNX export below).
pip install -e ".[dev,local-models]"
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd apps/web
npm ci
npm run dev
```

> **Embeddings:** TRUSTRAG runs local BGE (384d) embeddings — zero cloud cost, zero keys. `EMBEDDING_PROVIDER=huggingface` (PyTorch, via the `local-models` extra) or `onnx` (torch-free ONNX Runtime; one-time export with `python scripts/export_bge_onnx.py`, then `EMBEDDING_PROVIDER=onnx`). Cloud embeddings were removed. (`EMBEDDING_MODEL` selects between BGE and MiniLM.)
>
> **OCR:** scanned/image PDF pages fall back to local RapidOCR-ONNX (`rapidocr-onnxruntime`, a default `pyproject.toml` dependency reusing the shipped `onnxruntime` — no Dockerfile change, no system binaries). Models download once to `~/.onnx` on the first scanned page and are cached afterwards: **pre-warm on deploy** (ingest one scanned PDF) or the first scanned upload stalls on the download. Disable per-deploy with `ingestion.ocr.enabled: false` in `models.yaml` if scanned input is out of scope.
>
> **Re-index windows (combine into one operator re-upload):** pre-IDF Qdrant collections recreate empty on next init (sparse values are scoring-incompatible); the newline-preserving normalization change shifts chunk text/embeddings; switching `ingestion.chunking_strategy` changes boundaries. All three require document re-upload.

---

## MongoDB Atlas Setup

1. Create a free account at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create a **free M0 cluster** (512MB storage — sufficient for MVP)
3. Create a database user with read/write access
4. Whitelist your IP address (or `0.0.0.0/0` for development — not recommended for production)
5. Get the connection string: `Clusters → Connect → Connect your application → Python`
6. Set `MONGODB_URI` in `.env`

> **Free tier note:** M0 clusters have a 500 connections limit and no dedicated RAM. Sufficient for MVP.

---

## Qdrant Cloud Setup

For production, use Qdrant Cloud free tier:

1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster (1GB storage)
3. Get the cluster URL and API key
4. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

For local development, use the Docker Compose Qdrant service:
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=   # empty = no auth
```

---

## Gemini API Setup (conditional — only if a Gemini provider/model is selected)

The default stack is fully local (llama.cpp/Ollama + BGE embeddings) and boots with zero keys.

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key (free tier available)
3. Set `GEMINI_API_KEY` in `.env`
4. Verify the configured model ID (`gemini-2.5-flash` family in `config/models.yaml`) is available for your API key

> **Model ID verification:** Run `python -c "from app.core.model_registry import get_llm; print(get_llm())"` after setting up credentials.

---

## Backend Deployment

### Render.com (Recommended — 100% Free, Zero Friction)

Render provides free Docker web service hosting with zero credit-card setup hurdles and automatic SSL.

> 💡 **If you already have a Render account using Google Login**:
> 1. **Option A (Link GitHub)**: In Render Dashboard, click your Profile Avatar (top-right) → **Account Settings** → **Connected Accounts** → click **Connect GitHub**. Then deploy via **Blueprint** (`render.yaml`).
> 2. **Option B (Public Git URL)**: Click **New +** → **Web Service** → paste public URL `https://github.com/MaithreshVaddi-27/TrustRAG` → select branch `ui-redesign` (no GitHub account linking required!).

#### Method 1: Web Service Deploy via Dockerfile (Fastest)
Deploy directly using `apps/api/Dockerfile`:
1. Log in to [render.com](https://render.com) using your Google or GitHub account.
2. In the Render Dashboard, click **New +** → **Web Service**.
3. Select your repository: `TrustRAG` (branch: `ui-redesign`).
4. Configure the environment variables:
   - `JWT_SECRET`: *(auto-generated by Render)*
   - `GEMINI_API_KEY`: *(only if models.yaml uses gemini)*
   - `MONGODB_URI`: Paste your MongoDB Atlas connection string
   - `QDRANT_URL`: Paste your Qdrant cluster URL
   - `QDRANT_API_KEY`: Paste your Qdrant API key
5. Click **Apply**.
6. Render builds the container and provisions your live HTTPS URL:
   ```
   https://trustrag-api.onrender.com
   ```
7. Verify health:
   ```bash
   curl https://trustrag-api.onrender.com/api/v1/health
   ```

### Google Cloud Run (Alternative for GCP Users)

```bash
docker build -t trustrag-api ./apps/api
docker run -p 8000:8000 --env-file .env trustrag-api
```

---

## Frontend Deployment

### Cloudflare Pages (Recommended for Production)

Cloudflare Pages provides global CDN edge delivery with zero-config preview deployments and custom domains.

> 💡 **If you already have a Cloudflare account using Google Login**:
> Log in with your Google account, then click **Connect to Git** to link GitHub to your Cloudflare account, OR deploy directly via `npx wrangler pages deploy dist`.

#### Method A: Git Integration (Recommended)
1. Log in to the [Cloudflare Dashboard](https://dash.cloudflare.com/) using your Google account and navigate to **Workers & Pages** → **Create application** → **Pages** → **Connect to Git**.
2. Select your `TrustRAG` repository.
3. Configure the build parameters:
   - **Project name**: `trustrag`
   - **Production branch**: `ui-redesign`
   - **Framework preset**: `Vite`
   - **Root directory**: `apps/web`
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
4. Under **Environment variables**, set:
   - `VITE_API_URL`: `https://trustrag-api.onrender.com` (your live Render backend URL).
5. Click **Save and Deploy**.

#### Method B: Direct Upload via Wrangler CLI
```bash
cd apps/web
npm install
VITE_API_URL="https://trustrag-api.onrender.com" npm run build
npx wrangler pages deploy dist --project-name trustrag
```

#### SPA Routing & Security Headers
The repository automatically includes:
- `apps/web/public/_redirects`: Routes all SPA paths (`/playground`, `/knowledge-bases`, `/evidence`, `/claims`, etc.) to `/index.html 200` without 404s.
- `apps/web/public/_headers`: Enforces `X-Frame-Options: DENY`, `nosniff`, and cache rules on immutable static bundles.

### Vercel / Netlify (Alternative)

- Root directory: `apps/web`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL=https://your-cloud-run-url.a.run.app`

---

## CORS Configuration

Set `CORS_ORIGINS` to match your frontend URL:

```env
# Development
CORS_ORIGINS=http://localhost:5173

# Production
CORS_ORIGINS=https://trustrag.netlify.app

# Multiple origins
CORS_ORIGINS=https://trustrag.netlify.app,https://trustrag.vercel.app
```

---

## Health Checks

```bash
# Public health (load balancers, Docker HEALTHCHECK) — minimal, no auth
GET /api/v1/health

# Expected response
{
  "status": "ok",
  "timestamp": "2026-09-12T10:00:00Z",
  "app": "TRUSTRAG",
  "version": "0.1.0"
}

# Detailed health (services, models, hardware) — requires Bearer JWT
GET /api/v1/health/detailed
# curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/health/detailed
```

---

## Indexing & Re-indexing

MongoDB indexes are created automatically on startup (idempotent).

When you change the embedding model in `models.yaml`:
1. Increment `embedding.version`
2. All existing Qdrant collections become stale
3. Re-ingest documents: delete the old Qdrant collection and re-upload documents
4. The system will detect embedding version mismatches and warn

The same re-upload applies when the **sparse config changes** (pre-IDF collections
recreate empty on next init), when **normalization/chunking changes** (chunk text and
embeddings shift), or when switching **`ingestion.chunking_strategy`**. Combine all
three into a single operator re-upload window.

---

## Troubleshooting

### API returns 503 on startup

Check MongoDB connectivity (detailed endpoint needs auth):
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/health/detailed | jq .services
# Look for "mongodb": "degraded"
```

Verify `MONGODB_URI` is correct and Atlas IP whitelist includes your server IP.

### Embedding latency & zero-GPU operation

By default TRUSTRAG uses local HuggingFace BGE (`BAAI/bge-small-en-v1.5`, 384d) via PyTorch — zero cloud cost and zero required credentials. On Apple Silicon it rides the Metal (MPS) device; on NVIDIA it picks CUDA; CPU hosts fall back cleanly. An in-memory thread-safe LRU cache serves repeat queries instantly.

For sub-16GB hosts or to remove PyTorch from the API process entirely (~500–1000 MB RSS savings), switch to torch-free ONNX Runtime embeddings:

```bash
# One-time export (needs torch + sentence-transformers locally)
python scripts/export_bge_onnx.py
cp apps/api/data/models/bge-small-en-v1.5.onnx apps/api/.model_cache/

# Enable in .env
EMBEDDING_PROVIDER=onnx
```

Cloud embeddings (Gemini/NVIDIA) were removed — embeddings are local-only. Knowledge bases indexed with a retired provider must be re-uploaded.

> **Docker note:** the API image ships `onnxruntime` but neither PyTorch nor model
> weights, so `EMBEDDING_PROVIDER=huggingface` cannot load inside the container —
> use `EMBEDDING_PROVIDER=onnx` and copy the exported model into the running
> container once (see the `model_cache` volume comment in `docker-compose.yml`).
> The tokenizer still downloads from the Hub on first boot (pinned revision).

### Local model server offline

- Start llama.cpp: `./scripts/start_local_llm.sh` (auto-detects Metal/CUDA)
- Or start Ollama: `ollama serve` (only needed when `AI_PROVIDER=ollama`)

### Gemini API errors (only when a Gemini provider/model is selected)

- Verify `GEMINI_API_KEY` is set
- Verify the `gemini-2.5-flash` family model ID is available in your region/plan at [AI Studio](https://aistudio.google.com)
- Check rate limits (Gemini free tier: 15 RPM, 1M tokens/day)

### CORS errors in browser

Ensure `CORS_ORIGINS` in `.env` exactly matches your frontend origin (including protocol and port).
