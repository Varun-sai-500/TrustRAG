# TRUSTRAG — System Architecture

## Overview

TRUSTRAG is an AI reliability workbench that implements a closed-loop reliability and self-healing engine over Retrieval-Augmented Generation (RAG):

```
Query → Route (simple / temporal / comparison / complex, deterministic, no LLM)
      → Retrieve (Dense + BM25-TF/IDF + MCP Live Web) → RRF fusion (fusion_top_k enforced)
      → Rerank (cross-encoder, OFF by default, depth-capped)
      → Grounded Generation with inline [Segment N] citations (Local llama.cpp / Ollama / Gemini / NVIDIA — per request) 
      → Propositional Claim Decomposition → NLI Claim Verification (+ targeted NEUTRAL-only re-retrieval) 
      → Evidence Integrity & Provenance Audit → Threshold Reliability Diagnosis 
      → Adaptive Recovery Loop (LangGraph StateGraph) 
      → Re-verify → Grounded Answer / Safe Abstention
```

The core portfolio differentiator is the **autonomous diagnosis → adaptive recovery loop**, not one-shot retrieval or generation alone.

---

## Component Map

```
React 18 + Vite (Port 5173)
    │
    │ REST /api/v1/... (Reverse Proxy) │ SSE /api/v1/analyses/{id}/stream
    ▼
FastAPI (Python 3.12, Default Port 8000)
    │
    ├─── app/core/         Settings, ModelRegistry, Logging, Security, Exceptions
    │       └── local_llm.py → ChatOllamaClient, ChatLlamaCppClient (LLM-only),
    │                          CLI introspection (ollama list, llama-server --cache-list)
    ├─── app/db/           MongoDB Community / Atlas client, Qdrant client
    ├─── app/ingestion/    Document parsing (+ per-page RapidOCR-ONNX fallback for
    │                      <50-native-char pages), selectable chunking strategies,
    │                      newline-preserving normalization, cryptographic hashing
    ├─── app/retrieval/    Dense (384d BGE) + sparse (client BM25-TF saturation +
    │                      server-side Qdrant Modifier.IDF) + Reciprocal Rank Fusion
    │                      (RRF, fusion_top_k enforced); recreate-on-mismatch for
    │                      pre-IDF collections; cross-encoder reranker (off by
    │                      default, top_k=20 depth cap)
    ├─── app/mcp/          Model Context Protocol (MCP) Server & Dispatcher
    │       ├── local_llm_chat    → Prompt local LLM (Ollama / llama.cpp) over MCP
    │       ├── local_llm_status  → Query local model health & discovery via MCP
    │       ├── tavily_search     → AI-curated RAG search with clean parsed snippets
    │       ├── duckduckgo_search → Zero-config, 100% free web search fallback
    │       └── hybrid_web_search → Parallel execution with URL deduplication
    ├─── app/services/     Search Service (SSRF sanitization, private IP guards);
    │                      KB lifecycle (snapshots, rollback with vector-less guard)
    ├─── app/generation/   Grounded answer generation (Local LLMs or Cloud) with
    │                      inline [Segment N] citations + invalid-ref strip post-check
    ├─── app/verification/ Propositional claim decomposition + NLI entailment +
    │                      targeted NEUTRAL-only claim retrieval (≤3/analysis);
    │                      brackets-exempt scaffold-echo filter
    ├─── app/integrity/    Cryptographic SHA-256 provenance & temporal audit
    ├─── app/agent/        LangGraph stateful self-healing workflow (deterministic
    │                      query router + bounded fan-out inside retrieval_node)
    └─── app/evaluation/   Experiment runner & benchmark metrics
         │
├─── Local Engines:
           │       Ollama (Port 11434, LLM-only): granite4.2:3b-q4_K_M, gemma3:1b
           │       llama.cpp (Port 8080, LLM-only): ibm-granite/granite-4.2-3b-GGUF:Q4_K_M
           │             (+ ibm-granite/granite-4.0-h-1b GGUF)
           │       HuggingFace: BAAI/bge-small-en-v1.5 (384d SOTA embeddings)
         │
         ├─── Cloud Engines, LLM-only (Optional):
          │       Google Gemini: gemini-2.5-flash family (embeddings: local BGE)
         │       NVIDIA NIM: meta/llama-3.3-70b-instruct (embeddings: local BGE)
         │
          ├─── Qdrant (Vector & Payload Store)
          │       Dense vector indexing (384d only — 768d retired) + sparse-text
          │       (Modifier.IDF) + Payload filtering; pre-IDF collections recreate
          │       on next init (re-upload those KBs)
         │
         └─── MongoDB (Operational Data Store)
                 Users, Knowledge Bases, Analyses, Claims, Evidence, Traces
```

---

## Model Context Protocol (MCP) Integration

TRUSTRAG adopts the open **Model Context Protocol (MCP)** specification to decouple agent reasoning from retrieval and external live grounding:

1. **MCP Server (`app/mcp/server.py`)**:
   - Exposes standardized JSON-RPC endpoints: `tools/list` and `tools/call`.
   - Built-in tools:
     - `tavily_search`: High-accuracy AI search tailored for RAG grounding.
     - `duckduckgo_search`: Free live search requiring zero API keys.
     - `hybrid_web_search`: Parallel execution across both engines with automatic URL deduplication.
2. **MCP Client Dispatcher (`app/mcp/client.py`)**:
   - Dispatches agent grounding requests through the standard MCP interface.
   - Converts web results into verified context segments with SHA-256 hashes and citation metadata.
3. **Defense-in-Depth Search Security (`app/services/search_service.py`)**:
   - Strict SSRF sanitization (`sanitize_url` rejects non-HTTP/HTTPS schemes and internal network probes).
   - Hard query boundary limits (`MAX_QUERY_LENGTH = 500`) and 8.0s timeout guards.

---

## LangGraph Self-Healing Workflow

```
       [START]
          │
          ▼
   [retrieval_node] ◄──────────────┐ (Adaptive Recovery Edge)
   (Route: simple/temporal/        │
    comparison/complex → Dense +   │
    Sparse + MCP Web, bounded      │
    fan-out merged by RRF)         │
          │                        │
          ▼                        │
   [generation_node]               │
   (Context-bound synthesis +      │
    [Segment N] citations)         │
          │                        │
          ▼                        │
   [verification_node]             │
   (Propositional NLI + targeted   │
    NEUTRAL-only claim retrieval)  │
          │                        │
          ▼                        │
   [verdict computation]           │
   (compute_verdict thresholds    │
    via verdict.py)                │
     Pass or Fail?                 │
     ├── PASS ─────────────────────┼──────────┐
     │                             │          │
     └── FAIL (within max attempts) │          │
          │                        │          │
          ▼                        │          │
   [recovery_node] ────────────────┘          │
   - Reset stale state["answer"]              │
   - Adaptive strategy selection:             │
     * Query Rewrite (sanitized, empty-guarded)│
     * Expanded Retrieval (capped widening)   │
     * Regenerate (retrieval short-circuit)   │
                                              │
     Max attempts reached?                    │
     ├── Thresholds met ──────────────────────┴──► [ANSWER]
     └── Low confidence / insufficient data ─────► [ABSTAIN]
```

**Guardrails:**
- Bounded strictly by `max_recovery_attempts` in `models.yaml`.
- Router kill-switch + ceiling (`retrieval.query_router.enabled`, `max_sub_queries: 3`); claim-retrieval budget (`cost_controls.max_claim_retrievals: 3`); reranker depth floor at `fusion_top_k`; per-branch retrieval timeouts (45 s each, 60 s total).
- State reset logic: `state["answer"] = None` and `state["claims"] = []` prevent stale abstentions from propagating when newly retrieved segments provide the missing facts.
- Refusal gate: hedged refusals skip decomposition/NLI entirely (zero LLM calls).
- Empty-decomposition backstop: valid-but-empty claim JSON falls back to deterministic sentence splitting (still NLI-verified downstream).
- Batch total-failure raises into retry + budgeted per-claim fallback (never all-NEUTRAL poison rows).
- Local-inference economy: task-sized output caps, single-flight local LLM semaphore, quantized KV cache (`-ctk/-ctv q8_0`), empty-rewrite and futile-regeneration short-circuits.

---

## Frontend Architecture (`apps/web`)

1. **Split-Screen Telemetry Workbench (`PlaygroundPage.jsx`)**:
   - **Left Control Panel**: Fixed width (`md:w-[320px] lg:w-[360px] xl:w-[400px]`), scrollable input container, docked "Run Analysis" button.
   - **Right Telemetry Panel**: Flexible width (`flex-1 min-w-0`), independent scroll container.
2. **Executive Telemetry HUD (`PipelineTelemetryHUD.jsx`)**:
   - 4-stage live architecture tracker: `Hybrid Retrieval` → `Cross-RRF Fusion` → `Grounded Synthesis` → `Claim Entailment`.
   - Real-time status indicators (emerald check, pulsing cyan spinner, slate pending).
3. **Classy Markdown & GFM Table Renderer (`FormattedAnswer.jsx`)**:
   - Integrated with `react-markdown` and `remark-gfm`.
   - Custom typography for headers, bold tokens, glowing cyan list markers, and responsive dark glass data tables (`<table>`, `<thead>`, `<tbody>`).
4. **Dual-Channel Active Polling Resiliency**:
   - Employs Server-Sent Events (SSE) for sub-second event streaming.
   - Concurrently runs a 2-second background status polling watcher to guarantee immediate completion transitions even if browser SSE connections buffer.
5. **Open Knowledge JSON-LD Audit Dossier**:
   - One-click export button generating schema-compliant JSON-LD audit packages containing full cryptographic provenance, verified claims, and reliability scores.

---

## MongoDB Collections

| Collection       | Purpose                                                 |
|------------------|---------------------------------------------------------|
| `users`          | Accounts, hashed passwords (bcrypt 12 rounds)           |
| `knowledge_bases`| Multi-tenant KB metadata and user ownership             |
| `documents`      | Document metadata, cryptographic SHA-256 provenance     |
| `document_chunks`| Ingested text segments with positional metadata         |
| `analyses`       | Analysis runs, queries, status, reliability diagnostics |
| `claims`         | Propositional claims decomposed and verified per run    |
| `evidence`       | Retrieved chunks and web grounding citations            |
| `recovery_runs`  | History of adaptive self-healing actions                |
| `trace_events`   | Persistent audit log of all pipeline events             |
| `experiments`    | Evaluation experiment datasets and benchmark results    |
| `feedback`       | User feedback on synthesized answers                    |
