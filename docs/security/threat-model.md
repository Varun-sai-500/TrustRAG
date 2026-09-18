# TRUSTRAG — Threat Model

**Version:** 1.1 | **Phase:** RAG quality 0–8  
**Reviewer:** Engineering

> This threat model covers MVP scope. It will be updated as features are added.
> "Complete protection" is not claimed for any threat.

---

## Asset Inventory

| Asset | Classification | Notes |
|-------|---------------|-------|
| User credentials (passwords) | Critical | bcrypt hashed; never stored plaintext |
| JWT tokens | Critical | Short-lived; HS256 |
| GEMINI_API_KEY | Critical | Server-side only; env var |
| MongoDB Atlas URI | Critical | Server-side only; env var |
| Qdrant API key | High | Server-side only; env var |
| Knowledge base documents | High | User-uploaded; content-hashed |
| Analysis results | High | User data |
| Execution traces | Medium | May contain query/evidence fragments |
| System configuration | Low | models.yaml — no secrets |
| OCR models (`~/.onnx`) | Medium | First-use download supply chain (PyPI); verify package pins in `uv.lock` |
| KB snapshots | High | Full data copies; same ownership checks as live KBs |
| Router/claim-retrieval budgets | Low | `max_sub_queries: 3`, `max_claim_retrievals: 3` bound fan-out cost |

---

## Threat Table

### T-01: Prompt Injection via Retrieved Documents

| Field | Value |
|-------|-------|
| **Threat** | Malicious content in uploaded documents that modifies LLM behavior |
| **Attack Vector** | User uploads a PDF containing instruction overrides |
| **Impact** | LLM follows attacker instructions, leaks data, or modifies behavior |
| **Likelihood** | Medium (requires document upload access) |
| **Mitigation** | Prompt architecture explicitly separates SYSTEM / USER QUERY / UNTRUSTED EVIDENCE. Retrieved content is labeled and never auto-promoted to system instruction. |
| **Residual Risk** | Low-Medium. Instruction separation is not perfectly robust against all LLM models. |

### T-02: Indirect Prompt Injection (Cross-User)

| Field | Value |
|-------|-------|
| **Threat** | Knowledge base shared between users; attacker poisons shared documents |
| **Attack Vector** | Attacker uploads documents with hidden instructions; other users retrieve them |
| **Impact** | LLM follows attacker instructions when answering other users' queries |
| **Likelihood** | Low-Medium (requires shared KB access) |
| **Mitigation** | MVP uses per-user knowledge bases. Authorization enforced server-side. Evidence integrity analysis flags suspicious content. |
| **Residual Risk** | Medium. LLM instruction separation is defense-in-depth, not a guarantee. |

### T-03: IDOR — Cross-User Resource Access

| Field | Value |
|-------|-------|
| **Threat** | User A accesses User B's analyses, documents, or knowledge bases via ID enumeration |
| **Attack Vector** | Modify ObjectId in API requests |
| **Impact** | Data leakage, privacy violation |
| **Likelihood** | Medium |
| **Mitigation** | Every resource access verifies `user_id` ownership server-side. MongoDB queries always include `user_id` filter. |
| **Residual Risk** | Low (if authorization is consistently applied) |

### T-04: Denial of Wallet / Local Exhaustion via AI Cost Explosion

| Field | Value |
|-------|-------|
| **Threat** | Attacker sends large/many queries triggering excessive cloud API calls or local compute exhaustion |
| **Attack Vector** | High-frequency requests, oversized documents, adversarial queries (multi-`?` fans out ≤3 retrievals; crafted NEUTRAL-heavy answers trigger +3 claim retrievals; scanned PDFs burn OCR compute) |
| **Impact** | Unexpected API costs; local CPU saturation |
| **Likelihood** | Medium |
| **Mitigation** | Per-client rate limits (analyses/auth/upload/url-ingest). Max input token limit enforced before LLM calls. Max recovery attempts bounded. Max context chunks limited. File size limit on uploads. Router fan-out ceiling (`max_sub_queries: 3`), claim-retrieval budget (`max_claim_retrievals: 3`), OCR density gate + fail-open. Gemini costs apply only when a Gemini provider is selected (default stack is local, zero marginal cost). |
| **Residual Risk** | Low-Medium. Free-tier Gemini has built-in rate limits as an additional safety net. |

### T-05: Credential / Secret Leakage

| Field | Value |
|-------|-------|
| **Threat** | API keys or credentials committed to Git or leaked in logs/responses |
| **Attack Vector** | Accidental `.env` commit; verbose error responses; log injection |
| **Impact** | Full credential compromise |
| **Likelihood** | Low (with proper controls) |
| **Mitigation** | `.gitignore` covers `.env`. CI secret scan job. Structured logs scrub sensitive keys. Exception handlers never send internal detail to clients. |
| **Residual Risk** | Low |

### T-06: Malicious Document Upload

| Field | Value |
|-------|-------|
| **Threat** | Malformed or adversarial files that crash the parser or consume excessive resources |
| **Attack Vector** | ZIP bomb inside PDF, excessively large file, malformed encoding |
| **Impact** | DoS, memory exhaustion |
| **Likelihood** | Low-Medium |
| **Mitigation** | File size limit enforced before parsing. Supported format allowlist (8 formats). Magic-bytes validation + per-format decompression-bomb ratios. OCR density gate with fail-open (engine failure keeps native text, never kills ingest). Parser runs in bounded context. MaliciousDocumentError exception type for flagging. |
| **Residual Risk** | Medium. PyMuPDF parsing cannot guarantee safety against all malformed inputs. |

### T-07: JWT Token Attacks

| Field | Value |
|-------|-------|
| **Threat** | Token forgery, replay, or brute-force of HS256 secret |
| **Attack Vector** | Weak JWT_SECRET; token interception |
| **Impact** | Authentication bypass |
| **Likelihood** | Low (with strong secret) |
| **Mitigation** | JWT_SECRET validation enforces minimum 32-char length. Short expiry (configurable default 60 min). HTTPS enforced in production. |
| **Residual Risk** | Low |

### T-08: Unsafe LLM Tool Use

| Field | Value |
|-------|-------|
| **Threat** | LLM generates tool calls that access unauthorized resources |
| **Attack Vector** | LLM output drives database queries or shell commands |
| **Impact** | Unauthorized data access, arbitrary code execution |
| **Likelihood** | Low (MVP has limited tool use) |
| **Mitigation** | No shell/arbitrary code execution allowed. Tool calls go through schema validation + authorization before execution. LLM output is structured and validated. |
| **Residual Risk** | Low |

### T-09: Stale/Outdated Evidence

| Field | Value |
|-------|-------|
| **Threat** | Old document versions provide contradictory or incorrect evidence |
| **Attack Vector** | User does not update knowledge base after policy changes |
| **Impact** | Incorrect answers delivered with high confidence |
| **Likelihood** | High (operational risk) |
| **Mitigation** | Evidence integrity analysis checks temporal validity (`effective_from`, `effective_until`). Source conflicts flagged. Version tracking per document. KB snapshots + rollback routes (rollback returns a NEW live id; vector-less snapshots refused with 409). |
| **Residual Risk** | Medium. Temporal analysis relies on metadata quality. |

### T-10: Snapshot/Rollback IDOR

| Field | Value |
|-------|-------|
| **Threat** | User A restores or reads User B's snapshot via ID enumeration |
| **Attack Vector** | Modify snapshot ObjectId in rollback/snapshot API requests |
| **Impact** | Data leakage across tenants; destructive restore of another user's KB |
| **Likelihood** | Medium |
| **Mitigation** | Snapshot, rollback, and restore paths all verify `user_id` ownership server-side (same `get_kb` gate as live KBs); rollback additionally rejects snapshots whose `parent_kb_id` does not match. Covered by lifecycle route tests. |
| **Residual Risk** | Low (if authorization is consistently applied) |

### T-11: OCR Supply-Chain / First-Use Stall

| Field | Value |
|-------|-------|
| **Threat** | Compromised OCR package/model, or first scanned upload stalling on a large model download |
| **Attack Vector** | Malicious `rapidocr-onnxruntime` release or model-host compromise; slow network on first scanned page |
| **Impact** | Code execution via dependency (high) / availability dip on first scan (low) |
| **Likelihood** | Low |
| **Mitigation** | `rapidocr-onnxruntime` pinned in `uv.lock` (`--locked` Docker builds); pip-audit in CI; engine import is lazy so API boots without it; OCR failures fail open to native text. Pre-warm models on deploy (ingest one scanned PDF). |
| **Residual Risk** | Low |

---

## Out of Scope (MVP)

- Network-level attacks (handled by cloud provider + HTTPS)
- Physical infrastructure attacks
- Supply chain attacks on PyPI/npm packages (pip-audit + npm audit in CI)
- Advanced persistent threat actors
