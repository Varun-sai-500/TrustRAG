"""
TRUSTRAG — LangGraph Agentic Adaptive Recovery Workflow.

Coordinates retrieval, generation, verification, and adaptive recovery loops
(query rewriting, expanded retrieval) when reliability thresholds fail.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any, TypedDict

from bson import ObjectId
from langgraph.graph import END, StateGraph

from app.core.config import get_model_config
from app.core.exceptions import RetrievalOutageError
from app.core.llm_utils import normalize_llm_content
from app.core.logging import get_logger
from app.core.model_registry import get_verification_model
from app.db.mongodb import Collections, get_collection
from app.generation.generator import generate_grounded_answer
from app.retrieval.reranker import rerank_candidate_chunks
from app.retrieval.retriever import _query_cache, retrieve_hybrid_chunks
from app.services.analysis_service import add_trace_event
from app.verification.integrity import audit_evidence_integrity
from app.verification.verdict import Thresholds, compute_verdict
from app.verification.verifier import execute_claim_verification, is_refusal_answer

logger = get_logger(__name__)

# H-BE-5: self-heal re-index batch size. A 10 k-chunk heal embeds + upserts in
# slices of this many chunks so peak RAM stays flat regardless of KB size.
SELF_HEAL_BATCH_SIZE = 128


# ─── LangGraph State Definition ──────────────────────────────────────────────


class AgentState(TypedDict):
    analysis_id: str
    user_id: str | None
    kb_id: str
    query: str
    current_query: str
    answer: str | None
    chunks: list[dict[str, Any]]
    evidence_ids: list[ObjectId]
    claims: list[dict[str, Any]]
    attempts: int
    verdict_status: str  # "PASS" | "FAIL"
    recovery_strategy: str | None  # "query_rewrite" | "re_retrieve" | None
    reliability_score: float | None
    diagnosis_type: (
        str | None
    )  # RETRIEVAL_FAILURE | RETRIEVAL_OUTAGE | EVIDENCE_CONFLICT | LOW_COVERAGE
    # | VERIFICATION_TIMEOUT | VERIFICATION_ERROR | RETRIEVAL_ERROR
    # | GENERATION_ERROR | None
    diagnosis_failures: list[str]
    web_search_enabled: bool
    web_search_provider: str  # "tavily" | "duckduckgo" | "both"
    llm_provider: str | None
    llm_model: str | None
    embedding_provider: str | None
    embedding_model: str | None
    # True when the answer text is reused from semantic cache; retrieval and
    # verification still rerun against the current knowledge base for auditability.
    cache_hit: bool
    # Error tracking for fallback paths
    node_errors: list[dict[str, Any]]


# ─── Standardized Error Handling ─────────────────────────────────────────────


async def _execute_with_fallback(
    state: AgentState,
    node_name: str,
    operation: callable,
    fallback_state: AgentState | None = None,
    timeout_seconds: int | None = None,
) -> AgentState:
    """
    Execute a node operation with standardized error handling and fallback.

    Args:
        state: Current agent state
        node_name: Name of the node for logging/tracing
        operation: Async callable that performs the node's work
        fallback_state: Optional state to return on failure
        timeout_seconds: Optional timeout for the operation

    Returns:
        Updated state (either from operation or fallback)
    """
    await add_trace_event(
        state["analysis_id"],
        f"{node_name}.started",
        {"message": f"Starting {node_name} node"},
    )

    try:
        if timeout_seconds:
            result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
        else:
            result = await operation()

        await add_trace_event(
            state["analysis_id"],
            f"{node_name}.completed",
            {"message": f"{node_name} completed successfully"},
        )
        return result

    except TimeoutError:
        logger.warning(f"{node_name} node timed out", timeout_seconds=timeout_seconds)
        error_info = {
            "node": node_name,
            "error_type": "TIMEOUT",
            "message": f"{node_name} exceeded {timeout_seconds}s timeout",
        }
        state["node_errors"] = [*state.get("node_errors", []), error_info]

        await add_trace_event(
            state["analysis_id"],
            f"{node_name}.timeout",
            {"message": error_info["message"]},
        )

        if fallback_state is not None:
            return fallback_state
        # Default fallback: mark as failed but allow recovery
        state["verdict_status"] = "FAIL"
        return state

    except Exception as exc:
        logger.error(f"{node_name} node failed", error=str(exc), exc_info=True)
        error_info = {
            "node": node_name,
            "error_type": type(exc).__name__,
            "message": f"{node_name} failed: {type(exc).__name__}",
        }
        state["node_errors"] = [*state.get("node_errors", []), error_info]

        await add_trace_event(
            state["analysis_id"],
            f"{node_name}.error",
            {"message": error_info["message"], "error_type": error_info["error_type"]},
        )

        if fallback_state is not None:
            return fallback_state
        # Default fallback: mark as failed but allow recovery
        state["verdict_status"] = "FAIL"
        return state


# ─── Graph Nodes ─────────────────────────────────────────────────────────────


async def retrieval_node(state: AgentState) -> AgentState:
    """Execute hybrid retrieval, evidence integrity audit, and evidence persistence."""
    cfg = get_model_config()
    retrieval_timeout = cfg.llm_timeout_seconds  # Reuse LLM timeout for retrieval

    async def _run_retrieval() -> AgentState:
        logger.info("Agent Retrieval Node starting", attempt=state["attempts"] + 1)

        # Regenerate path: evidence already sufficient, prior failure was
        # generation-side. Skip retrieval entirely (no embedding, Qdrant,
        # rerank, or web-search spend) and retry generation on saved chunks.
        if state.get("recovery_strategy") == "regenerate" and state.get("chunks"):
            await add_trace_event(
                state["analysis_id"],
                "retrieval.reused",
                {
                    "message": f"Reusing {len(state['chunks'])} saved segments — "
                    "no retrieval spend on regeneration retry",
                },
            )
            return state

        await add_trace_event(
            state["analysis_id"],
            "retrieval.started",
            {"message": f"Searching knowledge base for query: '{state['current_query']}'"},
        )

        # 1. Hybrid Retrieval
        top_k_override = None
        max_context_override = None
        if state["recovery_strategy"] == "re_retrieve":
            # Second layer (see recovery_node downgrade): only widen search when
            # evidence is actually thin — doubling on top of sufficient chunks
            # just burns embedding/rerank compute and overflows small contexts.
            if len(state.get("chunks") or []) >= cfg.max_context_chunks:
                await add_trace_event(
                    state["analysis_id"],
                    "recovery.re_retrieve_skipped",
                    {
                        "message": "Evidence already sufficient — keeping narrow "
                        "retrieval instead of doubling search"
                    },
                )
            else:
                # OPT (local-LLM load): cap widened retrieval so recovery does
                # not pay 2x Qdrant/rerank/Mongo for chunks the 8-chunk
                # generation cap throws away anyway.
                top_k_override = min(cfg.dense_top_k * 2, cfg.max_context_chunks + 16)
                max_context_override = min(cfg.max_context_chunks * 2, cfg.max_context_chunks + 4)
                logger.info(
                    "Recovery: expanded search retrieval size triggered",
                    top_k=top_k_override,
                    max_context=max_context_override,
                )

                await add_trace_event(
                    state["analysis_id"],
                    "recovery.re_retrieve",
                    {
                        "message": f"Expanding search parameters to double context "
                        f"(top_k={top_k_override})"
                    },
                )

        retrieve_kwargs: dict[str, Any] = {
            "query": state["current_query"],
            "kb_id": state["kb_id"],
            "top_k_override": top_k_override,
        }
        if state.get("embedding_provider"):
            retrieve_kwargs["embedding_provider"] = state.get("embedding_provider")
        if state.get("embedding_model"):
            retrieve_kwargs["embedding_model"] = state.get("embedding_model")

        # Deterministic query router: SIMPLE reuses today's single hybrid call
        # verbatim; TEMPORAL adds an explicit reference_time; COMPARISON and
        # COMPLEX fan out to bounded parallel retrievals merged by RRF score.
        # Everything downstream (rerank → integrity → persist) is untouched.
        from app.agent.router import fanout_retrieve, route_query

        routed = (
            route_query(state["current_query"], max_sub_queries=cfg.max_fanout_sub_queries)
            if cfg.router_enabled
            else None
        )
        if routed is not None and routed.reference_time is not None:
            retrieve_kwargs["reference_time"] = routed.reference_time
        if routed is not None and len(routed.sub_queries) > 1:
            await add_trace_event(
                state["analysis_id"],
                "retrieval.routed",
                {
                    "message": f"Query routed as {routed.route.value}: "
                    f"{len(routed.sub_queries)} parallel retrievals",
                    "route": routed.route.value,
                    "sub_queries": routed.sub_queries,
                },
            )

        try:
            if routed is None or len(routed.sub_queries) == 1:
                candidates = await retrieve_hybrid_chunks(**retrieve_kwargs)
            else:
                branch_kwargs = {k: v for k, v in retrieve_kwargs.items() if k != "query"}
                candidates = await fanout_retrieve(routed.sub_queries, branch_kwargs)
        except RetrievalOutageError as exc:
            # Infra outage (Qdrant / embedding service unreachable) — NOT
            # "no evidence". Surface it distinctly, store a clear message,
            # and exhaust recovery budget so the graph ends instead of
            # burning LLM calls on rewrites that cannot fix an outage.
            logger.error("Retrieval outage — vector search unavailable", error=str(exc))
            await add_trace_event(
                state["analysis_id"],
                "retrieval.outage",
                {"message": f"Search service unavailable: {exc}"},
            )
            state["chunks"] = []
            state["evidence_ids"] = []
            state["claims"] = []
            state["answer"] = (
                "The knowledge base search service is temporarily unavailable, "
                "so I could not search for evidence. Please wait a few minutes "
                "and try again — this is not a finding of 'no evidence'."
            )
            state["verdict_status"] = "FAIL"
            state["reliability_score"] = 0.0
            state["diagnosis_type"] = "RETRIEVAL_OUTAGE"
            state["diagnosis_failures"] = [str(exc)]
            state["attempts"] = cfg.max_recovery_attempts
            return state

        if not candidates and state.get("attempts", 0) == 0:
            from app.db.qdrant import get_collection_name, get_qdrant_client, init_kb_collection

            try:
                q_client = await get_qdrant_client()
                col_name = get_collection_name(state["kb_id"])
                col_exists = await q_client.collection_exists(col_name)
                col_info = await q_client.get_collection(col_name) if col_exists else None
                points_count = col_info.points_count if col_info else 0

                # Check if MongoDB has chunks for this KB
                chunks_coll = get_collection(Collections.DOCUMENT_CHUNKS)
                mongo_chunks_count = await chunks_coll.count_documents(
                    {"knowledge_base_id": ObjectId(state["kb_id"])}
                )

                if points_count == 0 and mongo_chunks_count > 0:
                    logger.info(
                        "Self-healing: Re-indexing chunks from MongoDB into Qdrant",
                        kb_id=state["kb_id"],
                        chunks_count=mongo_chunks_count,
                    )
                    from qdrant_client.http import models

                    from app.core.model_registry import get_embedding_model
                    from app.ingestion.pipeline import hashlib_qdrant_id
                    from app.ingestion.sparse_vector import generate_sparse_vector

                    await init_kb_collection(state["kb_id"])
                    chunks = (
                        await chunks_coll.find({"knowledge_base_id": ObjectId(state["kb_id"])})
                        .sort("chunk_index", 1)
                        .to_list(10_000)  # Support large KBs; embedded below in batches
                    )

                    doc_coll = get_collection(Collections.DOCUMENTS)
                    # Single batched lookup (was N+1 find_one per distinct
                    # document — thousands of round trips on large KBs).
                    doc_ids = list({c["document_id"] for c in chunks})
                    doc_map: dict[str, str] = {}
                    if doc_ids:
                        async for d_obj in doc_coll.find(
                            {"_id": {"$in": doc_ids}}, {"filename": 1}
                        ):
                            doc_map[str(d_obj["_id"])] = d_obj.get("filename", "document")

                    embed_model = get_embedding_model(
                        provider=state.get("embedding_provider"),
                        model=state.get("embedding_model"),
                    )
                    # H-BE-5: embed + upsert in bounded batches so a 10 k-chunk
                    # self-heal never holds all vectors/points in RAM at once.
                    # Progress events keep the trace UI honest on long heals
                    # (the feed compacts consecutive same-type events).
                    total_chunks = len(chunks)
                    for batch_start in range(0, total_chunks, SELF_HEAL_BATCH_SIZE):
                        batch = chunks[batch_start : batch_start + SELF_HEAL_BATCH_SIZE]
                        batch_texts = [
                            f"[{doc_map.get(str(c['document_id']), 'document')} | "
                            f"{c.get('zone', 'body').upper()}] {c['text']}"
                            for c in batch
                        ]
                        batch_vectors = await asyncio.to_thread(
                            embed_model.embed_documents, batch_texts
                        )
                        sync_points = []
                        for i, c in enumerate(batch):
                            doc_id_str = str(c["document_id"])
                            chunk_zone = c.get("zone", "body")
                            sparse_vec = generate_sparse_vector(batch_texts[i], zone=chunk_zone)
                            point_id = hashlib_qdrant_id(doc_id_str, c["chunk_index"])
                            payload = {
                                "document_id": doc_id_str,
                                "knowledge_base_id": state["kb_id"],
                                "user_id": str(c.get("user_id", "")),
                                "chunk_index": c["chunk_index"],
                                "page": c.get("page", 1),
                                "character_offset": c.get("character_offset", 0),
                                "zone": chunk_zone,
                                "text": c["text"],
                            }
                            sync_points.append(
                                models.PointStruct(
                                    id=point_id,
                                    vector={
                                        "": batch_vectors[i],
                                        "sparse-text": models.SparseVector(
                                            indices=sparse_vec["indices"],
                                            values=sparse_vec["values"],
                                        ),
                                    },
                                    payload=payload,
                                )
                            )
                        await q_client.upsert(collection_name=col_name, points=sync_points)
                        await add_trace_event(
                            state["analysis_id"],
                            "retrieval.self_heal_batch",
                            {
                                "message": (
                                    f"Self-heal re-indexed "
                                    f"{min(batch_start + SELF_HEAL_BATCH_SIZE, total_chunks)}"
                                    f"/{total_chunks} chunks"
                                ),
                                "completed": min(batch_start + SELF_HEAL_BATCH_SIZE, total_chunks),
                                "total": total_chunks,
                            },
                        )
                    candidates = await retrieve_hybrid_chunks(**retrieve_kwargs)
                elif points_count == 0 and mongo_chunks_count == 0:
                    logger.warning("Knowledge base collection is empty", kb_id=state["kb_id"])
                    await add_trace_event(
                        state["analysis_id"],
                        "retrieval.empty",
                        {"message": "Knowledge base has 0 indexed chunks. Upload documents first."},
                    )
                    state["answer"] = (
                        "This knowledge base has no indexed document content. "
                        "Please upload a document to this knowledge base on the "
                        "Knowledge Bases page before running an analysis."
                    )
                    state["chunks"] = []
                    state["evidence_ids"] = []
                    state["claims"] = []
                    state["verdict_status"] = "PASS"
                    state["reliability_score"] = 0.0
                    state["diagnosis_type"] = "RETRIEVAL_FAILURE"
                    state["diagnosis_failures"] = ["Knowledge base contains 0 indexed chunks"]
                    return state
            except Exception as exc:
                logger.warning("Error during collection point verification/sync", error=str(exc))

        # 2. Rerank
        top_chunks = await rerank_candidate_chunks(
            state["current_query"], candidates, max_context_override=max_context_override
        )

        # 3. Evidence Integrity Audit
        audited_chunks = await audit_evidence_integrity(top_chunks)
        verified_chunks = [c for c in audited_chunks if c.get("integrity_status") == "VERIFIED"]

        # 3b. Live Web Search Grounding via MCP (Tavily / DuckDuckGo / Both)
        if state.get("web_search_enabled"):
            search_prov = state.get("web_search_provider", "both")
            await add_trace_event(
                state["analysis_id"],
                "web_search.started",
                {"message": f"Executing live web search grounding via MCP ({search_prov.upper()})"},
            )
            try:
                from app.mcp.client import execute_mcp_tool

                tool_name = (
                    "tavily_search"
                    if search_prov == "tavily"
                    else (
                        "duckduckgo_search" if search_prov == "duckduckgo" else "hybrid_web_search"
                    )
                )
                tool_args: dict[str, Any] = {"query": state["current_query"], "max_results": 5}
                if tool_name == "hybrid_web_search":
                    tool_args["provider"] = "both"

                web_items = await execute_mcp_tool(tool_name, tool_args)
                if web_items and isinstance(web_items, list):
                    logger.info("Web search MCP returned results", count=len(web_items))
                    from app.services.search_service import sanitize_url

                    for w_idx, w in enumerate(web_items):
                        w_title = str(w.get("title") or "Web Source").strip()[:150]
                        w_url = sanitize_url(w.get("url"))
                        w_content = str(w.get("content") or "").strip()
                        if not w_content:
                            continue
                        w_chunk = {
                            "chunk_id": f"web_{w_idx}",
                            "document_id": None,
                            "filename": w_title,
                            "url": w_url,
                            "text": f"[WEB CITATION: {w_title}] {w_content}",
                            "dense_score": float(w.get("score", 0.8)),
                            "rrf_score": float(w.get("score", 0.8)),
                            "rerank_score": float(w.get("score", 0.8)),
                            "method": f"mcp_{w.get('source', search_prov)}",
                            "integrity_status": "VERIFIED",
                            "page": 1,
                        }
                        audited_chunks.append(w_chunk)
                        verified_chunks.append(w_chunk)

                    web_sources = [
                        {"title": w.get("title"), "url": w.get("url")} for w in web_items
                    ]
                    web_msg = f"Retrieved {len(web_items)} live citations via MCP"
                    await add_trace_event(
                        state["analysis_id"],
                        "web_search.completed",
                        {
                            "message": web_msg,
                            "sources": web_sources,
                        },
                    )
            except Exception as web_exc:
                logger.error("Web search MCP grounding failed", error=str(web_exc))

        # Trace log outcomes
        corrupted_count = len(audited_chunks) - len(verified_chunks)
        if corrupted_count > 0:
            await add_trace_event(
                state["analysis_id"],
                "integrity.failed",
                {"message": f"Excluded {corrupted_count} corrupted or tampered segments"},
            )

        await add_trace_event(
            state["analysis_id"],
            "retrieval.completed",
            {
                "message": f"Retrieved {len(verified_chunks)} verified segments for reasoning",
                "segments": [
                    {
                        "filename": c.get("filename") or "unknown_doc",
                        "page": c.get("page", 1),
                        "score": c.get("rerank_score") or c.get("rrf_score", 0.0),
                        "url": c.get("url"),
                    }
                    for c in verified_chunks
                ],
            },
        )

        # 4. Save evidence records in MongoDB (Batch Optimized)
        evidence_coll = get_collection(Collections.EVIDENCE)
        evidence_ids: list[ObjectId] = []
        if audited_chunks:
            evidence_docs = []
            for c in audited_chunks:
                doc_id = ObjectId(c["document_id"]) if c.get("document_id") else None
                evt_doc = {
                    "analysis_id": ObjectId(state["analysis_id"]),
                    "user_id": ObjectId(state["user_id"]) if state.get("user_id") else None,
                    "text": c["text"],
                    "document_id": doc_id,
                    "filename": c.get("filename"),
                    "url": c.get("url"),
                    "retrieval_score": c.get("dense_score", 0.0),
                    "fusion_score": c.get("rrf_score", 0.0),
                    "rerank_score": c.get("rerank_score"),
                    "method": c.get("method", "hybrid"),
                    "integrity_status": c.get("integrity_status", "CORRUPTED"),
                    "effective_from": c.get("effective_from"),
                    "effective_until": c.get("effective_until"),
                    "created_at": datetime.now(UTC),
                }
                evidence_docs.append(evt_doc)

            try:
                insert_res = await evidence_coll.insert_many(evidence_docs)
                evidence_ids = list(insert_res.inserted_ids)
            except TypeError:
                for doc in evidence_docs:
                    res = await evidence_coll.insert_one(doc)
                    evidence_ids.append(res.inserted_id)

        # Filter out Mongo IDs for verified evidence only
        verified_evidence_ids = []
        for i, c in enumerate(audited_chunks):
            if c.get("integrity_status") == "VERIFIED" and i < len(evidence_ids):
                verified_evidence_ids.append(evidence_ids[i])

        state["chunks"] = verified_chunks
        state["evidence_ids"] = verified_evidence_ids

        # PERF/SPIRAL GUARD 2026-09-06: expanded recovery retrieval widens the
        # CANDIDATE pool (top_k=40), but generation must never exceed
        # max_context_chunks. Stuffing 16-32 chunks into a 2k-context local LLM
        # overflows num_ctx and yields truncated stubs (e.g. answer "The").
        gen_cap = cfg.max_context_chunks
        if len(state["chunks"]) > gen_cap:
            dropped = len(state["chunks"]) - gen_cap
            state["chunks"] = state["chunks"][:gen_cap]
            await add_trace_event(
                state["analysis_id"],
                "retrieval.capped",
                {
                    "message": f"Capped generation context at {gen_cap} chunks "
                    f"({dropped} extra kept as evidence only)",
                },
            )
        return state

    # Fallback state for retrieval failure
    fallback_state = {
        **state,
        "chunks": [],
        "evidence_ids": [],
        "verdict_status": "FAIL",
        "diagnosis_type": "RETRIEVAL_ERROR",
        "diagnosis_failures": ["Retrieval failed due to internal error"],
    }

    return await _execute_with_fallback(
        state=state,
        node_name="retrieval",
        operation=_run_retrieval,
        fallback_state=fallback_state,
        timeout_seconds=retrieval_timeout,
    )


async def generation_node(state: AgentState) -> AgentState:
    """Generate answer grounded in retrieved context with context management."""
    cfg = get_model_config()
    generation_timeout = cfg.llm_timeout_seconds

    async def _run_generation() -> AgentState:
        logger.info("Agent Generation Node starting")

        # Semantic-cache answers are reused only after fresh retrieval has
        # persisted evidence for this run. Verification below must re-prove the
        # answer against the current knowledge base, never trust cached claims.
        if state.get("cache_hit") and state.get("answer"):
            await add_trace_event(
                state["analysis_id"],
                "generation.cache_reused",
                {"message": "Reused cached answer; verifying against fresh evidence"},
            )
            return state

        # If answer was already formulated by the 0-chunk empty KB guard, preserve it
        empty_kb_guard = state.get("diagnosis_failures") == [
            "Knowledge base contains 0 indexed chunks"
        ]
        # A retrieval outage also stores a final user-facing message in the
        # retrieval node — never overwrite it with a grounded-ABSTAIN.
        outage_guard = state.get("diagnosis_type") == "RETRIEVAL_OUTAGE"
        if state.get("answer") and not state.get("chunks") and (empty_kb_guard or outage_guard):
            return state

        # Futile-regeneration guard: on a regenerate retry the chunk set is
        # unchanged (retrieval short-circuits above), and the model already
        # refused these exact segments. Re-invoking burns a full local
        # generation (~60s on 2-3B models) for a certain repeat refusal —
        # keep the refusal and let verification close out the run.
        if (
            state.get("recovery_strategy") == "regenerate"
            and is_refusal_answer(state.get("answer"))
            and state.get("chunks")
        ):
            logger.info("Skipping futile regeneration (prior refusal on identical chunks)")
            await add_trace_event(
                state["analysis_id"],
                "generation.skipped",
                {
                    "message": "Model already abstained on these segments — "
                    "skipping repeat generation",
                },
            )
            return state

        await add_trace_event(
            state["analysis_id"],
            "generation.started",
            {"message": "Reasoning grounded answer from verified context"},
        )

        answer = await generate_grounded_answer(
            state["current_query"],
            state["chunks"],
            provider=state.get("llm_provider"),
            model=state.get("llm_model"),
        )

        state["answer"] = answer
        return state

    # Fallback state for generation failure
    fallback_state = {
        **state,
        "answer": "ABSTAIN",
        "verdict_status": "FAIL",
        "diagnosis_type": "GENERATION_ERROR",
        "diagnosis_failures": ["Generation failed due to internal error"],
    }

    return await _execute_with_fallback(
        state=state,
        node_name="generation",
        operation=_run_generation,
        fallback_state=fallback_state,
        timeout_seconds=generation_timeout,
    )


async def verification_node(state: AgentState) -> AgentState:
    """Run claims decomposition and NLI verification, and evaluate reliability thresholds."""
    logger.info("Agent Verification Node starting")

    cfg = get_model_config()
    verification_timeout = cfg.max_verification_time_seconds

    # Empty-KB fast path: the retrieval node already stored the final answer
    # for a knowledge base with zero chunks. Only take it when there is still
    # no evidence — after recovery, chunks may exist with a stale diagnosis
    # from an earlier round, and the fresh answer MUST be verified.
    if (
        state.get("diagnosis_type") in ("RETRIEVAL_FAILURE", "RETRIEVAL_OUTAGE")
        and state.get("answer")
        and not state.get("chunks")
    ):
        state["attempts"] = cfg.max_recovery_attempts
        if state.get("diagnosis_type") == "RETRIEVAL_OUTAGE":
            # Outage stays FAIL so it is never cached as an answer and never
            # mistaken for a passed analysis; attempts=max still ends the run.
            state["verdict_status"] = "FAIL"
        else:
            state["verdict_status"] = "PASS"
        return state

    answer = state["answer"]

    if not answer or answer == "ABSTAIN" or is_refusal_answer(answer):
        # Refusal-gate hit log (tuning signal: which answers skip NLI entirely).
        logger.info(
            "Refusal gate hit: skipping claim verification",
            answer_len=len(answer) if answer else 0,
            has_chunks=bool(state.get("chunks")),
        )
        state["claims"] = []
        state["reliability_score"] = None
        state["diagnosis_type"] = "RETRIEVAL_FAILURE"
        state["diagnosis_failures"] = (
            ["No relevant evidence segments were retrieved"]
            if not state["chunks"]
            else ["Retrieved segments contained insufficient information to answer the query"]
        )
        attempts = state.get("attempts", 0)
        if attempts < cfg.max_recovery_attempts:
            state["verdict_status"] = "FAIL"
            logger.info(
                "Generation abstained due to insufficient context, triggering adaptive recovery",
                attempt=attempts + 1,
                max_attempts=cfg.max_recovery_attempts,
            )
        else:
            state["verdict_status"] = "PASS"
            logger.info("Generation abstained and maximum recovery attempts reached")
        return state

    await add_trace_event(
        state["analysis_id"],
        "claims.started",
        {"message": "Decomposing answer and executing NLI verification checks"},
    )

    async def _run_verification() -> list[dict[str, Any]]:
        """Inner verification logic with timeout protection."""
        return await execute_claim_verification(
            analysis_id_str=state["analysis_id"],
            answer=answer,
            chunks=state["chunks"],
            evidence_ids=state["evidence_ids"],
            user_id_str=state.get("user_id"),
            provider=state.get("llm_provider"),
            model=state.get("llm_model"),
            attempt=state.get("attempts", 0),
            kb_id_str=state.get("kb_id"),
        )

    try:
        claims = await asyncio.wait_for(_run_verification(), timeout=verification_timeout)
    except TimeoutError:
        logger.warning(
            "Verification node timed out, triggering recovery",
            timeout_seconds=verification_timeout,
            attempt=state.get("attempts", 0) + 1,
        )
        state["claims"] = []
        state["reliability_score"] = None
        state["diagnosis_type"] = "VERIFICATION_TIMEOUT"
        state["diagnosis_failures"] = [f"Verification exceeded {verification_timeout}s timeout"]
        attempts = state.get("attempts", 0)
        if attempts < cfg.max_recovery_attempts:
            state["verdict_status"] = "FAIL"
        else:
            state["verdict_status"] = "PASS"
        return state
    except Exception as exc:
        logger.error("Verification node failed with error", error=str(exc))
        state["claims"] = []
        state["reliability_score"] = None
        state["diagnosis_type"] = "VERIFICATION_ERROR"
        state["diagnosis_failures"] = [f"Verification failed: {type(exc).__name__}"]
        attempts = state.get("attempts", 0)
        if attempts < cfg.max_recovery_attempts:
            state["verdict_status"] = "FAIL"
        else:
            state["verdict_status"] = "PASS"
        return state

    state["claims"] = claims

    total = len(claims)
    if total == 0:
        # No verifiable claims extracted — MUST NOT report TRUSTED. Mirror
        # compute_verdict(total=0) semantics: FAIL so recovery/abstain runs.
        await add_trace_event(
            state["analysis_id"],
            "claims.empty",
            {"message": "Answer produced no verifiable claims; marking for recovery"},
        )
        state["verdict_status"] = "FAIL"
        state["reliability_score"] = 0.0
        state["diagnosis_type"] = "RETRIEVAL_FAILURE"
        state["diagnosis_failures"] = ["No claims extracted for verification"]
        return state

    supported = sum(1 for c in claims if c["state"] == "SUPPORTED")
    contradicted = sum(1 for c in claims if c["state"] == "CONTRADICTED")
    neutral = sum(1 for c in claims if c["state"] == "NEUTRAL")

    await add_trace_event(
        state["analysis_id"],
        "claims.verified",
        {
            "message": f"Verified {total} atomic claims",
            "stats": {"supported": supported, "contradicted": contradicted, "neutral": neutral},
        },
    )

    # Compute unified verdict
    thresholds = Thresholds(
        minimum_evidence_coverage=cfg.minimum_evidence_coverage,
        maximum_contradiction_rate=cfg.maximum_contradiction_rate,
        abstain_below=cfg.abstain_below,
    )

    verdict = compute_verdict(
        supported=supported,
        contradicted=contradicted,
        neutral=neutral,
        total=total,
        thresholds=thresholds,
        answer=state.get("answer"),
    )

    state["verdict_status"] = verdict.verdict_status.value
    state["reliability_score"] = verdict.reliability_score
    state["diagnosis_type"] = verdict.diagnosis_type.value
    state["diagnosis_failures"] = verdict.diagnosis_failures

    logger.info(
        "Unified verdict computed",
        verdict=verdict.verdict_status.value,
        reliability_score=verdict.reliability_score,
        diagnosis_type=verdict.diagnosis_type.value,
    )

    return state


# Small models echo the instruction frame around the rewrite itself
# ("Expanded Search Query: <query>"). Searching that literally pollutes
# retrieval with junk tokens, so strip known meta-prefixes, wrapping quotes,
# and collapsed whitespace before the rewrite is used or traced.
_REWRITE_META_PREFIXES = (
    "expanded search query:",
    "rewritten query:",
    "rewritten search query:",
    "search query:",
    "expanded query:",
)


def _sanitize_rewritten_query(raw: str | None) -> str:
    text = (raw or "").strip().strip("\"'`")
    lowered = text.lower()
    for prefix in _REWRITE_META_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip().strip("\"'`")
            lowered = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if _looks_like_instruction_echo(text):
        return ""
    return text


# Small local models sometimes echo the rewrite *instructions* instead of a
# query (observed: "Expand acronyms/abbreviations to full forms and add
# synonyms. Summarize main findings..."). Searching that literally pollutes
# retrieval with junk tokens, so detect the echo and return "" so the caller
# falls back to the original query via its existing empty-rewrite path.
_ECHO_MARKERS = (
    "original_query",
    "original query",
    "missing_claims",
    "missing claims",
    "missing facts",
    "output only",
    "no markdown",
    "never reply empty",
    "<original",
    "</",
)
_ECHO_PREFIXES = (
    "expand acronyms",
    "rewrite the query",
    "your task",
    "you are a",
)


def _looks_like_instruction_echo(text: str) -> bool:
    """True when a rewrite looks like echoed prompt instructions, not a query."""
    if not text:
        return False
    lowered = text.lower()
    if any(m in lowered for m in _ECHO_MARKERS):
        return True
    if any(lowered.startswith(p) for p in _ECHO_PREFIXES):
        return True
    # Rewrites are 5-12 words; a 20+ word paragraph is echoed instructions.
    if len(text.split()) > 20:
        return True
    return False


async def recovery_node(state: AgentState) -> AgentState:
    """Determine adaptive strategy and execute recovery step (e.g. Query Rewriting)."""
    cfg = get_model_config()
    recovery_timeout = cfg.llm_timeout_seconds

    async def _run_recovery() -> AgentState:
        state["attempts"] += 1

        # Snapshot failed-claim context BEFORE clearing: the query_rewrite
        # strategy targets missing facts, but state["claims"] is reset below.
        missing_claims_snapshot = [
            c["text"] for c in state.get("claims", []) if c.get("state") != "SUPPORTED"
        ]
        # Snapshot the refused answer too: the empty-rewrite guard below needs
        # to know the model already abstained on these chunks.
        prior_answer = state.get("answer")

        # Clear prior failed/abstained answer and claims so recovery generates and verifies freshly
        state["answer"] = None
        state["claims"] = []
        state["cache_hit"] = False
        # Clear the prior round's diagnosis/verdict too — verification_node
        # branches on diagnosis_type, and a stale RETRIEVAL_FAILURE would
        # short-circuit verification of the fresh answer (skipping it entirely).
        state["diagnosis_type"] = None
        state["diagnosis_failures"] = []
        state["verdict_status"] = "FAIL"
        state["reliability_score"] = None

        # Determine recovery strategy from public config property
        priority = cfg.recovery_strategy_priority
        idx = (state["attempts"] - 1) % len(priority)
        strategy = priority[idx]

        logger.info(
            "Triggering adaptive recovery loop", attempt=state["attempts"], strategy=strategy
        )

        if strategy == "query_rewrite":
            # Use LLM to expand acronyms and terms contextually (no hardcoded map)
            # so it adapts to any knowledge base domain.
            # Invoke LLM to rewrite the query targeting the missing facts
            missing_claims = missing_claims_snapshot
            if missing_claims:
                missing_str = "\n".join(f"- {c}" for c in missing_claims)
                rewrite_prompt = f"""You are a query expansion assistant for an IR system.
The original query may contain acronyms or ambiguous terms.
Your task: rewrite the query to search for the missing factual details below.
- Expand acronyms/abbreviations to full forms
  (e.g., API → Application Programming Interface)
- Add synonyms or related terms that would help retrieval
- Keep the query focused and concise (5 to 12 words)

Output only the expanded search query string. No markdown or commentary.
Never reply empty: if unsure, return the original query with spelling corrected.

<ORIGINAL_QUERY>
{state["query"]}
</ORIGINAL_QUERY>
<MISSING_CLAIMS>
{missing_str}
</MISSING_CLAIMS>
"""
            else:
                # Query rewrite triggered because generation abstained / insufficient context
                rewrite_prompt = f"""You are a search query expansion assistant for an IR system.
The original query did not return sufficient information to answer the question.
Your task: expand the query by resolving ambiguous acronyms and terms.
- Expand any acronyms/abbreviations to their full forms
- Add synonyms or related terms that would help retrieval
- Keep the query focused and concise (5 to 12 words)

Output only the expanded search query string. No markdown or quotes.
Never reply empty: if unsure, return the original query with spelling corrected.

<ORIGINAL_QUERY>
{state["query"]}
</ORIGINAL_QUERY>
"""
            try:
                from app.core.local_llm import local_cap_kwargs

                model = get_verification_model(
                    provider=state.get("llm_provider"), model=state.get("llm_model")
                )
                # Local-RAM: a 5-12 word rewrite must not reserve 1024 output
                # tokens of KV cache. Cloud providers ignore the foreign key.
                cap = local_cap_kwargs(
                    state.get("llm_provider") or cfg.verification_provider,
                    max_tokens=128,
                )
                invoker = model.bind(**cap) if cap else model
                response = await invoker.ainvoke(rewrite_prompt)
                new_query = normalize_llm_content(response.content)
                new_query = _sanitize_rewritten_query(str(new_query))

                if not new_query or len(new_query) < 3:
                    # Small local models sometimes return an empty rewrite.
                    # An empty query would waste a full retrieval+generation
                    # round on unranked content — keep the original instead.
                    logger.warning(
                        "Query rewrite returned empty text, keeping original query",
                        original=state["query"],
                    )
                    if is_refusal_answer(prior_answer) and state.get("chunks"):
                        # ...and the model already refused these exact chunks:
                        # re-searching the identical query can only return the
                        # same context for a certain repeat refusal. Route
                        # through regenerate so retrieval short-circuits and
                        # the futile-generation guard skips the repeat call —
                        # the round then costs ~zero instead of minutes.
                        state["recovery_strategy"] = "regenerate"
                        # Restore the refusal cleared above: it arms the
                        # futile-generation guard (regenerate + refusal +
                        # unchanged chunks → skip) so the round costs ~zero.
                        state["answer"] = prior_answer
                        await add_trace_event(
                            state["analysis_id"],
                            "recovery.regenerate",
                            {
                                "message": "Empty rewrite on already-refused evidence — "
                                "reusing saved segments without new retrieval spend",
                            },
                        )
                    else:
                        state["recovery_strategy"] = None
                else:
                    logger.info(
                        "Query rewritten successfully",
                        original=state["query"],
                        rewritten=new_query,
                    )
                    state["current_query"] = new_query
                    state["recovery_strategy"] = "query_rewrite"

                    await add_trace_event(
                        state["analysis_id"],
                        "recovery.rewrite",
                        {
                            "message": "Rewriting query to target missing details",
                            "original_query": state["query"],
                            "rewritten_query": new_query,
                        },
                    )
            except Exception as exc:
                logger.error("Query rewrite failed, falling back to original query", error=str(exc))
                state["recovery_strategy"] = None

        elif strategy == "re_retrieve":
            # Load-aware downgrade (decision layer): when evidence is already
            # sufficient, the failure is generation-side — widening search only
            # burns embedding/rerank/compute on a small local model. Retry
            # generation on the saved chunks instead (retrieval_node short-
            # circuits on the "regenerate" strategy).
            if len(state.get("chunks") or []) >= cfg.max_context_chunks:
                strategy = "regenerate"
                state["recovery_strategy"] = "regenerate"
                logger.info(
                    "Recovery downgraded re_retrieve → regenerate (evidence sufficient)",
                    chunks=len(state.get("chunks") or []),
                )
                await add_trace_event(
                    state["analysis_id"],
                    "recovery.regenerate",
                    {
                        "message": "Evidence sufficient — retrying generation on "
                        "saved segments without new retrieval spend",
                    },
                )
            else:
                state["recovery_strategy"] = "re_retrieve"
                # re_retrieve executes in retrieval_node via doubled search params

        elif strategy == "regenerate":
            # Explicit yaml strategy: same cheap retry, no retrieval spend.
            state["recovery_strategy"] = "regenerate"

        else:
            state["recovery_strategy"] = None

        # Persist recovery run record in MongoDB
        run_doc = {
            "analysis_id": ObjectId(state["analysis_id"]),
            "attempt": state["attempts"],
            "strategy": strategy,
            "query_used": state["current_query"],
            "created_at": datetime.now(UTC),
        }
        await get_collection(Collections.RECOVERY_RUNS).insert_one(run_doc)

        return state

    # Fallback state for recovery failure
    fallback_state = {
        **state,
        "recovery_strategy": None,
        "verdict_status": "FAIL",
        "diagnosis_type": "RECOVERY_ERROR",
        "diagnosis_failures": ["Recovery failed due to internal error"],
    }

    return await _execute_with_fallback(
        state=state,
        node_name="recovery",
        operation=_run_recovery,
        fallback_state=fallback_state,
        timeout_seconds=recovery_timeout,
    )


# ─── Conditional Edge Router ──────────────────────────────────────────────────


def should_recover(state: AgentState) -> str:
    """Determine if recovery node should execute or terminate the graph run."""
    cfg = get_model_config()
    max_recovery = cfg.max_recovery_attempts

    if state["verdict_status"] == "PASS" or state["attempts"] >= max_recovery:
        return "end"
    return "recover"


# ─── Graph Construction ────────────────────────────────────────────────────────

# Module-level compiled graph singleton — built once, reused per request.
_compiled_graph: Any = None


def build_agent_graph() -> Any:
    """Assemble and compile LangGraph State Graph workflow (cached singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("retrieval", retrieval_node)
        builder.add_node("generation", generation_node)
        builder.add_node("verification", verification_node)
        builder.add_node("recovery", recovery_node)

        # Map edges
        builder.set_entry_point("retrieval")
        builder.add_edge("retrieval", "generation")
        builder.add_edge("generation", "verification")

        builder.add_conditional_edges(
            "verification", should_recover, {"recover": "recovery", "end": END}
        )

        builder.add_edge("recovery", "retrieval")

        _compiled_graph = builder.compile()
        logger.info("LangGraph agent graph compiled and cached")
    return _compiled_graph


# ─── Coordinator Run ──────────────────────────────────────────────────────────


async def execute_agentic_rag_flow(
    analysis_id_str: str,
    kb_id_str: str,
    query: str,
    user_id_str: str | None = None,
    web_search_enabled: bool = False,
    web_search_provider: str = "both",
    llm_provider: str | None = None,
    llm_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Compile and execute the full agent graph pipeline."""
    graph = build_agent_graph()

    initial_state: AgentState = {
        "analysis_id": analysis_id_str,
        "user_id": user_id_str,
        "kb_id": kb_id_str,
        "query": query,
        "current_query": query,
        "answer": None,
        "chunks": [],
        "evidence_ids": [],
        "claims": [],
        "attempts": 0,
        "verdict_status": "FAIL",
        "recovery_strategy": None,
        "reliability_score": None,
        "diagnosis_type": None,
        "diagnosis_failures": [],
        "web_search_enabled": web_search_enabled,
        "web_search_provider": web_search_provider,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "cache_hit": False,
        "node_errors": [],
    }

    # ── Semantic answer reuse (safe mode) ──────────────────────────────────────
    # Cache only the answer text. Retrieval and NLI still rerun against the
    # current KB so claims, evidence IDs, integrity status, and verdict always
    # belong to this analysis and cannot be stale after document deletion.
    q_vec: list[float] | None = None
    if not web_search_enabled:
        try:
            from app.core.model_registry import get_embedding_model
            from app.core.semantic_cache import check_semantic_cache

            cache_key = f"{embedding_provider or ''}:{embedding_model or ''}:{query}"
            q_vec = _query_cache.get(cache_key)
            if q_vec is None:
                emb_model = get_embedding_model(provider=embedding_provider, model=embedding_model)
                try:
                    q_vec = await emb_model.aembed_query(query)
                except Exception:
                    q_vec = await asyncio.to_thread(emb_model.embed_query, query)
                _query_cache.set(cache_key, q_vec)

            cached_resp = check_semantic_cache(
                query,
                kb_id_str,
                q_vec,
                similarity_threshold=0.94,
                embedding_model=f"{embedding_provider or ''}:{embedding_model or ''}",
            )
            if cached_resp and isinstance(cached_resp.get("answer"), str):
                initial_state["answer"] = cached_resp["answer"]
                initial_state["cache_hit"] = True
                await add_trace_event(
                    analysis_id_str,
                    "cache.hit",
                    {
                        "message": (
                            "Semantic cache matched a prior answer (similarity >= 94%). "
                            "Generation is skipped; retrieval and NLI revalidation continue."
                        ),
                        "cached_query": query,
                    },
                )
        except Exception as cache_err:
            logger.debug("Semantic cache check bypassed", error=str(cache_err))

    logger.info("Executing Agentic RAG Flow graph", analysis_id=analysis_id_str)
    try:
        final_state = await graph.ainvoke(initial_state)

        # Store in semantic cache if verified and valid
        if (
            q_vec
            and final_state.get("verdict_status") == "PASS"
            and final_state.get("answer")
            and final_state["answer"] != "ABSTAIN"
            and not web_search_enabled
        ):
            try:
                from app.core.semantic_cache import store_semantic_cache

                store_semantic_cache(
                    query=query,
                    kb_id=kb_id_str,
                    query_vector=q_vec,
                    response_data={
                        "answer": final_state["answer"],
                    },
                    embedding_model=f"{embedding_provider or ''}:{embedding_model or ''}",
                )
            except Exception as store_err:
                logger.debug("Semantic cache store skipped", error=str(store_err))

        return final_state
    finally:
        from app.core.memory import trim_memory

        await asyncio.to_thread(trim_memory)
