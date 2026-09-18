"""
TRUSTRAG — Reranking engine.

Uses CrossEncoder from sentence-transformers to re-evaluate the relevance
of candidates before slicing generation context.

Implements early termination strategies:
- Approximate reranking with early exit for high-confidence results
- Batch processing with progressive scoring
- Adaptive top-k based on score distribution
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import get_model_config
from app.core.logging import get_logger
from app.core.model_registry import get_reranker

logger = get_logger(__name__)

# Early termination configuration
EARLY_TERMINATION_CONFIDENCE = 0.85  # Score threshold for early exit
EARLY_TERMINATION_MIN_BATCH = 16  # Minimum candidates before early termination check
SCORE_GAP_THRESHOLD = 0.15  # Gap between top-1 and top-k for early exit


def _rerank_sync(
    query: str, chunks: list[dict[str, Any]], max_context_override: int | None = None
) -> list[dict[str, Any]]:
    """
    Synchronous reranking implementation (CPU-bound).

    Implements early termination:
    - Progressive batch scoring with confidence checks
    - Early exit when top results exceed confidence threshold
    - Score gap analysis to skip low-value candidates
    """
    cfg = get_model_config()
    max_context = (
        max_context_override if max_context_override is not None else cfg.max_context_chunks
    )

    if not chunks:
        return []

    # Check if reranking is enabled
    if not cfg.reranker_enabled:
        logger.debug("Reranker disabled, returning candidates list directly", limit=max_context)
        # Adaptive Top-K: If top chunks are confident, bound to top 4 to save model context load
        if len(chunks) > 3 and chunks[0].get("dense_score", 0.0) >= 0.78:
            return chunks[: min(max_context, 4)]
        return chunks[:max_context]

    # Bound scoring depth (models.yaml: reranker.top_k). The floor at
    # fusion_top_k is load-bearing: capping below the fused width would
    # discard candidates before scoring, defeating reranking entirely.
    depth_cap = max(cfg.reranker_top_k, cfg.fusion_top_k, max_context)
    candidates = chunks[:depth_cap]

    try:
        model = get_reranker()
        if model is None:
            logger.warning("Reranker model factory returned None, skipping rerank")
            if len(chunks) > 3 and chunks[0].get("dense_score", 0.0) >= 0.78:
                return chunks[: min(max_context, 4)]
            return chunks[:max_context]

        logger.info(
            "Running cross-encoder reranking", model=cfg.reranker_model, count=len(candidates)
        )

        # Build query-document input pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Progressive batch scoring with early termination
        # Process in batches and check for early exit conditions
        batch_size = min(32, len(pairs))  # Optimal batch size for CrossEncoder
        all_scores: list[float] = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            batch_scores = model.predict(batch_pairs)
            all_scores.extend(batch_scores)

            # Early termination check after processing enough candidates
            processed = i + len(batch_scores)
            if processed >= EARLY_TERMINATION_MIN_BATCH:
                # Check if top result is confidently better than rest
                if len(all_scores) >= 3:
                    top_score = max(all_scores)
                    # Find second best among processed
                    sorted_scores = sorted(all_scores, reverse=True)
                    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
                    score_gap = top_score - second_best

                    # Early exit if top result is very confident and well separated
                    if (
                        top_score >= EARLY_TERMINATION_CONFIDENCE
                        and score_gap >= SCORE_GAP_THRESHOLD
                    ):
                        logger.info(
                            "Early termination: high confidence top result",
                            top_score=top_score,
                            score_gap=score_gap,
                            processed=processed,
                            total=len(pairs),
                        )
                        # Pad remaining scores with 0.0
                        remaining = len(pairs) - processed
                        all_scores.extend([0.0] * remaining)
                        break

        # Update scores inside chunks
        for i, score in enumerate(all_scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort descending by rerank score — on a copy, never the caller's list.
        ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Adaptive Top-K: If top chunks are confident, bound to top 4
        effective_limit = (
            min(max_context, 4)
            if (len(ranked) > 3 and ranked[0].get("rerank_score", 0.0) >= 0.80)
            else max_context
        )
        sliced = ranked[:effective_limit]
        logger.debug(
            "Reranking completed",
            top_score=sliced[0]["rerank_score"] if sliced else 0.0,
            count=len(sliced),
        )
        return sliced

    except Exception as exc:
        logger.error("Reranking execution failed, falling back to RRF rankings", error=str(exc))
        return chunks[:max_context]


async def rerank_candidate_chunks(
    query: str, chunks: list[dict[str, Any]], max_context_override: int | None = None
) -> list[dict[str, Any]]:
    """
    Rerank candidate chunks using the CrossEncoder model configured in models.yaml.
    Runs CPU-bound CrossEncoder.predict in a thread pool to avoid blocking the event loop.

    If reranking is disabled or candidates list is empty, returns original candidates list
    sliced by maximum context limits.
    """
    return await asyncio.to_thread(_rerank_sync, query, chunks, max_context_override)
