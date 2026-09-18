"""
TRUSTRAG — deterministic query router + fan-out merge.

Routes every analysis query BEFORE retrieval (no LLM anywhere on this path):

- SIMPLE:     today's single hybrid call, byte-identical kwargs.
- TEMPORAL:   single call + explicit reference_time when the query names a year
              ("in 2025" → 2025-07-01, mid-year point-in-time for effective
              range filtering); otherwise the caller's now().
- COMPARISON: "A vs B" → two parallel retrievals, merged by RRF score.
- COMPLEX:    multi-"?" → deterministic split per question, capped at
              max_sub_queries. Unsplit-table input falls back to SIMPLE, so the
              router can never do worse than today's path — only equal or wider.

Fan-out budget: at most max_sub_queries hybrid calls per retrieval round
(<=2x base for the 2-sub-query comparison case). Partial branch outage degrades
to the surviving branches; total outage still raises RetrievalOutageError so
callers never mistake it for "no evidence".
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.core.exceptions import RetrievalOutageError
from app.core.logging import get_logger
from app.retrieval.retriever import retrieve_hybrid_chunks

logger = get_logger(__name__)


class QueryRoute(StrEnum):
    """Router outcome classes."""

    SIMPLE = "simple"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class RoutedQuery:
    """Router decision: which retrieval shape to run."""

    route: QueryRoute
    sub_queries: list[str] = field(default_factory=list)
    reference_time: datetime | None = None


_COMPARISON_INTENT_RE = re.compile(
    r"\b(vs\.?|versus|compar\w*|differences?\s+between)\b", re.IGNORECASE
)
_COMPARISON_SPLIT_RES = (
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
)
_DIFFERENCE_BETWEEN_RE = re.compile(r"differences?\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE)
_COMPARE_AND_RE = re.compile(r"^\s*compare\s+(.+?)\s+and\s+(.+?)\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_TEMPORAL_WORDS_RE = re.compile(
    r"\b(currently|right now|as of|still valid|effective|outdated|latest|current|now)\b",
    re.IGNORECASE,
)
# Fragments below this are punctuation remnants from splitting ("?", "&").
# Kept deliberately low: short entity fragments ("Pro", "Team plan") are valid
# retrieval strings, and an unsplittable query falls back to SIMPLE anyway.
_MIN_SUB_QUERY_CHARS = 2


def _clean_part(text: str) -> str:
    """Normalize a split fragment into a retrieval string."""
    return text.strip().strip("?.!").strip()


def _split_comparison(query: str) -> list[str] | None:
    """Split "A vs B" style queries into [A, B]; None when not splittable."""
    match = _DIFFERENCE_BETWEEN_RE.search(query)
    if match:
        parts = [match.group(1), match.group(2)]
    else:
        match = _COMPARE_AND_RE.search(query)
        if match:
            parts = [match.group(1), match.group(2)]
        else:
            parts = None
            for splitter in _COMPARISON_SPLIT_RES:
                if splitter.search(query):
                    parts = splitter.split(query)
                    break
    if not parts:
        return None
    cleaned = [_clean_part(p) for p in parts]
    cleaned = [p for p in cleaned if len(p) >= _MIN_SUB_QUERY_CHARS]
    return cleaned if len(cleaned) >= 2 else None


def _split_questions(query: str, max_sub_queries: int) -> list[str] | None:
    """Split multi-"?" queries per question; None when there is only one."""
    parts = [_clean_part(p) for p in query.split("?")]
    parts = [p for p in parts if len(p) >= _MIN_SUB_QUERY_CHARS]
    if len(parts) < 2:
        return None
    capped = parts[:max_sub_queries]
    # Same floor as comparisons: a ceiling below 2 means no fan-out.
    return capped if len(capped) >= 2 else None


def _reference_time_for_query(query: str) -> datetime | None:
    """Explicit year in query → mid-year point-in-time; else None (caller now)."""
    match = _YEAR_RE.search(query)
    if not match:
        return None
    return datetime(int(match.group(1)), 7, 1, tzinfo=UTC)


def route_query(query: str, max_sub_queries: int = 3) -> RoutedQuery:
    """Classify a query and produce its retrieval shape.

    First match wins: COMPARISON → TEMPORAL → COMPLEX → SIMPLE.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return RoutedQuery(route=QueryRoute.SIMPLE, sub_queries=[query])

    if _COMPARISON_INTENT_RE.search(cleaned_query):
        split = _split_comparison(cleaned_query)
        if split is not None:
            capped = split[:max_sub_queries]
            # A fan-out ceiling below 2 is meaningless for a two-sided
            # comparison — run the full query instead of half of it.
            if len(capped) < 2:
                return RoutedQuery(route=QueryRoute.SIMPLE, sub_queries=[query])
            return RoutedQuery(route=QueryRoute.COMPARISON, sub_queries=capped)
        return RoutedQuery(route=QueryRoute.SIMPLE, sub_queries=[query])

    if _YEAR_RE.search(cleaned_query) or _TEMPORAL_WORDS_RE.search(cleaned_query):
        return RoutedQuery(
            route=QueryRoute.TEMPORAL,
            sub_queries=[query],
            reference_time=_reference_time_for_query(cleaned_query),
        )

    split = _split_questions(cleaned_query, max_sub_queries)
    if split is not None:
        return RoutedQuery(route=QueryRoute.COMPLEX, sub_queries=split)

    return RoutedQuery(route=QueryRoute.SIMPLE, sub_queries=[query])


def merge_fanout_results(results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-sub-query fused lists: dedup by chunk id keeping best RRF, sort desc.

    Deterministic (stable sort); empty input → [].
    """
    merged: dict[str, dict[str, Any]] = {}
    for result_list in results:
        for chunk in result_list:
            key = str(chunk.get("id"))
            current = merged.get(key)
            score = float(chunk.get("rrf_score") or 0.0)
            if current is None or score > float(current.get("rrf_score") or 0.0):
                merged[key] = chunk
    return sorted(merged.values(), key=lambda c: float(c.get("rrf_score") or 0.0), reverse=True)


async def fanout_retrieve(
    sub_queries: list[str], retrieve_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run one hybrid retrieval per sub-query concurrently and merge.

    A failed branch degrades to the surviving branches; every branch failing
    raises RetrievalOutageError (never silent "no evidence").
    """
    if not sub_queries:
        return []
    branch_results = await asyncio.gather(
        *(retrieve_hybrid_chunks(query=sub, **retrieve_kwargs) for sub in sub_queries),
        return_exceptions=True,
    )
    succeeded: list[list[dict[str, Any]]] = []
    failed = 0
    for result in branch_results:
        if isinstance(result, Exception):
            failed += 1
            logger.warning("Fan-out retrieval branch failed", error=str(result))
        else:
            succeeded.append(result)
    if not succeeded and failed:
        raise RetrievalOutageError("Fan-out retrieval: all sub-query branches failed")
    return merge_fanout_results(succeeded)
