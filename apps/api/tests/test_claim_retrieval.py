"""
Unit tests for targeted per-claim evidence retrieval (Phase 5).

RED: retrieve_evidence_for_claim does not exist; execute_claim_verification
takes no kb_id; dead reliability weights still sit in models.yaml.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from bson import ObjectId

from app.verification.verifier import execute_claim_verification, retrieve_evidence_for_claim

ANALYSIS_ID = "64ee39d09c6292376e191983"


def _fused_chunk(doc_suffix: str, idx: int, text: str) -> dict:
    return {
        "document_id": f"64ee39d09c6292376e19198{doc_suffix}",
        "chunk_index": idx,
        "text": text,
        "filename": "policy.txt",
        "page": 1,
        "dense_score": 0.5,
        "rrf_score": 0.02,
    }


def _mock_collections():
    """One mock serving CLAIMS + EVIDENCE collections: nothing pre-exists."""
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_collection.insert_many = AsyncMock(
        return_value=MagicMock(inserted_ids=[ObjectId("64ee39d09c6292376e191990")])
    )
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=ObjectId("64ee39d09c6292376e191991"))
    )
    return mock_collection


def _no_fused_two_step(neutral_claims: list[str]):
    """Patch stack: fused off, two-step decomposes to neutral_claims, batch all-NEUTRAL."""
    batch_map = {
        i + 1: {"verdict": "NEUTRAL", "supporting_segments": [], "explanation": "Missing"}
        for i in range(len(neutral_claims))
    }
    return (
        patch("app.verification.verifier.fused_decompose_verify", new=AsyncMock(return_value=None)),
        patch(
            "app.verification.verifier.decompose_answer_to_claims",
            new=AsyncMock(return_value=list(neutral_claims)),
        ),
        patch(
            "app.verification.verifier.batch_verify_claims_nli",
            new=AsyncMock(return_value=batch_map),
        ),
    )


# ── retrieve_evidence_for_claim ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claim_retrieval_drops_seen_chunks_and_caps_top_k():
    seen = {("64ee39d09c6292376e19198A", 0, "already seen text")}
    hybrid = [
        _fused_chunk("A", 0, "already seen text"),
        _fused_chunk("B", 3, "fresh evidence text one"),
        _fused_chunk("C", 4, "fresh evidence text two"),
    ]
    with patch(
        "app.retrieval.retriever.retrieve_hybrid_chunks", new=AsyncMock(return_value=hybrid)
    ):
        fresh = await retrieve_evidence_for_claim("some claim", "kb1", seen, top_k=5)
    assert [c["text"] for c in fresh] == ["fresh evidence text one", "fresh evidence text two"]


@pytest.mark.asyncio
async def test_claim_retrieval_returns_empty_on_outage():
    with patch(
        "app.retrieval.retriever.retrieve_hybrid_chunks",
        new=AsyncMock(side_effect=Exception("Qdrant down")),
    ):
        assert await retrieve_evidence_for_claim("some claim", "kb1", set()) == []


# ── execute-level flip ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_neutral_claim_flips_supported_with_new_evidence_linkage():
    p1, p2, p3 = _no_fused_two_step(["Refunds are fast."])
    fresh = _fused_chunk("B", 3, "Refunds are processed within 5 business days.")
    reverify = {
        "verdict": "SUPPORTED",
        "supporting_segments": [1],
        "explanation": "Targeted retrieval found support.",
    }
    with (
        p1,
        p2,
        p3,
        patch(
            "app.retrieval.retriever.retrieve_hybrid_chunks", new=AsyncMock(return_value=[fresh])
        ),
        patch("app.verification.verifier.verify_claim_nli", new=AsyncMock(return_value=reverify)),
        patch(
            "app.verification.integrity.audit_evidence_integrity",
            new=AsyncMock(
                side_effect=lambda chunks: [{**c, "integrity_status": "VERIFIED"} for c in chunks]
            ),
        ),
        patch("app.verification.verifier.get_collection", return_value=_mock_collections()),
    ):
        claims = await execute_claim_verification(
            analysis_id_str=ANALYSIS_ID,
            answer="Refunds are fast.",
            chunks=[_fused_chunk("A", 0, "Unrelated shipping text here.")],
            evidence_ids=[ObjectId("64ee39d09c6292376e191985")],
            kb_id_str="kb1",
        )
    assert len(claims) == 1
    assert claims[0]["state"] == "SUPPORTED"
    # Linked to the NEWLY persisted evidence, not the original unrelated one.
    assert claims[0]["evidence_ids"] == [ObjectId("64ee39d09c6292376e191990")]
    assert "targeted" in claims[0]["explanation"].lower()


@pytest.mark.asyncio
async def test_claim_retrieval_budget_caps_hybrid_calls():
    p1, p2, p3 = _no_fused_two_step([f"Claim number {i}." for i in range(5)])
    hybrid_mock = AsyncMock(return_value=[])
    with (
        p1,
        p2,
        p3,
        patch("app.retrieval.retriever.retrieve_hybrid_chunks", new=hybrid_mock),
        patch("app.verification.verifier.get_collection", return_value=_mock_collections()),
    ):
        claims = await execute_claim_verification(
            analysis_id_str=ANALYSIS_ID,
            answer="Five neutral claims here.",
            chunks=[_fused_chunk("A", 0, "Some unrelated context text.")],
            evidence_ids=[ObjectId("64ee39d09c6292376e191985")],
            kb_id_str="kb1",
        )
    assert len(claims) == 5
    assert all(c["state"] == "NEUTRAL" for c in claims)
    # models.yaml: cost_controls.max_claim_retrievals == 3
    assert hybrid_mock.await_count == 3


@pytest.mark.asyncio
async def test_contradicted_claims_are_never_re_retrieved():
    p1 = patch("app.verification.verifier.fused_decompose_verify", new=AsyncMock(return_value=None))
    p2 = patch(
        "app.verification.verifier.decompose_answer_to_claims",
        new=AsyncMock(return_value=["Bad claim here."]),
    )
    p3 = patch(
        "app.verification.verifier.batch_verify_claims_nli",
        new=AsyncMock(
            return_value={
                1: {"verdict": "CONTRADICTED", "supporting_segments": [], "explanation": "No"}
            }
        ),
    )
    hybrid_mock = AsyncMock(return_value=[])
    with (
        p1,
        p2,
        p3,
        patch("app.retrieval.retriever.retrieve_hybrid_chunks", new=hybrid_mock),
        patch("app.verification.verifier.get_collection", return_value=_mock_collections()),
    ):
        claims = await execute_claim_verification(
            analysis_id_str=ANALYSIS_ID,
            answer="Bad claim here.",
            chunks=[_fused_chunk("A", 0, "Some unrelated context text.")],
            evidence_ids=[ObjectId("64ee39d09c6292376e191985")],
            kb_id_str="kb1",
        )
    assert claims[0]["state"] == "CONTRADICTED"
    hybrid_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_kb_id_skips_claim_retrieval_entirely():
    p1, p2, p3 = _no_fused_two_step(["Refunds are fast."])
    hybrid_mock = AsyncMock(return_value=[])
    with (
        p1,
        p2,
        p3,
        patch("app.retrieval.retriever.retrieve_hybrid_chunks", new=hybrid_mock),
        patch("app.verification.verifier.get_collection", return_value=_mock_collections()),
    ):
        claims = await execute_claim_verification(
            analysis_id_str=ANALYSIS_ID,
            answer="Refunds are fast.",
            chunks=[_fused_chunk("A", 0, "Unrelated shipping text here.")],
            evidence_ids=[ObjectId("64ee39d09c6292376e191985")],
        )
    assert claims[0]["state"] == "NEUTRAL"
    hybrid_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_inline_answer_citations_union_into_evidence_ids():
    """Phase-4 deferral: [Segment N] markers surviving in claim text link evidence."""
    p1 = patch("app.verification.verifier.fused_decompose_verify", new=AsyncMock(return_value=None))
    p2 = patch(
        "app.verification.verifier.decompose_answer_to_claims",
        new=AsyncMock(return_value=["Refunds are fast [Segment 1]."]),
    )
    p3 = patch(
        "app.verification.verifier.batch_verify_claims_nli",
        new=AsyncMock(
            return_value={
                1: {"verdict": "SUPPORTED", "supporting_segments": [], "explanation": "Ok"}
            }
        ),
    )
    evidence_ids = [ObjectId("64ee39d09c6292376e191985")]
    with (
        p1,
        p2,
        p3,
        patch("app.verification.verifier.get_collection", return_value=_mock_collections()),
    ):
        claims = await execute_claim_verification(
            analysis_id_str=ANALYSIS_ID,
            answer="Refunds are fast [Segment 1].",
            chunks=[_fused_chunk("A", 0, "Refunds are fast, processed quickly.")],
            evidence_ids=evidence_ids,
        )
    assert claims[0]["evidence_ids"] == evidence_ids


def test_dead_reliability_weights_removed_from_models_yaml():
    with open("config/models.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for dead_key in (
        "citation_correctness_weight",
        "evidence_coverage_weight",
        "source_integrity_weight",
    ):
        assert dead_key not in data.get("reliability", {}), f"dead config: {dead_key}"
