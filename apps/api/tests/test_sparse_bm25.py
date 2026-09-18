"""
Unit tests for BM25-style sparse weighting (Phase 1).

Properties under test (all deterministic, no live services):
- TF saturation: repeated terms grow sublinearly (sat, not linear TF).
- Length normalization: the same term scores lower in a longer chunk.
- Query side: no length norm (rank-neutral), noise words filtered.
- Zone boosts remain multiplicative orderings (title > header > body).
"""

from __future__ import annotations

import pytest
import xxhash

from app.ingestion.preprocessor import stem_word
from app.ingestion.sparse_vector import (
    VOCAB_SIZE_LIMIT,
    _tf_saturate,
    generate_sparse_vector,
)

FILLER_8 = "alpha beta gamma delta epsilon zeta theta kappa"
LONG_FILLER = "alpha beta gamma delta epsilon zeta theta kappa lambda mu"


def _value_for(vec: dict, token: str) -> float:
    """Look up the value of a token's hashed index in a sparse vector."""
    idx = xxhash.xxh32(stem_word(token).encode("utf-8")).intdigest() % VOCAB_SIZE_LIMIT
    pos = vec["indices"].index(idx)
    return vec["values"][pos]


def test_tf_saturation_unit_values():
    # k1 = 1.2: sat(freq) = freq * 2.2 / (freq + 1.2)
    assert _tf_saturate(1, 1.2) == pytest.approx(1.0)
    assert _tf_saturate(2, 1.2) == pytest.approx(1.375)
    assert _tf_saturate(10, 1.2) == pytest.approx(1.9642857)
    # 5x the occurrences → far less than 5x the weight (linear TF would be 5.0)
    assert _tf_saturate(10, 1.2) / _tf_saturate(2, 1.2) == pytest.approx(1.4285714)


def test_repeated_terms_grow_sublinearly_end_to_end():
    # Same document length (10 tokens) isolates saturation from length norm.
    few = generate_sparse_vector(f"refund refund {FILLER_8}")
    many = generate_sparse_vector("refund " * 10)
    ratio = _value_for(many, "refund") / _value_for(few, "refund")
    assert ratio == pytest.approx(1.4285714)  # sat(10)/sat(2); linear TF gives 5.0


def test_longer_chunk_scores_same_term_lower():
    short = generate_sparse_vector("refund alpha")
    long = generate_sparse_vector(f"refund {LONG_FILLER}")
    assert _value_for(short, "refund") > _value_for(long, "refund")


def test_query_side_has_no_length_norm_and_filters_noise():
    # "please explain" are query-noise stopwords; single surviving freq-1 term → 1.0
    vec = generate_sparse_vector("please explain refund", is_query=True)
    assert vec["values"] == pytest.approx([1.0])
    # Repeated query term saturates but is not length-normalized
    vec2 = generate_sparse_vector("refund refund", is_query=True)
    assert vec2["values"] == pytest.approx([1.375])


def test_zone_boost_ordering_preserved():
    text = "automatic indexing"
    body = generate_sparse_vector(text, zone="body")
    header = generate_sparse_vector(text, zone="header")
    title = generate_sparse_vector(text, zone="title")
    assert _value_for(title, "automatic") > _value_for(header, "automatic")
    assert _value_for(header, "automatic") > _value_for(body, "automatic")


def test_empty_text_returns_empty_vector():
    assert generate_sparse_vector("") == {"indices": [], "values": []}
    assert generate_sparse_vector("the and or", is_query=True) == {"indices": [], "values": []}
