"""
TRUSTRAG — client-side BM25-style sparse vectorizer.

Generates consistent integer indices and weight values for text chunks.
Qdrant applies IDF server-side (sparse-text uses Modifier.IDF — see
app/db/qdrant.py), so client values carry only the TF side of BM25:

    index value = zone_boost * tf_sat(freq) / length_norm(doc_len)
    query value = tf_sat(freq)

with tf_sat(freq) = freq * (k1 + 1) / (freq + k1) and
length_norm(doc_len) = (1 - b) + b * (doc_len / avg_len).

Notes:
- The query side skips the length norm deliberately: it is a single
  per-vector factor, hence rank-neutral, and skipping it keeps query
  term weights clean (freq-1 terms → exactly 1.0).
- avg_len is a fixed reference (≈ default chunk size), not a live corpus
  statistic: the true corpus IDF comes from Qdrant's index statistics.
- k1 / b / avg_len come from models.yaml (retrieval.sparse_*), with
  standard BM25 defaults as fallback.
"""

from __future__ import annotations

from typing import Any

import xxhash

from app.ingestion.preprocessor import ZONE_WEIGHT_BOOSTS, lexical_analyze

VOCAB_SIZE_LIMIT = 1_000_000

# Fallbacks when models.yaml lacks the retrieval.sparse_* keys.
DEFAULT_SPARSE_K1 = 1.2
DEFAULT_SPARSE_B = 0.75
DEFAULT_SPARSE_AVG_LEN_TOKENS = 128


def tokenize(text: str) -> list[str]:
    """
    Clean, normalize, tokenize, filter stopwords, and stem words using Porter Stemmer.

    Ensures consistent morphological root alignment between document indexing and query retrieval.
    """
    return lexical_analyze(text, stem=True)


def _sparse_params() -> tuple[float, float, int]:
    """Read BM25 TF params from config, falling back to standard defaults."""
    try:
        from app.core.config import get_model_config

        cfg = get_model_config()
        return (cfg.sparse_k1, cfg.sparse_b, cfg.sparse_avg_len_tokens)
    except Exception:
        return (DEFAULT_SPARSE_K1, DEFAULT_SPARSE_B, DEFAULT_SPARSE_AVG_LEN_TOKENS)


def _tf_saturate(freq: int, k1: float) -> float:
    """BM25 term-frequency saturation: diminishing returns on repeats."""
    return freq * (k1 + 1.0) / (freq + k1)


def generate_sparse_vector(
    text: str,
    zone: str = "body",
    is_query: bool = False,
) -> dict[str, list[Any]]:
    """
    Generate sparse vector indices and values for the input text.

    Incorporates:
      - Text normalization, de-hyphenation, and contraction expansion
      - Conversational query noise filtering when is_query=True
      - Porter Stemming
      - BM25 TF saturation (k1) so repeated terms cannot dominate linearly
      - BM25 length normalization (b) on the index side, against a fixed
        reference length (true corpus IDF is applied server-side by Qdrant)
      - Document Zoning boost: terms appearing in TITLE or HEADER zones receive
        amplified weights (e.g. 2.0x for Title, 1.5x for Header)
    """
    tokens = lexical_analyze(text, stem=True, is_query=is_query)
    if not tokens:
        return {"indices": [], "values": []}

    k1, b, avg_len = _sparse_params()

    # Apply document zone weight multiplier
    zone_boost = ZONE_WEIGHT_BOOSTS.get(zone, 1.0)

    freqs: dict[int, int] = {}
    for token in tokens:
        idx = xxhash.xxh32(token.encode("utf-8")).intdigest() % VOCAB_SIZE_LIMIT
        freqs[idx] = freqs.get(idx, 0) + 1

    doc_len = len(tokens)
    # BM25 length norm; rank-neutral on the query side, so applied to index
    # vectors only. Guarded against zero/negative avg_len from misconfiguration.
    length_norm = 1.0
    if not is_query and avg_len > 0:
        length_norm = (1.0 - b) + b * (doc_len / avg_len)

    # Sort indices for predictability
    sorted_indices = sorted(freqs.keys())
    values = [
        float(zone_boost * _tf_saturate(freqs[idx], k1) / length_norm) for idx in sorted_indices
    ]

    return {"indices": sorted_indices, "values": values}
