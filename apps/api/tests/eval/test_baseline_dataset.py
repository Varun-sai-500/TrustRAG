"""
Dataset validation for the frozen Phase-0 baseline set.

Guards the eval foundation: schema, class balance, unique ids, and — most
importantly — that every gold snippet actually occurs in its referenced
fixture corpus file (so the dataset can never silently rot).
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "datasets" / "baseline_v1.jsonl"
CORPUS_DIR = EVAL_DIR / "fixtures" / "corpus"

ALLOWED_CLASSES = {"factual", "temporal", "conflicting", "missing_evidence", "adversarial"}
ALLOWED_OUTCOMES = {"completed", "abstained"}
REQUIRED_KEYS = {
    "id",
    "query",
    "query_class",
    "gold_evidence",
    "gold_answer_keywords",
    "expected_outcome",
    "notes",
}


def load_dataset() -> list[dict]:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{DATASET_PATH}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def test_dataset_has_25_queries_with_unique_ids():
    rows = load_dataset()
    assert len(rows) == 25, f"baseline_v1 must stay frozen at 25 queries, got {len(rows)}"
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate query ids"


def test_dataset_schema():
    for row in load_dataset():
        missing = REQUIRED_KEYS - set(row)
        assert not missing, f"{row.get('id')}: missing keys {missing}"
        assert row["query"] and isinstance(row["query"], str)
        assert row["query_class"] in ALLOWED_CLASSES, f"{row['id']}: bad class"
        assert row["expected_outcome"] in ALLOWED_OUTCOMES, f"{row['id']}: bad outcome"
        assert isinstance(row["gold_evidence"], list)
        assert isinstance(row["gold_answer_keywords"], list)
        for ev in row["gold_evidence"]:
            assert set(ev) == {"document", "snippet"}, f"{row['id']}: bad evidence entry {ev}"


def test_dataset_class_balance():
    rows = load_dataset()
    adversarial = [r for r in rows if r["query_class"] == "adversarial"]
    standard = [r for r in rows if r["query_class"] != "adversarial"]
    assert len(standard) >= 20, f"need ≥20 standard queries, got {len(standard)}"
    assert len(adversarial) >= 5, f"need ≥5 adversarial queries, got {len(adversarial)}"


def test_missing_evidence_queries_expect_abstention():
    for row in load_dataset():
        if row["query_class"] == "missing_evidence":
            assert row["gold_evidence"] == [], f"{row['id']}: missing_evidence must have no gold"
            assert row["expected_outcome"] == "abstained", f"{row['id']}: must expect abstention"
        elif row["query_class"] != "adversarial" or row["id"] != "q024":
            # Every query except the no-coverage ones must pin at least one gold snippet
            if row["id"] != "q024":
                assert row["gold_evidence"], f"{row['id']}: needs ≥1 gold evidence entry"


def test_gold_snippets_occur_in_referenced_fixtures():
    """Every gold snippet must occur verbatim (case-insensitive) in its fixture file."""
    for row in load_dataset():
        for ev in row["gold_evidence"]:
            fixture = CORPUS_DIR / ev["document"]
            assert fixture.is_file(), f"{row['id']}: fixture missing: {ev['document']}"
            corpus_text = fixture.read_text(encoding="utf-8").lower()
            assert ev["snippet"].lower() in corpus_text, (
                f"{row['id']}: snippet not found in {ev['document']}: {ev['snippet']!r}"
            )
