"""Aggregate supplied item scores without selecting an evaluation population."""

from __future__ import annotations

import statistics
from typing import Any


def harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else 2 * left * right / (left + right)


def build_report(
    scores: list[dict[str, Any]], *, expected_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Report quality Q, abstention recall A, and their harmonic mean.

    All rates are percentages. An explicit complete item population containing
    both classes is required for a headline composite. Partial reports retain
    their available-item diagnostics and identify their missing items.
    """
    if not scores:
        raise ValueError("cannot report an empty set of scores")
    ids = [row["instance_id"] for row in scores]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate score IDs")
    if expected_ids is not None and len(expected_ids) != len(set(expected_ids)):
        raise ValueError("duplicate expected IDs")
    expected = set(expected_ids) if expected_ids is not None else None
    if expected is not None and set(ids) - expected:
        raise ValueError("scores contain IDs outside the specified population")
    missing = sorted(expected - set(ids)) if expected is not None else None
    complete = missing == []
    patches = [row for row in scores if row["expected_outcome"] == "patch"]
    abstentions = [row for row in scores if row["expected_outcome"] == "abstain"]
    if len(patches) + len(abstentions) != len(scores):
        raise ValueError("unknown expected outcome")
    correct_patches = [row for row in patches if row["decision_correct"]]
    q = statistics.fmean(row["overall"] for row in patches) if patches else None
    a = (100 * sum(row["decision_correct"] for row in abstentions) / len(abstentions)
         if abstentions else None)
    summary = {
        "items": len(scores), "expected_items": len(expected) if expected is not None else None,
        "complete": complete, "missing_ids": missing,
        "patch_needed": len(patches), "no_op": len(abstentions),
        "delivered_patch_quality": q, "no_op_recall": a,
        "composite_score": harmonic_mean(q, a) if complete and q is not None and a is not None else None,
        "decision_accuracy_percent": 100 * sum(row["decision_correct"] for row in scores) / len(scores),
        "patch_recall_percent": 100 * len(correct_patches) / len(patches) if patches else None,
        "p0_clean_delivery_percent": (
            100 * sum(row["decision_correct"] and row["mergeable"] for row in patches) / len(patches)
            if patches else None
        ),
        "documentation_quality_mean": (
            statistics.fmean(row["overall"] for row in correct_patches) if correct_patches else None
        ),
        "documentation_quality_median": (
            statistics.median(row["overall"] for row in correct_patches) if correct_patches else None
        ),
        "execution_failures": sum(row["status"] != "completed" for row in scores),
    }
    return {"schema_version": "dogbench-report-v1", "summary": summary, "scores": scores}
