from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dogbench.judge import (
    criterion_blocks, numeric_score, score_predictions, validate_score,
)
from dogbench.reporting import build_report


RUBRIC = """### C1 — Document the timeout

**Priority:** P0 | **Category:** accuracy | **Scoring:** requirement

Document the default timeout.
"""
PATCH = """diff --git a/docs.md b/docs.md
--- a/docs.md
+++ b/docs.md
@@ -1 +1,2 @@
 # Configuration
+The default timeout is 30 seconds.
"""


def judgment(cid="C1", kind="requirement", **updates):
    return {
        "id": cid, "scoring_type": kind, "triggered": None, "verdict": "pass",
        "violated": None, "triggering_patch_quote": None,
        "supporting_patch_quote": "The default timeout is 30 seconds.",
        "contradicting_patch_quote": None, "missing_requirement": None,
        "reason": "The default is documented.", **updates,
    }


def outcome(instance_id="patch-task", expected="patch"):
    return {
        "instance_id": instance_id, "expected_outcome": expected,
        "rubric_markdown": RUBRIC if expected == "patch" else None,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest() if expected == "patch" else None,
    }


class JudgeTests(unittest.TestCase):
    def test_neutral_criteria_do_not_earn_points_and_p0_caps(self):
        blocks = {"C1": {"priority": "P0", "scoring_type": "requirement"}}
        criteria = [judgment(verdict="fail", supporting_patch_quote=None,
                              missing_requirement="The default is absent.")]
        for i in range(2, 12):
            blocks[f"C{i}"] = {"priority": "P3", "scoring_type": "requirement"}
            criteria.append(judgment(f"C{i}"))
        blocks["C12"] = {"priority": "P0", "scoring_type": "conditional"}
        criteria.append(judgment("C12", "conditional", triggered=False, verdict=None,
                                  supporting_patch_quote=None))
        blocks["C13"] = {"priority": "P0", "scoring_type": "deduction_only"}
        criteria.append(judgment("C13", "deduction_only", violated=False, verdict=None,
                                  supporting_patch_quote=None))
        payload = {"criteria": criteria, "summary": "One active P0 failure."}
        validate_score(payload, blocks, PATCH)
        score = numeric_score(payload, blocks)
        self.assertEqual(score["requirement_total_weight"], 11)
        self.assertEqual(score["uncapped_overall"], 90.9)
        self.assertEqual(score["overall"], 60)
        self.assertEqual(score["blocking_p0"], ["C1"])

    def test_evidence_rejects_invented_quote_and_missing_criterion(self):
        blocks = criterion_blocks(RUBRIC)
        for payload in (
            {"criteria": [judgment(supporting_patch_quote="invented")], "summary": ""},
            {"criteria": [], "summary": ""},
        ):
            with self.assertRaises(ValueError):
                validate_score(payload, blocks, PATCH)

    def test_evidence_must_come_from_added_or_post_change_lines(self):
        patch = """diff --git a/docs.md b/docs.md
--- a/docs.md
+++ b/docs.md
@@ -1,2 +1,2 @@
-Removed evidence sentence.
+Added evidence sentence.
 context evidence sentence.
"""
        blocks = criterion_blocks(RUBRIC)

        removed = {"criteria": [judgment(
            supporting_patch_quote="Removed evidence sentence.")], "summary": ""}
        with self.assertRaises(ValueError):
            validate_score(removed, blocks, patch)

        for quote in ("context evidence sentence.", "Added evidence sentence."):
            with self.subTest(quote=quote):
                payload = {"criteria": [judgment(supporting_patch_quote=quote)], "summary": ""}
                validate_score(payload, blocks, patch)

    def test_saved_hash_bound_judgment_scores_without_model_call(self):
        payload = {"criteria": [judgment()], "summary": "Pass", "original_field": {"kept": True}}
        predictions = [
            {"instance_id": "patch-task", "decision": "patch", "patch": PATCH},
            {"instance_id": "noop-task", "decision": "abstain", "patch": ""},
        ]
        outcomes = [outcome(), outcome("noop-task", "abstain")]
        saved = {"patch-task": {
            "judgment": payload,
            "patch_sha256": hashlib.sha256(PATCH.encode()).hexdigest(),
            "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        }}
        with tempfile.TemporaryDirectory() as temporary, patch("dogbench.judge.run_codex") as model:
            root = Path(temporary)
            scores = score_predictions(predictions, outcomes, None, root, saved_judgments=saved)
            model.assert_not_called()
            preserved = json.loads((root / "judgments/patch-task/judgment.json").read_text())
            self.assertEqual(preserved, payload)
            self.assertEqual([s["overall"] for s in scores], [100, 100])
            report = build_report(scores, expected_ids=["patch-task", "noop-task"])
            self.assertEqual(report["summary"]["composite_score"], 100)

    def test_saved_judgment_missing_or_hash_mismatched_never_falls_back_to_model(self):
        prediction = {"instance_id": "patch-task", "decision": "patch", "patch": PATCH}
        payload = {"criteria": [judgment()], "summary": "Pass"}
        for saved in ({}, {"patch-task": {"judgment": payload, "patch_sha256": "wrong"}}):
            with tempfile.TemporaryDirectory() as temporary, patch("dogbench.judge.run_codex") as model:
                with self.assertRaises(ValueError):
                    score_predictions([prediction], [outcome()], None, Path(temporary),
                                      saved_judgments=saved)
                model.assert_not_called()

    def test_saved_binding_distinguishes_raw_partial_and_complete_provenance(self):
        prediction = {"instance_id": "patch-task", "decision": "patch", "patch": PATCH}
        payload = {"criteria": [judgment()], "summary": "Pass", "original_metadata": {"kept": True}}
        patch_hash = hashlib.sha256(PATCH.encode()).hexdigest()
        rubric_hash = hashlib.sha256(RUBRIC.encode()).hexdigest()
        cases = [
            (payload, "unverified", {}),
            ({"judgment": payload, "patch_sha256": patch_hash, "origin": "original"},
             "patch_only_verified", {"patch_sha256": patch_hash}),
            ({"judgment": payload, "rubric_sha256": rubric_hash},
             "rubric_only_verified", {"rubric_sha256": rubric_hash}),
            ({"judgment": payload, "patch_sha256": patch_hash, "rubric_sha256": rubric_hash},
             "both_hashes_verified", {"patch_sha256": patch_hash, "rubric_sha256": rubric_hash}),
        ]
        for entry, status, provided_hashes in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary, \
                    patch("dogbench.judge.run_codex") as model:
                root = Path(temporary)
                scores = score_predictions([prediction], [outcome()], None, root,
                                           saved_judgments={"patch-task": entry})
                model.assert_not_called()
                self.assertEqual(scores[0]["overall"], 100)
                self.assertEqual(scores[0]["judgment"], payload)
                binding = scores[0]["judgment_binding"]
                self.assertEqual(binding["status"], status)
                self.assertEqual(binding["provided_hashes"], provided_hashes)
                saved_input = root / binding["saved_input_path"]
                self.assertEqual(json.loads(saved_input.read_text()), entry)
                self.assertEqual(hashlib.sha256(saved_input.read_bytes()).hexdigest(),
                                 binding["saved_input_sha256"])

    def test_missed_and_failed_patches_are_zero_without_judging(self):
        predictions = [
            {"instance_id": "missed", "decision": "abstain"},
            {"instance_id": "failed", "status": "timeout", "error": "Time limit"},
            {"instance_id": "unnecessary", "decision": "patch", "patch": PATCH},
        ]
        outcomes = [outcome("missed"), outcome("failed"), outcome("unnecessary", "abstain")]
        with tempfile.TemporaryDirectory() as temporary, patch("dogbench.judge.run_codex") as model:
            scores = score_predictions(predictions, outcomes, None, Path(temporary), saved_judgments={})
            model.assert_not_called()
            self.assertEqual([s["overall"] for s in scores], [0, 0, 0])
            report = build_report(scores, expected_ids=[r["instance_id"] for r in outcomes])
            self.assertEqual(report["summary"]["composite_score"], 0)
            self.assertEqual(report["summary"]["execution_failures"], 1)


class ReportingTests(unittest.TestCase):
    def test_full_population_mean_includes_misses(self):
        rows = [
            {"instance_id": "p1", "expected_outcome": "patch", "overall": 60,
             "decision_correct": True, "mergeable": False, "status": "completed"},
            {"instance_id": "p2", "expected_outcome": "patch", "overall": 0,
             "decision_correct": False, "mergeable": False, "status": "completed"},
            {"instance_id": "n1", "expected_outcome": "abstain", "overall": 100,
             "decision_correct": True, "mergeable": False, "status": "completed"},
        ]
        summary = build_report(rows, expected_ids=["p1", "p2", "n1"])["summary"]
        self.assertEqual(summary["delivered_patch_quality"], 30)
        self.assertEqual(summary["documentation_quality_mean"], 60)
        self.assertAlmostEqual(summary["composite_score"], 2 * 30 * 100 / 130)
        self.assertIsNone(build_report(rows, expected_ids=["p1", "p2", "n1", "n2"])
                          ["summary"]["composite_score"])
        self.assertIsNone(build_report(rows)["summary"]["composite_score"])


if __name__ == "__main__":
    unittest.main()
