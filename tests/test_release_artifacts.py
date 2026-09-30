"""Offline integrity and development-only boundary checks for bundled artifacts."""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ReleaseArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads((DATA / "release-reference.json").read_text())
        cls.development = set(cls.reference["development_item_ids"])

    def test_pinned_release_reference(self) -> None:
        self.assertEqual(self.reference["dataset_id"], "promptless-research/dogbench-dev")
        self.assertEqual(
            self.reference["dataset_revision"], "f8e5dd2040414810cc79e8f77167c84a12d76932"
        )
        self.assertEqual(len(self.development), 175)
        self.assertEqual(self.reference["expected_outcomes"], {"patch": 123, "abstain": 52})
        self.assertEqual(
            self.reference["files"]["items.jsonl"]["sha256"],
            "fd3c4707b67418da605edcadc47ff11da6c4412e0eafbea76ae855f4830412ef",
        )
        self.assertEqual(
            self.reference["files"]["outcomes.jsonl"]["sha256"],
            "48481665c0887a64a136a34dd7dc6d07c96dac3ed33716b674d538277b3efcce",
        )
        self.assertFalse((DATA / "items.jsonl").exists())
        self.assertFalse((DATA / "outcomes.jsonl").exists())

    def test_sample_integrity_and_outcome_alignment(self) -> None:
        sample = DATA / "sample-5"
        manifest = json.loads((sample / "manifest.json").read_text())
        for relative, expected in manifest["files"].items():
            self.assertEqual(sha256((sample / relative).read_bytes()), expected, relative)
        from dogbench.items import load_items
        from dogbench.judge import criterion_blocks

        items = load_items(sample / "items.jsonl")
        outcomes = rows(sample / "outcomes.jsonl")
        item_ids = {row["instance_id"] for row in items}
        self.assertEqual(len(items), 5)
        self.assertEqual({row["instance_id"] for row in outcomes}, item_ids)
        self.assertLessEqual(item_ids, self.development)
        self.assertEqual(
            collections.Counter(row["expected_outcome"] for row in outcomes),
            {"patch": 5},
        )
        rubrics = {row["instance_id"]: row["rubric"] for row in rows(sample / "rubrics.jsonl")}
        for outcome in outcomes:
            if outcome["expected_outcome"] == "patch":
                self.assertEqual(outcome["rubric_markdown"], rubrics[outcome["instance_id"]])
                self.assertEqual(
                    sha256(outcome["rubric_markdown"].encode()), outcome["rubric_sha256"]
                )
        self.assertEqual(
            sum(len(criterion_blocks(row["rubric_markdown"])) for row in outcomes),
            manifest["rubric_criterion_count"],
        )
        self.assertEqual((sample / "DATASET_LICENSE.md").read_bytes(),
                         (ROOT / "DATASET_LICENSE.md").read_bytes())

    def test_historical_archive_integrity_and_membership(self) -> None:
        archive = ROOT / "historical-trajectories"
        spec = importlib.util.spec_from_file_location("archive_verify", archive / "verify.py")
        assert spec and spec.loader
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        verifier.main()
        index = json.loads((archive / "index.json").read_text())
        self.assertLessEqual(set(index["item_ids"]), self.development)
        self.assertEqual(len(index["records"]), 735)
        self.assertEqual(len(index["models"]), 7)

if __name__ == "__main__":
    unittest.main()
