"""Post-run contamination auditing for public DogBench prediction bundles."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

from .contamination import (
    candidate_patch_path_findings,
    patch_similarity,
    scan_protected_text,
    trace_audit,
)
from .sources import SourceAttestationError, attest_item_sources
from .workspaces import WorkspacePreparationError, _prepare_docs


HUMAN_SIMILARITY_THRESHOLD = 0.90
MIN_HUMAN_TOKENS = 20
CROSS_AGENT_SIMILARITY_THRESHOLD = 0.90
MIN_CROSS_AGENT_TOKENS = 80


def _human_diff(item: dict[str, Any]) -> str:
    with tempfile.TemporaryDirectory(prefix="dogbench-audit-") as raw:
        result = _prepare_docs(item["docs"], Path(raw) / "docs")
        return str(result["human_diff"])


def _trace_paths(trace_root: Path, bundle: Path, item_id: str) -> list[Path]:
    candidates = [
        trace_root / bundle.stem / item_id,
        trace_root / item_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            return sorted(path for path in candidate.rglob("*") if path.is_file())
    return []


def audit_prediction_bundles(
    *,
    items: Sequence[dict[str, Any]],
    bundles: Sequence[tuple[Path, Sequence[dict[str, Any]]]],
    trace_root: Path | None,
    allow_missing_traces: bool,
    offline: bool,
) -> dict[str, Any]:
    item_by_id = {item["instance_id"]: item for item in items}
    human_diffs: dict[str, str] = {}
    source_attestations: dict[str, dict[str, Any]] = {}
    for item in items:
        try:
            source_attestations[item["instance_id"]] = attest_item_sources(
                item, offline=offline
            )
        except SourceAttestationError as exc:
            raise WorkspacePreparationError(str(exc)) from exc
        human_diffs[item["instance_id"]] = _human_diff(item)

    reports: list[dict[str, Any]] = []
    patches_by_item: dict[str, list[tuple[str, str]]] = {}
    for bundle_path, records in bundles:
        for record in records:
            item_id = record["instance_id"]
            item = item_by_id[item_id]
            patch = record.get("patch") if record.get("decision") == "patch" else ""
            patch = patch if isinstance(patch, str) else ""
            hard_findings: list[dict[str, Any]] = []
            similarity_alerts: list[dict[str, Any]] = []

            for finding in scan_protected_text(patch, item):
                hard_findings.append({"code": "protected_identity_in_candidate", **finding})
            hard_findings.extend(candidate_patch_path_findings(patch))

            human_similarity = patch_similarity(patch, human_diffs[item_id])
            human_similarity["evaluated"] = (
                human_similarity["reference_added_tokens"] >= MIN_HUMAN_TOKENS
            )
            human_similarity["threshold"] = HUMAN_SIMILARITY_THRESHOLD
            if human_similarity["evaluated"] and (
                human_similarity["exact"]
                or human_similarity["ratio"] >= HUMAN_SIMILARITY_THRESHOLD
            ):
                similarity_alerts.append(
                    {"code": "human_patch_near_verbatim", **human_similarity}
                )

            paths = (
                _trace_paths(trace_root, bundle_path, item_id)
                if trace_root is not None
                else []
            )
            trace_result = trace_audit(paths, item)
            trace_result["coverage"] = "present" if paths else "missing"
            if not paths and not allow_missing_traces:
                hard_findings.append({"code": "primary_trace_missing"})
            elif not trace_result["ok"]:
                hard_findings.append(
                    {
                        "code": "post_dispatch_trace_audit_failed",
                        "findings": trace_result["findings"],
                    }
                )

            label = f"{bundle_path.resolve()}:{item_id}"
            patches_by_item.setdefault(item_id, []).append((label, patch))
            reports.append(
                {
                    "bundle": str(bundle_path),
                    "instance_id": item_id,
                    "source_attestation": source_attestations[item_id]["mode"],
                    "human_similarity": human_similarity,
                    "trace_audit": trace_result,
                    "hard_findings": hard_findings,
                    "similarity_alerts": similarity_alerts,
                }
            )

    report_by_label = {
        f"{Path(row['bundle']).resolve()}:{row['instance_id']}": row for row in reports
    }
    for item_id, candidates in patches_by_item.items():
        for index, (left_label, left_patch) in enumerate(candidates):
            for right_label, right_patch in candidates[index + 1 :]:
                metrics = patch_similarity(left_patch, right_patch)
                if (
                    metrics["candidate_added_tokens"] < MIN_CROSS_AGENT_TOKENS
                    or metrics["reference_added_tokens"] < MIN_CROSS_AGENT_TOKENS
                    or (
                        not metrics["exact"]
                        and metrics["ratio"] < CROSS_AGENT_SIMILARITY_THRESHOLD
                    )
                ):
                    continue
                for label, other in ((left_label, right_label), (right_label, left_label)):
                    report_by_label[label]["similarity_alerts"].append(
                        {
                            "code": "other_agent_patch_near_verbatim",
                            "other_candidate": other,
                            **metrics,
                        }
                    )

    quarantine_count = 0
    global_stop = False
    for row in reports:
        contaminated = bool(row["hard_findings"] or row["similarity_alerts"])
        if contaminated:
            quarantine_count += 1
        stopping_codes = {
            finding["code"]
            for finding in row["hard_findings"] + row["similarity_alerts"]
        } - {"primary_trace_missing", "other_agent_patch_near_verbatim"}
        if stopping_codes:
            global_stop = True
        row["ok"] = not contaminated
        row["decision"] = (
            "accept"
            if not contaminated
            else "quarantine_and_stop"
            if stopping_codes
            else "quarantine_without_global_stop"
        )

    return {
        "schema_version": "dogbench-contamination-audit-v1",
        "ok": quarantine_count == 0,
        "decision": "stop" if global_stop else "continue",
        "candidates": len(reports),
        "quarantined": quarantine_count,
        "reports": reports,
    }
