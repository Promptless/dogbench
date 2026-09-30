"""Deterministic post-run contamination checks for agent candidates.

This module runs in the trusted parent process after an agent has finished.  It
must never be mounted into, or otherwise exposed to, the agent workspace: it
loads the held-out human patch solely to audit the completed candidate.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "docbench-contamination-gate-v1"
STOP_SCHEMA_VERSION = "docbench-contamination-stop-v1"
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_MIN_HUMAN_TOKENS = 20
DEFAULT_MIN_CROSS_AGENT_TOKENS = 80
MANAGED_AGENTS = {"mintlify", "devin", "promptless"}
WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")

# One preferred artifact per canonical lane.  Historical directories also
# contain aliases, experiments, and superseded budget variants; comparing
# against every ``*.json`` can mistake an older copy of the same lane for an
# independent agent and can even attempt to decode macOS AppleDouble files.
CANONICAL_CANDIDATE_KEY_GROUPS = (
    ("codex",),
    ("devin",),
    ("mintlify",),
    (
        "claude__bedrock-sonnet-4-6",
        "claude__claude-sonnet-4-6",
        "claude",
    ),
    (
        "claude__bedrock-opus-4-8",
        "claude__claude-opus-4-8",
    ),
    (
        "opencode__openrouter_qwen_qwen3-coder-next",
        "opencode__openrouter_qwen_qwen3-coder-plus",
    ),
    ("opencode__openrouter_z-ai_glm-5.2",),
    ("opencode__openrouter_moonshotai_kimi-k2.7-code",),
)

# These filenames contain the exact bytes sent to a managed or local agent.
# Attestations and candidate JSONs are deliberately excluded: they are written
# by the trusted parent and may legitimately record the protected identity.
MODEL_INPUT_ARTIFACT_NAMES = {
    "prompt.txt",
    "mintlify_prompt.txt",
    "mintlify_job_payload.json",
    "devin_session_request.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_added_tokens(diff_text: str) -> list[str]:
    """Return normalized tokens from actual unified-diff additions only."""
    added: list[str] = []
    for line in (diff_text or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added.extend(token.casefold() for token in WORD_RE.findall(line[1:]))
    return added


def patch_similarity(candidate_diff: str, reference_diff: str) -> dict[str, Any]:
    candidate_tokens = normalized_added_tokens(candidate_diff)
    reference_tokens = normalized_added_tokens(reference_diff)
    ratio = (
        difflib.SequenceMatcher(None, candidate_tokens, reference_tokens, autojunk=False).ratio()
        if candidate_tokens and reference_tokens
        else 0.0
    )
    return {
        "candidate_added_tokens": len(candidate_tokens),
        "reference_added_tokens": len(reference_tokens),
        "ratio": round(ratio, 6),
        "exact": bool(reference_tokens) and candidate_tokens == reference_tokens,
    }


def load_human_diff(item_id: str, candidates_root: Path) -> tuple[str, str | None]:
    human_json = candidates_root / item_id / "human.json"
    if human_json.exists():
        try:
            data = json.loads(human_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        for key in ("human_docs_diff", "docs_diff"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key], str(human_json)
    human_diff = candidates_root / item_id / "human.diff"
    if human_diff.exists():
        return human_diff.read_text(encoding="utf-8", errors="replace"), str(human_diff)
    return "", None


def _candidate_records(
    item_id: str,
    candidates_root: Path,
    *,
    exclude_candidate_key: str,
) -> Iterable[tuple[str, Path, dict[str, Any]]]:
    candidate_dir = candidates_root / item_id
    if not candidate_dir.exists():
        return
    for aliases in CANONICAL_CANDIDATE_KEY_GROUPS:
        # Do not compare a rerun with a stale alias from its own lane.
        if exclude_candidate_key in aliases:
            continue
        for candidate_key in aliases:
            path = candidate_dir / f"{candidate_key}.json"
            if not path.is_file() or path.name.startswith("._"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("docs_diff"), str):
                continue
            yield candidate_key, path, data
            break


def _identity_markers(
    *,
    item_id: str,
    source_repo: str,
    source_pr_number: int | None,
    source_head_shas: Iterable[str],
) -> list[tuple[str, str]]:
    # The repository slug is normal task context: the model sees its frozen
    # base tree and may encounter upstream package/import names. Only the
    # exact source-PR route (which would let it retrieve the held-out human
    # patch) and source commit identities are protected. Keep this aligned
    # with run_candidate.audit_final_agent_prompt.
    markers = [
        ("item_id", item_id),
    ]
    if source_pr_number is not None:
        markers.extend([
            ("source_pr_url", f"github.com/{source_repo}/pull/{source_pr_number}"),
            ("source_pr_api", f"repos/{source_repo}/pulls/{source_pr_number}"),
        ])
    for sha in source_head_shas:
        cleaned = str(sha or "").strip()
        if len(cleaned) >= 12:
            markers.append(("source_head_sha", cleaned))
            markers.append(("source_head_sha_short", cleaned[:12]))
    return [(kind, marker) for kind, marker in markers if marker]


def scan_model_inputs(
    paths: Iterable[Path],
    *,
    item_id: str,
    source_repo: str,
    source_pr_number: int | None,
    source_head_shas: Iterable[str] = (),
) -> dict[str, Any]:
    markers = _identity_markers(
        item_id=item_id,
        source_repo=source_repo,
        source_pr_number=source_pr_number,
        source_head_shas=source_head_shas,
    )
    forbidden_benchmark_markers = (
        "human.json",
        "human.diff",
        "frozen-context",
        "review-state",
        "data/candidates/",
        "data/source-maps/",
    )
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in sorted(set(paths)):
        if not path.exists() or not path.is_file():
            continue
        scanned.append(str(path))
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        for kind, marker in markers:
            if marker.casefold() in text:
                findings.append({"artifact": str(path), "kind": kind, "marker": marker})
        for marker in forbidden_benchmark_markers:
            if marker.casefold() in text:
                findings.append({
                    "artifact": str(path),
                    "kind": "benchmark_state_marker",
                    "marker": marker,
                })
    return {"scanned_artifacts": scanned, "findings": findings, "ok": not findings}


def model_input_artifacts(log_dir: Path) -> list[Path]:
    return [log_dir / name for name in sorted(MODEL_INPUT_ARTIFACT_NAMES)]


def build_contamination_report(
    *,
    item_id: str,
    agent: str,
    candidate_key: str,
    candidate_diff: str,
    candidates_root: Path,
    model_input_paths: Iterable[Path],
    source_repo: str,
    source_pr_number: int | None,
    source_head_shas: Iterable[str] = (),
    prompt_audit: dict[str, Any] | None = None,
    trace_audit: dict[str, Any] | None = None,
    managed_mirror_attestation: dict[str, Any] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_human_tokens: int = DEFAULT_MIN_HUMAN_TOKENS,
    min_cross_agent_tokens: int = DEFAULT_MIN_CROSS_AGENT_TOKENS,
) -> dict[str, Any]:
    """Build one fail-closed, candidate-level contamination decision."""
    hard_findings: list[dict[str, Any]] = []
    similarity_alerts: list[dict[str, Any]] = []

    input_scan = scan_model_inputs(
        model_input_paths,
        item_id=item_id,
        source_repo=source_repo,
        source_pr_number=source_pr_number,
        source_head_shas=source_head_shas,
    )
    for finding in input_scan["findings"]:
        hard_findings.append({"code": "protected_identity_in_model_input", **finding})

    if prompt_audit is not None and not prompt_audit.get("ok"):
        hard_findings.append({
            "code": "prompt_contamination_audit_failed",
            "errors": list(prompt_audit.get("errors") or []),
        })
    if trace_audit is not None and not trace_audit.get("ok"):
        trace_categories = {
            str(category)
            for finding in trace_audit.get("findings") or []
            for category in finding.get("categories") or []
        }
        trace_code = (
            "post_dispatch_policy_audit_failed"
            if trace_categories and trace_categories <= {
                "forbidden_web_tool", "blocked_source_repo_access"
            }
            else "post_dispatch_trace_audit_failed"
        )
        hard_findings.append({
            "code": trace_code,
            "errors": list(trace_audit.get("errors") or []),
            "findings": list(trace_audit.get("findings") or []),
        })
    if managed_mirror_attestation is not None and not managed_mirror_attestation.get("ok"):
        hard_findings.append({
            "code": "managed_mirror_attestation_failed",
            "errors": list(managed_mirror_attestation.get("errors") or []),
        })

    human_diff, human_path = load_human_diff(item_id, candidates_root)
    human_similarity = patch_similarity(candidate_diff, human_diff)
    human_similarity.update({
        "reference_path": human_path,
        "evaluated": human_similarity["reference_added_tokens"] >= min_human_tokens,
        "threshold": similarity_threshold,
        "minimum_reference_tokens": min_human_tokens,
    })
    if human_similarity["evaluated"] and (
        human_similarity["exact"] or human_similarity["ratio"] >= similarity_threshold
    ):
        similarity_alerts.append({
            "code": "human_patch_near_verbatim",
            **human_similarity,
        })

    cross_agent_similarity: list[dict[str, Any]] = []
    for other_key, other_path, other in _candidate_records(
        item_id,
        candidates_root,
        exclude_candidate_key=candidate_key,
    ):
        metrics = patch_similarity(candidate_diff, other.get("docs_diff") or "")
        if (
            metrics["candidate_added_tokens"] < min_cross_agent_tokens
            or metrics["reference_added_tokens"] < min_cross_agent_tokens
        ):
            continue
        if not (metrics["exact"] or metrics["ratio"] >= similarity_threshold):
            continue
        row = {
            "candidate_key": other_key,
            "candidate_path": str(other_path),
            "minimum_candidate_and_reference_tokens": min_cross_agent_tokens,
            **metrics,
        }
        cross_agent_similarity.append(row)
        similarity_alerts.append({"code": "other_agent_patch_near_verbatim", **row})

    trace_coverage = (trace_audit or {}).get("trace_coverage")
    if not trace_coverage:
        trace_coverage = "unavailable_managed_service" if agent in MANAGED_AGENTS else "missing"
    if agent not in MANAGED_AGENTS and trace_coverage == "missing":
        hard_findings.append({"code": "local_primary_trace_missing"})

    # A missing local trace is audit-insufficient, and cross-agent textual
    # convergence is a review signal rather than proof that benchmark gold
    # leaked. Quarantine those candidates without halting unrelated lanes.
    # Direct human-patch similarity and protected-input/trace findings remain
    # global stops.
    non_stopping_findings = {
        "local_primary_trace_missing",
        "post_dispatch_policy_audit_failed",
    }
    non_stopping_similarity_alerts = {"other_agent_patch_near_verbatim"}
    only_non_stopping_findings = (
        bool(hard_findings or similarity_alerts)
        and all(row.get("code") in non_stopping_findings for row in hard_findings)
        and all(
            row.get("code") in non_stopping_similarity_alerts
            for row in similarity_alerts
        )
    )
    contaminated = bool(hard_findings or similarity_alerts)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "item_id": item_id,
        "agent": agent,
        "candidate_key": candidate_key,
        "ok": not contaminated,
        "decision": (
            "quarantine_without_global_stop" if only_non_stopping_findings
            else "quarantine_and_stop" if contaminated else "accept"
        ),
        "hard_findings": hard_findings,
        "similarity_alerts": similarity_alerts,
        "input_scan": input_scan,
        "trace_coverage": trace_coverage,
        "human_similarity": human_similarity,
        "cross_agent_similarity": cross_agent_similarity,
    }


def write_contamination_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_pre_dispatch_failure(
    *,
    log_dir: Path,
    item_id: str,
    agent: str,
    candidate_key: str,
    code: str,
    message: str,
    stop_batch: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Persist a terminal input-validity/contamination failure before dispatch."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "phase": "pre_dispatch",
        "item_id": item_id,
        "agent": agent,
        "candidate_key": candidate_key,
        "ok": False,
        "decision": (
            "quarantine_and_stop" if stop_batch else "quarantine_without_global_stop"
        ),
        "hard_findings": [{"code": code, "message": message}],
        "similarity_alerts": [],
        "input_scan": {"scanned_artifacts": [], "findings": [], "ok": True},
        "trace_coverage": "not_started",
        "human_similarity": {"evaluated": False},
        "cross_agent_similarity": [],
    }
    report_path = log_dir / "contamination_report.json"
    write_contamination_report(report, report_path)
    trigger_batch_stop(
        stop_file_from_env(),
        report=report,
        report_path=report_path,
    )
    return report, report_path


def stop_file_from_env() -> Path | None:
    raw = os.environ.get("DOCBENCH_CONTAMINATION_STOP_FILE", "").strip()
    return Path(raw) if raw else None


def trigger_batch_stop(
    stop_path: Path | None,
    *,
    report: dict[str, Any],
    report_path: Path,
) -> bool:
    """Create the shared stop marker once. Return whether this call created it."""
    if (
        stop_path is None
        or report.get("ok")
        or report.get("decision") == "quarantine_without_global_stop"
    ):
        return False
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": STOP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "item_id": report.get("item_id"),
        "agent": report.get("agent"),
        "candidate_key": report.get("candidate_key"),
        "report_path": str(report_path),
        "hard_finding_codes": [row.get("code") for row in report.get("hard_findings", [])],
        "similarity_alert_codes": [
            row.get("code") for row in report.get("similarity_alerts", [])
        ],
    }
    try:
        fd = os.open(stop_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return True


def batch_stop_requested(path: Path | None = None) -> bool:
    stop_path = path or stop_file_from_env()
    return bool(stop_path and stop_path.exists())
