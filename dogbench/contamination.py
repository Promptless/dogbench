"""Portable, deterministic DogBench contamination checks.

This module intentionally has no model or service dependencies.  The trusted
controller calls it before exposing a workspace and again after candidate
generation.  Human-reference text must never be copied into an agent workspace.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
NETWORK_RE = re.compile(
    r"\b(?:curl|wget|httpie|git\s+clone|gh\s+(?:api|repo|pr)|web[_-]?(?:search|fetch))\b",
    re.IGNORECASE,
)
BENCHMARK_MARKERS = (
    "human.json",
    "human.diff",
    "gold.patch",
    "frozen-context",
    "review-state",
    "data/candidates/",
    "data/source-maps/",
)


class ContaminationError(RuntimeError):
    """Raised when protected information reaches an agent-visible artifact."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def words(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(text or "")]


def normalized_added_tokens(diff_text: str) -> list[str]:
    tokens: list[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            tokens.extend(words(line[1:]))
    return tokens


def added_lines(diff_text: str, *, minimum_tokens: int = 5) -> list[str]:
    result: list[str] = []
    for line in (diff_text or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        normalized = " ".join(words(line[1:]))
        if len(normalized.split()) >= minimum_tokens:
            result.append(normalized)
    return sorted(set(result))


def _github_pr_markers(url: str) -> list[str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.casefold() != "github.com" or len(parts) != 4 or parts[2] != "pull":
        return []
    owner, repo, _, number = parts
    return [
        f"github.com/{owner}/{repo}/pull/{number}",
        f"repos/{owner}/{repo}/pulls/{number}",
    ]


def protected_markers(item: dict[str, Any]) -> list[tuple[str, str]]:
    markers: list[tuple[str, str]] = [("instance_id", item["instance_id"])]
    urls = [item["source_url"]]
    shas = [item["docs"]["base_sha"], item["docs"].get("head_sha")]
    if item.get("code"):
        urls.append(item["code"]["source_url"])
        shas.extend([item["code"]["base_sha"], item["code"]["head_sha"]])
    for url in urls:
        markers.append(("source_url", url))
        markers.extend(("source_pr_route", marker) for marker in _github_pr_markers(url))
    for sha in shas:
        if sha:
            markers.append(("source_sha", sha))
            markers.append(("source_sha_short", sha[:12]))
    deduplicated: dict[tuple[str, str], None] = {}
    for kind, marker in markers:
        cleaned = str(marker or "").strip()
        if cleaned:
            deduplicated[(kind, cleaned)] = None
    return list(deduplicated)


def scan_protected_text(text: str, item: dict[str, Any]) -> list[dict[str, str]]:
    lowered = text.casefold()
    findings: list[dict[str, str]] = []
    for kind, marker in protected_markers(item):
        if marker.casefold() in lowered:
            findings.append({"code": "protected_identity", "kind": kind, "marker": marker})
    for marker in BENCHMARK_MARKERS:
        if marker.casefold() in lowered:
            findings.append(
                {"code": "benchmark_state_marker", "kind": "path", "marker": marker}
            )
    return findings


def context_audit(
    context: str,
    item: dict[str, Any],
    human_diff: str,
    *,
    protected_source_prose: Iterable[str] = (),
) -> dict[str, Any]:
    findings = scan_protected_text(context, item)
    source_prose = list(protected_source_prose)
    context_words = words(context)
    gold_words = normalized_added_tokens(human_diff)
    normalized_context = " ".join(context_words)

    exact_overlaps = [
        line for line in added_lines(human_diff) if line and line in normalized_context
    ]
    if exact_overlaps:
        findings.append(
            {
                "code": "heldout_gold_line_overlap",
                "kind": "human_reference",
                "marker": sha256_bytes("\n".join(exact_overlaps).encode()),
            }
        )

    longest_overlap = 0
    if context_words and gold_words:
        longest_overlap = difflib.SequenceMatcher(
            None, context_words, gold_words, autojunk=False
        ).find_longest_match().size
    if longest_overlap >= 12:
        findings.append(
            {
                "code": "heldout_gold_token_overlap",
                "kind": "human_reference",
                "marker": f"{longest_overlap}-token contiguous overlap",
            }
        )

    source_prose_overlaps: list[str] = []
    for prose in source_prose:
        prose_words = words(prose)
        if len(prose_words) < 4:
            continue
        normalized_prose = " ".join(prose_words)
        exact = normalized_prose in normalized_context
        longest = (
            difflib.SequenceMatcher(
                None, context_words, prose_words, autojunk=False
            ).find_longest_match().size
            if context_words
            else 0
        )
        if exact or longest >= 12:
            source_prose_overlaps.append(sha256_bytes(prose.encode()))
    if source_prose_overlaps:
        findings.append(
            {
                "code": "source_pr_prose_overlap",
                "kind": "source_pr_solution_prose",
                "marker": source_prose_overlaps[0],
            }
        )

    return {
        "schema_version": "dogbench-context-contamination-v1",
        "ok": not findings,
        "context_sha256": sha256_bytes(context.encode()),
        "human_diff_sha256": sha256_bytes(human_diff.encode()) if human_diff else None,
        "human_added_tokens": len(gold_words),
        "longest_gold_token_overlap": longest_overlap,
        "source_prose_fragments_checked": len(source_prose),
        "findings": findings,
    }


def patch_similarity(candidate_diff: str, reference_diff: str) -> dict[str, Any]:
    candidate = normalized_added_tokens(candidate_diff)
    reference = normalized_added_tokens(reference_diff)
    ratio = (
        difflib.SequenceMatcher(None, candidate, reference, autojunk=False).ratio()
        if candidate and reference
        else 0.0
    )
    return {
        "candidate_added_tokens": len(candidate),
        "reference_added_tokens": len(reference),
        "ratio": round(ratio, 6),
        "exact": bool(reference) and candidate == reference,
    }


def candidate_patch_path_findings(diff_text: str) -> list[dict[str, str]]:
    """Reject paths that escape the docs workspace or target protected internals."""
    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            candidates = line[len("diff --git ") :].split(" b/", 1)
        elif line.startswith(("--- ", "+++ ")):
            candidates = [line[4:]]
        else:
            continue
        for raw in candidates:
            path = raw.strip().strip('"')
            if path == "/dev/null":
                continue
            path = path.removeprefix("a/").removeprefix("b/")
            parts = Path(path).parts
            if (
                not path
                or path.startswith("/")
                or ".." in parts
                or ".git" in parts
                or parts[:1] in {("code",), ("_code_repo",)}
            ):
                if path not in seen:
                    findings.append(
                        {
                            "code": "unsafe_candidate_path",
                            "kind": "path",
                            "marker": path,
                        }
                    )
                    seen.add(path)
    return findings


def trace_audit(paths: Iterable[Path], item: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    source_routes = {
        marker.casefold()
        for kind, marker in protected_markers(item)
        if kind in {"source_url", "source_pr_route"}
    }
    source_repo_markers: set[str] = set()
    for component in (item["docs"], item.get("code")):
        if not component:
            continue
        parsed = urlparse(component["repo_url"])
        if parsed.netloc.casefold() == "github.com":
            source_repo_markers.add(parsed.path.strip("/").removesuffix(".git").casefold())

    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        scanned.append(str(path))
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            lowered = line.casefold()
            categories: list[str] = []
            if any(marker in lowered for marker in source_routes):
                categories.append("source_pr_identity_observed")
            if NETWORK_RE.search(line) and any(
                marker in lowered for marker in source_repo_markers
            ):
                categories.append("source_repository_network_access")
            if NETWORK_RE.search(line) and re.search(r"web[_-]?(?:search|fetch)", line, re.I):
                categories.append("forbidden_web_tool")
            if any(marker.casefold() in lowered for marker in BENCHMARK_MARKERS):
                categories.append("benchmark_state_access")
            if categories:
                findings.append(
                    {
                        "artifact": str(path),
                        "line": number,
                        "categories": sorted(set(categories)),
                        "line_sha256": sha256_bytes(line.encode()),
                    }
                )
                if len(findings) >= 100:
                    break
        if len(findings) >= 100:
            break
    return {
        "schema_version": "dogbench-trace-contamination-v1",
        "ok": not findings,
        "scanned_artifacts": scanned,
        "findings": findings,
    }
