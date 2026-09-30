"""Original deterministic trajectory audit; host paths are caller configuration."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PRIMARY_AGENT_TRACE_NAMES = {
    "claude.stream.jsonl",
    "codex.events.jsonl",
    "opencode.events.jsonl",
    "devin_session.json",
    "devin_final_status.json",
    "mintlify_final_job.json",
    "mintlify_job_payload.json",
    "promptless_trace.json",
}



def _trace_action_text(line: str) -> str:
    """Extract agent-controlled tool arguments without including tool output."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        # Some OpenCode traces contain literal newlines inside a JSON string,
        # so the serialized event is not valid JSON.  Do not fall back to the
        # complete record in that case: it includes tool output, where ordinary
        # documentation may contain GitHub URLs or benchmark-looking paths.
        # Recover only known action-argument string fields.  Plain-text traces
        # (for example a raw shell command) retain the legacy fallback below.
        if line.lstrip().startswith("{"):
            values: list[str] = []
            action_field = re.compile(
                r'"(?:command|cmd|url|path|file_path|filePath|query)"\s*:\s*'
                r'("(?:\\.|[^"\\])*")'
            )
            for match in action_field.finditer(line):
                try:
                    values.append(json.loads(match.group(1)))
                except json.JSONDecodeError:
                    continue
            return "\n".join(values)
        return line

    action_keys = {
        "command",
        "cmd",
        "url",
        "path",
        "file_path",
        "query",
    }
    values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).casefold() in action_keys:
                    if isinstance(child, str):
                        values.append(child)
                    else:
                        walk(child)
                elif isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return "\n".join(values)



FORBIDDEN_WEB_TOOL_NAMES = {
    "browser",
    "browser_use",
    "browser_use_external",
    "web_fetch",
    "web_search",
    "web_search_call",
    "webfetch",
    "websearch",
}



def _trace_forbidden_web_tools(line: str) -> list[str]:
    """Return provider-side web tools invoked by a structured trace event."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    str(key).casefold() in {"name", "tool", "tool_name", "type"}
                    and isinstance(child, str)
                    and child.casefold() in FORBIDDEN_WEB_TOOL_NAMES
                ):
                    found.add(child.casefold())
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return sorted(found)



def _trace_records(path: Path) -> list[tuple[int, str]]:
    """Return structured trace records without mistaking pretty JSON for actions.

    Devin writes its final status as one indented JSON object. Scanning that
    file line-by-line turns the initial user prompt into a faux shell/web
    action. JSONL traces remain one record per physical line.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        # JSONL records normally occupy one physical line.  OpenCode can emit
        # invalid records with literal newlines inside tool output, however.
        # Group those continuation lines with the event that starts with "{"
        # so output URLs cannot become independent faux actions.
        records: list[tuple[int, str]] = []
        current_line: int | None = None
        current: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            if line.lstrip().startswith("{"):
                if current:
                    records.append((current_line or line_number, "\n".join(current)))
                current_line = line_number
                current = [line]
            elif current:
                current.append(line)
            else:
                records.append((line_number, line))
        if current:
            records.append((current_line or 1, "\n".join(current)))
        return records
    # Promptless exports one structured JSON document containing every remote
    # tool call. Treat each call as its own action record. Flattening the whole
    # document into one action string can combine unrelated substrings from
    # different calls (for example ``docs_repos/`` and a Java package path
    # ``org/apache/flink``) into a fake GitHub API access.
    if isinstance(payload, dict) and isinstance(payload.get("agent_runs"), list):
        tool_records: list[tuple[int, str]] = []
        ordinal = 1
        for run in payload["agent_runs"]:
            if not isinstance(run, dict):
                continue
            calls = run.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                tool_records.append((ordinal, json.dumps(call, sort_keys=True)))
                ordinal += 1
        if tool_records:
            return tool_records
    return [(1, json.dumps(payload, sort_keys=True))]



def audit_post_dispatch_agent_trace(
    log_dir: Path,
    *,
    source_repo: str,
    source_pr_number: int | None,
    agent: str | None = None,
    protected_host_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reject source-PR retrieval and benchmark-host probes after dispatch."""
    source_repo_lower = source_repo.casefold()
    pr_marker = f"/pull/{source_pr_number}" if source_pr_number is not None else None
    exact_source_pr_markers = (
        (
            f"github.com/{source_repo_lower}/pull/{source_pr_number}",
            f"repos/{source_repo_lower}/pulls/{source_pr_number}",
        )
        if source_pr_number is not None
        else ()
    )
    source_repo_url_marker = f"github.com/{source_repo_lower}"
    source_repo_api_pattern = re.compile(
        rf"(?<![A-Za-z0-9_])repos/{re.escape(source_repo_lower)}(?:/|$)"
    )
    host_markers = (*protected_host_paths, ".human.json", ".human.diff")
    findings: list[dict[str, Any]] = []
    scanned_artifacts: list[str] = []
    for path in sorted(log_dir.iterdir()) if log_dir.exists() else []:
        if not path.is_file() or path.name not in PRIMARY_AGENT_TRACE_NAMES:
            continue
        scanned_artifacts.append(path.name)
        for line_number, line in _trace_records(path):
            lowered = line.casefold()
            action_lowered = _trace_action_text(line).casefold()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            # Structured local-agent traces emit both started and completed
            # records for the same command. Decide access from the completed
            # record, where blocked/empty transport is observable, rather than
            # globally stopping on the speculative started record.
            if isinstance(event, dict):
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                if event.get("type") in {"item.started", "tool_start"} or item.get("status") == "in_progress":
                    continue
            categories: list[str] = []
            forbidden_web_tools = _trace_forbidden_web_tools(line)
            if forbidden_web_tools:
                categories.append("forbidden_web_tool")
            exact_source_pr_cli = bool(
                source_pr_number is not None
                and source_repo_lower in action_lowered
                and re.search(
                    rf"\bgh\s+pr\s+(?:view|diff|checkout)\s+{source_pr_number}\b",
                    action_lowered,
                )
            )
            if any(marker in action_lowered for marker in exact_source_pr_markers):
                categories.append("source_pr_url_observed")
            if exact_source_pr_cli:
                categories.append("source_pr_cli_access")
            source_repo_cli_access = bool(
                source_repo_lower
                and source_repo_lower in action_lowered
                and re.search(
                    r"\b(?:git\s+clone|gh\s+repo\s+(?:clone|view)|"
                    r"gh\s+pr\s+(?:view|diff|checkout))\b",
                    action_lowered,
                )
            )
            source_repo_reference = bool(
                source_repo_lower and (
                    source_repo_url_marker in action_lowered
                    or source_repo_api_pattern.search(action_lowered)
                    or source_repo_cli_access
                )
            )
            network_command = bool(re.search(
                r"\b(?:curl|wget|httpie|gh\s+(?:api|repo|pr)|git\s+clone)\b",
                action_lowered,
            ))
            if source_repo_reference and (network_command or source_repo_cli_access):
                blocked_markers = (
                    "http 403", "error: 403", "returned error: 403",
                    "could not resolve host", "connection refused", "(no output)",
                    "no such file or directory",
                )
                no_transport_output = any(marker in lowered for marker in blocked_markers)
                if no_transport_output:
                    categories = [
                        category for category in categories
                        if category not in {
                            "source_pr_url_observed", "source_pr_cli_access",
                            "source_pr_url_access", "source_repo_access",
                        }
                    ]
                    categories.append("blocked_source_repo_access")
                else:
                    categories.append("source_repo_access")
            if (
                source_repo_lower
                and pr_marker
                and source_repo_lower in action_lowered
                and pr_marker in action_lowered
            ):
                categories.append("source_pr_url_access")
            if any(marker.casefold() in action_lowered for marker in host_markers):
                categories.append("benchmark_host_path_access")
            if categories:
                findings.append({
                    "artifact": path.name,
                    "line": line_number,
                    "categories": sorted(set(categories)),
                    "forbidden_web_tools": forbidden_web_tools,
                    "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                })
                if len(findings) >= 50:
                    break
        if len(findings) >= 50:
            break
    trace_coverage = (
        "remote_metadata_only"
        if agent in {"mintlify", "devin"} and scanned_artifacts
        else (
            "full_remote_trace"
            if agent == "promptless" and scanned_artifacts
            else ("full_primary_local" if scanned_artifacts else "missing")
        )
    )
    return {
        "schema_version": "docbench-post-dispatch-source-access-v1",
        "ok": not findings,
        "source_repo": source_repo,
        "source_pr_number": source_pr_number,
        "trace_coverage": trace_coverage,
        "trace_artifacts_scanned": scanned_artifacts,
        "findings": findings,
        "errors": (
            [f"agent trajectory contains {len(findings)} forbidden source/host/web access event(s)"]
            if findings else []
        ),
    }

