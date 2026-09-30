"""Offline verification for prepared DogBench workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contamination import canonical_json_bytes, scan_protected_text, sha256_bytes
from .workspaces import (
    WorkspacePreparationError,
    _assert_safe_symlinks,
    _blob_sha256,
    _git,
    _run,
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspacePreparationError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkspacePreparationError(f"expected a JSON object: {path}")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspacePreparationError(f"attested path escapes output root: {relative}") from exc
    return candidate


def _verify_artifact(path: Path, expected: dict[str, Any]) -> None:
    data = path.read_bytes()
    if len(data) != expected.get("bytes") or sha256_bytes(data) != expected.get("sha256"):
        raise WorkspacePreparationError(f"artifact hash mismatch: {path}")


def _verify_repo_is_sealed(repo: Path, source_shas: list[str]) -> None:
    if _git(repo, "remote").decode().strip():
        raise WorkspacePreparationError(f"workspace repository has a remote: {repo}")
    if _git(repo, "status", "--porcelain", "--untracked-files=all").decode().strip():
        raise WorkspacePreparationError(f"workspace repository is not clean: {repo}")
    for sha in source_shas:
        result = _run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=False,
        )
        if result.returncode == 0:
            raise WorkspacePreparationError(
                f"upstream source commit is reachable from anonymous repository: {repo}"
            )
    if _git(repo, "fsck", "--unreachable", "--no-reflogs").decode().strip():
        raise WorkspacePreparationError(
            f"anonymous repository retains unreachable source objects: {repo}"
        )


def verify_attestation(output_root: Path, path: Path) -> dict[str, Any]:
    attestation = _load_object(path)
    if attestation.get("schema_version") != "dogbench-workspace-attestation-v1":
        raise WorkspacePreparationError(f"unsupported workspace attestation: {path}")
    recorded_digest = attestation.get("logical_input_sha256")
    digest_input = dict(attestation)
    digest_input.pop("logical_input_sha256", None)
    actual_digest = sha256_bytes(canonical_json_bytes(digest_input))
    if recorded_digest != actual_digest:
        raise WorkspacePreparationError(f"logical input hash mismatch: {path}")

    workspace = _safe_child(output_root, str(attestation.get("workspace") or ""))
    if not workspace.is_dir():
        raise WorkspacePreparationError(f"workspace is missing: {workspace}")
    _assert_safe_symlinks(workspace)
    expected_entries = {"TASK.md", "workspace.json", "docs"}
    if attestation.get("code") is not None:
        expected_entries.add("code")
    actual_entries = {entry.name for entry in workspace.iterdir()}
    if actual_entries != expected_entries:
        raise WorkspacePreparationError(
            "workspace top-level entries drifted: "
            f"expected {sorted(expected_entries)}, got {sorted(actual_entries)}"
        )
    item = attestation.get("item") or {}
    if not isinstance(item, dict) or not item.get("instance_id"):
        raise WorkspacePreparationError(f"attestation has no item contract: {path}")
    if (attestation.get("context_audit") or {}).get("ok") is not True:
        raise WorkspacePreparationError(f"context audit is not clean: {path}")
    if (attestation.get("source_attestation") or {}).get("ok") is not True:
        raise WorkspacePreparationError(f"source attestation is not clean: {path}")

    artifacts = attestation.get("artifacts") or {}
    if set(artifacts) != {"TASK.md", "workspace.json"}:
        raise WorkspacePreparationError("workspace artifact set is not canonical")
    for name, expected in artifacts.items():
        _verify_artifact(workspace / name, expected)
    model_visible = "\n".join(
        (workspace / name).read_text(encoding="utf-8", errors="replace")
        for name in ("TASK.md", "workspace.json")
    )
    findings = scan_protected_text(model_visible, item)
    if findings:
        raise WorkspacePreparationError(
            f"protected identity found in workspace artifacts: {findings[0]['kind']}"
        )

    docs_repo = workspace / "docs"
    docs_attestation = attestation.get("docs") or {}
    docs_source = item["docs"]
    _verify_repo_is_sealed(
        docs_repo,
        [sha for sha in (docs_source["base_sha"], docs_source.get("head_sha")) if sha],
    )
    if _git(docs_repo, "rev-parse", "HEAD").decode().strip() != docs_attestation.get("commit"):
        raise WorkspacePreparationError("documentation workspace head drifted")
    if _git(docs_repo, "rev-parse", "HEAD^{tree}").decode().strip() != docs_attestation.get("tree"):
        raise WorkspacePreparationError("documentation workspace tree drifted")

    code_source = item.get("code")
    code_attestation = attestation.get("code")
    if code_source is None:
        if code_attestation is not None or (workspace / "code").exists():
            raise WorkspacePreparationError("docs-only workspace unexpectedly contains code input")
    else:
        if not isinstance(code_attestation, dict):
            raise WorkspacePreparationError("code attestation is missing")
        code_repo = workspace / "code"
        _verify_repo_is_sealed(
            code_repo, [code_source["base_sha"], code_source["head_sha"]]
        )
        head = _git(code_repo, "rev-parse", "HEAD").decode().strip()
        parent = _git(code_repo, "rev-parse", "HEAD^").decode().strip()
        if head != code_attestation.get("head_commit"):
            raise WorkspacePreparationError("code workspace head drifted")
        if parent != code_attestation.get("base_commit"):
            raise WorkspacePreparationError("code workspace parent drifted")
        paths = sorted(
            line
            for line in _git(
                code_repo, "diff", "--name-only", "--no-renames", "HEAD^..HEAD"
            ).decode().splitlines()
            if line
        )
        if paths != sorted(code_attestation.get("paths") or []):
            raise WorkspacePreparationError("code workspace path set drifted")
        actual_blobs = {path: _blob_sha256(code_repo, "HEAD", path) for path in paths}
        if actual_blobs != code_attestation.get("blob_sha256"):
            raise WorkspacePreparationError("code workspace blob content drifted")

    from .research_inputs import input_binding, verify_research_workspace
    binding = input_binding(item)
    if binding is not None:
        if attestation.get("research_input") != binding:
            raise WorkspacePreparationError(
                "frozen research binding missing or changed; prepare a fresh workspace"
            )
        verify_research_workspace(item, workspace, (workspace / "TASK.md").read_bytes())

    return {
        "instance_id": item["instance_id"],
        "workspace": str(workspace),
        "logical_input_sha256": recorded_digest,
        "source_attestation": (attestation.get("source_attestation") or {}).get("mode"),
        "status": "ready",
    }


def verify_output(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    index = _load_object(output_root / "index.json")
    if index.get("schema_version") != "dogbench-workspace-index-v1":
        raise WorkspacePreparationError("unsupported workspace index")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in index.get("items") or []:
        item_id = row.get("instance_id")
        if item_id in seen:
            raise WorkspacePreparationError(f"duplicate item in workspace index: {item_id}")
        seen.add(item_id)
        attestation_path = Path(str(row.get("attestation") or ""))
        if not attestation_path.is_absolute():
            attestation_path = _safe_child(output_root, str(attestation_path))
        else:
            try:
                attestation_path.resolve().relative_to(output_root)
            except ValueError as exc:
                raise WorkspacePreparationError(
                    f"attestation is outside output root: {attestation_path}"
                ) from exc
        result = verify_attestation(output_root, attestation_path)
        if result["instance_id"] != item_id:
            raise WorkspacePreparationError("workspace index item identity mismatch")
        if result["logical_input_sha256"] != row.get("logical_input_sha256"):
            raise WorkspacePreparationError("workspace index logical input hash mismatch")
        indexed_workspace = _safe_child(output_root, str(row.get("path") or ""))
        if indexed_workspace != Path(result["workspace"]).resolve():
            raise WorkspacePreparationError("workspace index path mismatch")
        results.append(result)
    if not results:
        raise WorkspacePreparationError("workspace index is empty")
    indexed_attestations = {
        _safe_child(output_root, str(row.get("attestation") or "")).resolve()
        for row in index.get("items") or []
    }
    actual_attestations = {
        path.resolve() for path in (output_root / "attestations").glob("*.json")
    }
    if indexed_attestations != actual_attestations:
        raise WorkspacePreparationError("workspace index does not cover every attestation")
    indexed_workspaces = {
        _safe_child(output_root, str(row.get("path") or "")).resolve()
        for row in index.get("items") or []
    }
    actual_workspaces = {
        path.resolve() for path in (output_root / "workspaces").iterdir() if path.is_dir()
    }
    if indexed_workspaces != actual_workspaces:
        raise WorkspacePreparationError("workspace index does not cover every workspace")
    return {
        "schema_version": "dogbench-workspace-verification-v1",
        "ok": True,
        "verified": len(results),
        "items": results,
    }
