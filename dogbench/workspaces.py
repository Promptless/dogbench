"""Prepare identity-masked local workspaces from public DogBench items."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .contamination import canonical_json_bytes, context_audit, sha256_bytes
from .sources import SourceAttestationError, attest_item_sources
from .research_inputs import research_input, input_binding, verify_research_workspace


class WorkspacePreparationError(RuntimeError):
    """Raised when a Git input cannot be materialized exactly."""


_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "DogBench",
    "GIT_AUTHOR_EMAIL": "dogbench@noreply.invalid",
    "GIT_COMMITTER_NAME": "DogBench",
    "GIT_COMMITTER_EMAIL": "dogbench@noreply.invalid",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
}


def _run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            list(args),
            cwd=cwd,
            input=input_bytes,
            capture_output=True,
            check=False,
            env={**os.environ, **_GIT_IDENTITY},
        )
    except FileNotFoundError as exc:
        raise WorkspacePreparationError("git is required to prepare DogBench inputs") from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkspacePreparationError(
            f"command failed: {' '.join(args)}" + (f"\n{detail}" if detail else "")
        )
    return result


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return _run(
        ["git", "-C", str(repo), *args], input_bytes=input_bytes
    ).stdout


def _fetch(repo: Path, repo_url: str, sha: str) -> None:
    remotes = _git(repo, "remote").decode().splitlines()
    if "source" not in remotes:
        _git(repo, "remote", "add", "source", repo_url)
    _git(repo, "fetch", "--quiet", "--no-tags", "--depth=1", "source", sha)
    resolved = _git(repo, "rev-parse", "--verify", f"{sha}^{{commit}}").decode().strip()
    if resolved.lower() != sha.lower():
        raise WorkspacePreparationError(
            f"{repo_url}: requested {sha}, fetched {resolved or 'nothing'}"
        )


def _neutral_commit(repo: Path, tree: str, message: str, *, parent: str | None = None) -> str:
    args = ["commit-tree", tree, "-m", message]
    if parent:
        args.extend(["-p", parent])
    return _git(repo, *args).decode().strip()


def _remove_source_identity(repo: Path, source_shas: Sequence[str]) -> None:
    _git(repo, "remote", "remove", "source")
    for name in ("FETCH_HEAD", "ORIG_HEAD", "shallow"):
        (repo / ".git" / name).unlink(missing_ok=True)
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "gc", "--prune=now", "--quiet")
    for sha in source_shas:
        visible = _run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=False,
        )
        if visible.returncode == 0:
            raise WorkspacePreparationError(
                "an upstream commit remained reachable in the anonymized repository"
            )
    unreachable = _git(repo, "fsck", "--unreachable", "--no-reflogs").decode().strip()
    if unreachable:
        raise WorkspacePreparationError(
            "unreachable source objects remained in the anonymized repository"
        )


def _blob_sha256(repo: Path, revision: str, path: str) -> str | None:
    result = _run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"], check=False
    )
    return sha256_bytes(result.stdout) if result.returncode == 0 else None


def _assert_safe_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_symlink():
            continue
        target = os.readlink(path)
        if os.path.isabs(target):
            raise WorkspacePreparationError(f"absolute symlink is unsafe: {path}")
        resolved = (path.parent / target).resolve(strict=False)
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise WorkspacePreparationError(f"symlink escapes the workspace: {path}") from exc


def _prepare_docs(docs: dict[str, Any], target: Path) -> dict[str, Any]:
    repo_url = docs["repo_url"]
    base_sha = docs["base_sha"]
    head_sha = docs.get("head_sha")
    _run(["git", "init", "--quiet", "--initial-branch=main", str(target)])
    _fetch(target, repo_url, base_sha)
    human_diff = b""
    changed_paths: set[str] = set()
    if head_sha is not None:
        _fetch(target, repo_url, head_sha)
        human_diff = _git(
            target,
            "diff",
            "--binary",
            "--full-index",
            "--no-renames",
            f"{base_sha}..{head_sha}",
            "--",
            *docs["paths"],
        )
        changed_paths = {
            line
            for line in _git(
                target,
                "diff",
                "--name-only",
                "--no-renames",
                f"{base_sha}..{head_sha}",
                "--",
                *docs["paths"],
            ).decode().splitlines()
            if line
        }
        missing = sorted(set(docs["paths"]) - changed_paths)
        if missing:
            raise WorkspacePreparationError(
                "held-out documentation paths contain no change: " + ", ".join(missing)
            )
        if docs["paths"] and not human_diff.strip():
            raise WorkspacePreparationError("held-out documentation diff is empty")
    tree = _git(target, "rev-parse", f"{base_sha}^{{tree}}").decode().strip()
    commit = _neutral_commit(target, tree, "snapshot")
    _git(target, "update-ref", "refs/heads/main", commit)
    _git(target, "reset", "--quiet", "--hard", commit)
    _remove_source_identity(target, [sha for sha in (base_sha, head_sha) if sha])
    _assert_safe_symlinks(target)
    return {
        "commit": commit,
        "tree": tree,
        "human_diff": human_diff.decode("utf-8", errors="replace"),
        "human_diff_sha256": sha256_bytes(human_diff),
        "changed_paths": sorted(changed_paths),
    }


def _prepare_code(code: dict[str, Any], target: Path) -> dict[str, Any]:
    base_sha = code["base_sha"]
    head_sha = code["head_sha"]
    _run(["git", "init", "--quiet", "--initial-branch=main", str(target)])
    _fetch(target, code["repo_url"], base_sha)
    _git(target, "fetch", "--quiet", "--no-tags", "--depth=1", "source", head_sha)
    _git(target, "rev-parse", "--verify", f"{head_sha}^{{commit}}")

    patch = _git(
        target,
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        f"{base_sha}..{head_sha}",
        "--",
        *code["paths"],
    )
    if not patch.strip():
        raise WorkspacePreparationError(
            "the selected code paths contain no changes between base_sha and head_sha"
        )

    base_tree = _git(target, "rev-parse", f"{base_sha}^{{tree}}").decode().strip()
    base_commit = _neutral_commit(target, base_tree, "snapshot")
    _git(target, "update-ref", "refs/heads/main", base_commit)
    _git(target, "reset", "--quiet", "--hard", base_commit)
    _git(target, "apply", "--binary", "--index", "-", input_bytes=patch)
    _git(target, "commit", "--quiet", "-m", "change")
    head_commit = _git(target, "rev-parse", "HEAD").decode().strip()

    changed = {
        line
        for line in _git(
            target, "diff", "--name-only", "--no-renames", f"{base_commit}..{head_commit}"
        ).decode().splitlines()
        if line
    }
    if changed != set(code["paths"]):
        raise WorkspacePreparationError(
            "materialized code paths do not match the item: "
            f"expected {sorted(code['paths'])}, got {sorted(changed)}"
        )

    expected_blob_hashes = {
        path: _blob_sha256(target, head_sha, path) for path in code["paths"]
    }
    actual_blob_hashes = {
        path: _blob_sha256(target, head_commit, path) for path in code["paths"]
    }
    if expected_blob_hashes != actual_blob_hashes:
        raise WorkspacePreparationError(
            "materialized code content differs from the frozen source head"
        )

    _remove_source_identity(target, [base_sha, head_sha])
    _assert_safe_symlinks(target)
    return {
        "base_commit": base_commit,
        "head_commit": head_commit,
        "base_tree": base_tree,
        "head_tree": _git(target, "rev-parse", "HEAD^{tree}").decode().strip(),
        "paths": sorted(changed),
        "blob_sha256": actual_blob_hashes,
        "patch_sha256": sha256_bytes(patch),
    }


def _task_markdown(item: dict[str, Any], *, has_code: bool) -> str:
    code_note = (
        "The `code/` repository is read-only reference material. Its `change` "
        "commit contains the implementation change.\n\n"
        if has_code
        else ""
    )
    return (
        "# DogBench task\n\n"
        "Decide whether the triggering event requires a user-facing documentation "
        "update. If it does, edit only the `docs/` repository. Otherwise, abstain.\n\n"
        f"{code_note}"
        "## Context\n\n"
        f"{item['context'].strip()}\n"
    )


def prepare_item(
    item: dict[str, Any],
    output_dir: Path,
    *,
    offline: bool = False,
) -> dict[str, Any]:
    """Prepare one item without copying upstream identities into the workspace."""
    frozen = research_input(item)
    try:
        source_attestation = attest_item_sources(item, offline=offline)
    except SourceAttestationError as exc:
        raise WorkspacePreparationError(str(exc)) from exc

    workspace_key = secrets.token_hex(12)
    workspace_root = output_dir / "workspaces"
    attestation_root = output_dir / "attestations"
    destination = workspace_root / f"workspace-{workspace_key}"
    workspace_root.mkdir(parents=True, exist_ok=True)
    attestation_root.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".workspace.", dir=workspace_root))
    try:
        protected_source_prose: list[str] = []
        for component in source_attestation.get("components") or []:
            protected_source_prose.extend(component.pop("_protected_source_prose", []))
        docs_result = _prepare_docs(item["docs"], staging / "docs")
        gold_diff = docs_result.pop("human_diff")
        context_result = context_audit(
            item["context"],
            item,
            gold_diff,
            protected_source_prose=protected_source_prose,
        )
        if not context_result["ok"]:
            codes = ", ".join(row["code"] for row in context_result["findings"])
            raise WorkspacePreparationError(
                f"context contamination audit failed for {item['instance_id']}: {codes}"
            )

        code_result: dict[str, Any] | None = None
        code_summary: dict[str, str] | None = None
        if item["code"] is not None:
            code_result = _prepare_code(item["code"], staging / "code")
            code_summary = {
                "path": "code",
                "base_commit": code_result["base_commit"],
                "head_commit": code_result["head_commit"],
            }

        task_bytes = (
            frozen["prompt"].encode("utf-8") if frozen is not None
            else _task_markdown(item, has_code=code_summary is not None).encode()
        )
        if frozen is not None:
            verify_research_workspace(item, staging, task_bytes)
        workspace_payload = {
            "schema_version": "dogbench-anonymous-workspace-v1",
            "docs": {"path": "docs", "base_commit": docs_result["commit"]},
            "code": code_summary,
        }
        workspace_bytes = json.dumps(
            workspace_payload, indent=2, sort_keys=True
        ).encode() + b"\n"
        (staging / "TASK.md").write_bytes(task_bytes)
        (staging / "workspace.json").write_bytes(workspace_bytes)
        _assert_safe_symlinks(staging)
        staging.replace(destination)

        attestation = {
            "schema_version": "dogbench-workspace-attestation-v1",
            "workspace_key": workspace_key,
            "workspace": str(destination.relative_to(output_dir)),
            "item": item,
            "source_attestation": source_attestation,
            "context_audit": context_result,
            "artifacts": {
                "TASK.md": {"sha256": sha256_bytes(task_bytes), "bytes": len(task_bytes)},
                "workspace.json": {
                    "sha256": sha256_bytes(workspace_bytes),
                    "bytes": len(workspace_bytes),
                },
            },
            "docs": docs_result,
            "code": code_result,
        }
        if frozen is not None:
            attestation["research_input"] = input_binding(item)
        attestation["logical_input_sha256"] = sha256_bytes(
            canonical_json_bytes(attestation)
        )
        attestation_path = attestation_root / f"{workspace_key}.json"
        attestation_path.write_text(
            json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return {
        "instance_id": item["instance_id"],
        "path": str(destination),
        "attestation": str(attestation_path),
        "logical_input_sha256": attestation["logical_input_sha256"],
        "has_code": item["code"] is not None,
    }


def prepare_items(
    items: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    offline: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise WorkspacePreparationError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    index_path = output_dir / "index.json"
    try:
        prepared = [prepare_item(item, output_dir, offline=offline) for item in items]
        index_items = []
        for row in prepared:
            index_items.append(
                {
                    **row,
                    "path": str(Path(row["path"]).relative_to(output_dir)),
                    "attestation": str(
                        Path(row["attestation"]).relative_to(output_dir)
                    ),
                }
            )
        index = {
            "schema_version": "dogbench-workspace-index-v1",
            "items": index_items,
        }
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Import locally to avoid a module cycle: the verifier reuses the
        # low-level sealed-repository helpers defined above.
        from .verify import verify_output

        verification = verify_output(output_dir)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return {
        "prepared": len(prepared),
        "output": str(output_dir),
        "index": str(index_path),
        "source_attestation": "offline_explicit" if offline else "live_github",
        "verified": verification["verified"],
        "items": prepared,
    }
