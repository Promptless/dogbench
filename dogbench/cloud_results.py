"""Original managed-output SHA validation and constrained PR repair."""
from __future__ import annotations
import json
import os
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Any

def _expected_short(sha: str | None) -> str | None:
    return sha[:12] if sha else None


def _fetch_pr_metadata(pr_url: str, token: str) -> dict[str, Any] | None:
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        [
            "gh", "pr", "view", pr_url,
            "--json",
            "url,number,baseRefName,baseRefOid,headRefName,headRefOid",
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _fetch_first_pr_commit_parent_oid(
    *,
    repo: str,
    number: int | str,
    token: str,
    attempts: int = 8,
    retry_delay_s: float = 1.0,
) -> str | None:
    env = {**os.environ, "GH_TOKEN": token, "GITHUB_TOKEN": token}
    for attempt in range(max(1, attempts)):
        res = subprocess.run(
            [
                "gh", "api",
                f"repos/{repo}/pulls/{number}/commits?per_page=1",
            ],
            capture_output=True, text=True, env=env, check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            try:
                commits = json.loads(res.stdout)
            except json.JSONDecodeError:
                commits = None
            if isinstance(commits, list) and commits:
                parents = commits[0].get("parents")
                if isinstance(parents, list) and parents:
                    sha = parents[0].get("sha")
                    if isinstance(sha, str) and sha:
                        return sha
        # A just-created PR can be visible through `gh pr view` before its
        # commits endpoint is populated. Retry this availability gap; an
        # actual, populated parent mismatch is still returned and fails closed.
        if attempt + 1 < max(1, attempts):
            time.sleep(retry_delay_s)
    return None


def _fetch_commit_tree_oid(
    *,
    repo: str,
    oid: str | None,
    token: str,
) -> str | None:
    if not oid:
        return None
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        [
            "gh", "api",
            f"repos/{repo}/git/commits/{oid}",
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    tree = data.get("tree") if isinstance(data, dict) else None
    if not isinstance(tree, dict):
        return None
    sha = tree.get("sha")
    return sha if isinstance(sha, str) and sha else None


def validate_pr_sha_context(
    *,
    pr_url: str,
    token: str,
    expected_mirror_repo: str,
    expected_base_ref: str,
    expected_base_oid: str,
    allowed_base_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Verify an agent docs PR was opened against the exact prepared mirror.

    The mirror `main` branch is force-reset between items. A Git-connected
    service can also mutate `main` while a run is in flight. `gh pr view`
    reports the current base branch OID, so the durable invariant is the first
    parent of the PR's first commit: the actual commit the candidate patch was
    generated against.
    """
    expected = {
        "mirror_repo": expected_mirror_repo,
        "base_ref": expected_base_ref,
        "base_oid": expected_base_oid,
        "base_oid_short": _expected_short(expected_base_oid),
    }
    validation: dict[str, Any] = {
        "ok": False,
        "pr_url": pr_url,
        "expected": expected,
        "actual": {},
        "errors": [],
        "warnings": [],
    }
    marker = f"github.com/{expected_mirror_repo}/pull/"
    if marker.lower() not in pr_url.lower():
        validation["errors"].append({
            "code": "wrong_pr_repo",
            "message": f"PR URL is not under expected mirror {expected_mirror_repo}",
        })

    meta = _fetch_pr_metadata(pr_url, token)
    if not meta:
        validation["errors"].append({
            "code": "pr_metadata_unavailable",
            "message": "could not fetch PR metadata for SHA validation",
        })
        return validation

    actual = {
        "url": meta.get("url") or pr_url,
        "number": meta.get("number"),
        "base_ref": meta.get("baseRefName"),
        "current_base_oid": meta.get("baseRefOid"),
        "current_base_oid_short": _expected_short(meta.get("baseRefOid")),
        "head_ref": meta.get("headRefName"),
        "head_oid": meta.get("headRefOid"),
        "head_oid_short": _expected_short(meta.get("headRefOid")),
    }
    first_parent_oid = _fetch_first_pr_commit_parent_oid(
        repo=expected_mirror_repo,
        number=actual["number"],
        token=token,
    ) if actual["number"] else None
    actual["first_commit_parent_oid"] = first_parent_oid
    actual["first_commit_parent_oid_short"] = _expected_short(first_parent_oid)
    validation["actual"] = actual
    allowed_refs = allowed_base_refs or {expected_base_ref}
    if actual["base_ref"] not in allowed_refs:
        validation["errors"].append({
            "code": "base_ref_mismatch",
            "message": (
                f"PR base ref {actual['base_ref']!r} does not match "
                f"expected {sorted(allowed_refs)!r}"
            ),
        })
    if first_parent_oid is None:
        validation["errors"].append({
            "code": "first_commit_parent_unavailable",
            "message": "could not fetch the first parent of the PR's first commit",
        })
    elif first_parent_oid != expected_base_oid:
        expected_tree_oid = _fetch_commit_tree_oid(
            repo=expected_mirror_repo,
            oid=expected_base_oid,
            token=token,
        )
        first_parent_tree_oid = _fetch_commit_tree_oid(
            repo=expected_mirror_repo,
            oid=first_parent_oid,
            token=token,
        )
        actual["first_commit_parent_tree_oid"] = first_parent_tree_oid
        actual["expected_base_tree_oid"] = expected_tree_oid
        if expected_tree_oid and first_parent_tree_oid == expected_tree_oid:
            validation["warnings"].append({
                "code": "base_tree_equivalent",
                "message": (
                    "PR first-commit parent SHA differs from expected, but its tree "
                    "matches the expected benchmark base tree exactly."
                ),
            })
        else:
            validation["errors"].append({
                "code": "base_oid_mismatch",
                "message": (
                    f"PR first-commit parent SHA {actual['first_commit_parent_oid_short']} does not match "
                    f"expected {expected['base_oid_short']}"
                ),
            })
    validation["ok"] = not validation["errors"]
    return validation


def retarget_pr_to_expected_base_if_safe(
    *,
    pr_url: str,
    token: str,
    expected_base_ref: str,
    expected_base_oid: str,
    validation: dict[str, Any],
) -> bool:
    """Retarget a PR that was created from the pinned base but aimed at main."""
    error_codes = {error.get("code") for error in validation.get("errors", [])}
    actual = validation.get("actual") or {}
    if (
        error_codes != {"base_ref_mismatch"}
        or actual.get("first_commit_parent_oid") != expected_base_oid
    ):
        return False

    env = os.environ.copy()
    env["GH_TOKEN"] = token
    result = subprocess.run(
        ["gh", "pr", "edit", pr_url, "--base", expected_base_ref],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return result.returncode == 0


def republish_pr_after_protected_base_write_if_safe(
    *,
    pr_url: str,
    token: str,
    expected_mirror_repo: str,
    expected_base_ref: str,
    expected_base_oid: str,
    expected_head_ref: str,
    validation: dict[str, Any],
) -> str | None:
    """Preserve an agent commit that was pushed onto the pinned base ref.

    Some managed agents interpret "open against <base>" as permission to use
    that existing branch as their PR head. Repair only the exact, provably safe
    topology: the unexpected head is the protected base ref, its sole commit
    directly parents the frozen base OID, and ``main`` still points at that same
    frozen OID. The trusted controller copies the unchanged agent commit to the
    per-run work ref, restores the protected ref, closes the malformed PR, and
    opens a replacement PR from work to base.
    """
    error_codes = {error.get("code") for error in validation.get("errors", [])}
    actual = validation.get("actual") or {}
    head_oid = actual.get("head_oid")
    if (
        error_codes != {"base_ref_mismatch"}
        or actual.get("base_ref") != "main"
        or actual.get("current_base_oid") != expected_base_oid
        or actual.get("head_ref") != expected_base_ref
        or actual.get("first_commit_parent_oid") != expected_base_oid
        or not head_oid
        or expected_head_ref == expected_base_ref
    ):
        return None

    env = os.environ.copy()
    env["GH_TOKEN"] = token

    metadata = subprocess.run(
        ["gh", "pr", "view", pr_url, "--json", "title,body"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if metadata.returncode != 0:
        return None
    try:
        pr_meta = json.loads(metadata.stdout)
    except json.JSONDecodeError:
        return None

    create_ref = subprocess.run(
        [
            "gh", "api", "--method", "POST",
            f"repos/{expected_mirror_repo}/git/refs",
            "-f", f"ref=refs/heads/{expected_head_ref}",
            "-f", f"sha={head_oid}",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if create_ref.returncode != 0:
        return None

    restore_base = subprocess.run(
        [
            "gh", "api", "--method", "PATCH",
            f"repos/{expected_mirror_repo}/git/refs/heads/{expected_base_ref}",
            "-f", f"sha={expected_base_oid}",
            "-F", "force=true",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if restore_base.returncode != 0:
        return None

    closed = subprocess.run(
        ["gh", "pr", "close", pr_url],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if closed.returncode != 0:
        return None

    with tempfile.TemporaryDirectory(prefix="dogbench-pr-body-") as temporary:
        body_path = Path(temporary) / "body.md"
        body_path.write_text(str(pr_meta.get("body") or ""), encoding="utf-8")
        replacement = subprocess.run(
            [
                "gh", "pr", "create", "--repo", expected_mirror_repo,
                "--head", expected_head_ref, "--base", expected_base_ref,
                "--title", str(pr_meta.get("title") or "Documentation update"),
                "--body-file", str(body_path),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    if replacement.returncode != 0 or not replacement.stdout.strip():
        return None
    return replacement.stdout.strip().splitlines()[-1].strip()

