"""Frozen development prompts and source identities for the research local transport."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

class FrozenInputError(ValueError):
    """A released input differs from its frozen research binding."""

RESEARCH_LAYOUT = "research-local-v1"
REGISTRY = Path(__file__).with_name("assets") / "research_inputs.json"


@lru_cache(maxsize=1)
def input_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def item_digest(item: dict) -> str:
    return hashlib.sha256(
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def research_input(item: dict, *, required: bool = False) -> dict | None:
    record = input_registry()["items"].get(item["instance_id"])
    if record is None:
        if required:
            raise FrozenInputError(f"no frozen research input for {item['instance_id']}")
        return None
    if item_digest(item) != record["item_sha256"]:
        raise FrozenInputError(
            f"released item differs from frozen research input: {item['instance_id']}"
        )
    if hashlib.sha256(record["prompt"].encode()).hexdigest() != record["prompt_sha256"]:
        raise FrozenInputError(f"research prompt hash mismatch: {item['instance_id']}")
    return record


def verify_research_workspace(item: dict, workspace: Path, instructions: bytes) -> dict:
    """Reject changed prompts, missing files, or reference code from another input."""
    import subprocess

    frozen = research_input(item, required=True)
    if instructions != frozen["prompt"].encode():
        raise FrozenInputError("task instructions differ from the frozen research prompt")

    def git(repo: Path, *args: str) -> bytes:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
        if result.returncode:
            raise FrozenInputError(
                f"cannot verify research snapshot: {result.stderr.decode().strip()}"
            )
        return result.stdout

    for name, expected_tree in (
        ("docs", frozen["docs_base_tree"]),
        ("code", frozen["code_base_tree"]),
    ):
        repo = workspace / name
        if expected_tree is None:
            if repo.exists():
                raise FrozenInputError("unexpected reference code in research workspace")
            continue
        if not (repo / ".git").is_dir():
            raise FrozenInputError(f"research {name} snapshot is missing Git metadata")
        revision = "HEAD" if name == "docs" else "HEAD^"
        if git(repo, "rev-parse", f"{revision}^{{tree}}").decode().strip() != expected_tree:
            raise FrozenInputError(f"{name} base differs from frozen research input")
        if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
            raise FrozenInputError(f"research {name} snapshot has missing or changed files")
        if git(repo, "remote").strip():
            raise FrozenInputError(f"research {name} snapshot must have no remotes")
        expected_count = "1" if name == "docs" else "2"
        if git(repo, "rev-list", "--count", "HEAD").decode().strip() != expected_count:
            raise FrozenInputError(f"unexpected history in research {name} snapshot")
        if name == "code":
            paths = set(
                git(repo, "diff", "--name-only", "--no-renames", "-z", "HEAD^", "HEAD")
                .decode()
                .strip("\0")
                .split("\0")
            )
            if paths != set(frozen["code_blob_sha256"]):
                raise FrozenInputError("reference code path set differs from research")
            for path, hashes in frozen["code_blob_sha256"].items():
                for revision, key in (("HEAD^", "base_sha256"), ("HEAD", "head_sha256")):
                    expected = hashes[key]
                    if expected is None:
                        if git(repo, "ls-tree", revision, "--", path).strip():
                            raise FrozenInputError(f"unexpected code blob: {path}")
                    elif (
                        hashlib.sha256(git(repo, "show", f"{revision}:{path}")).hexdigest()
                        != expected
                    ):
                        raise FrozenInputError(f"reference code blob differs from research: {path}")
    return frozen


def input_binding(item: dict) -> dict | None:
    """Metadata tying a prepared item to the shipped research input export."""
    frozen = research_input(item)
    if frozen is None:
        return None
    return {
        "profile": RESEARCH_LAYOUT,
        "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        "item_sha256": frozen["item_sha256"],
        "prompt_sha256": frozen["prompt_sha256"],
        "research_logical_input_sha256": frozen["logical_input_sha256"],
    }


def local_execution_prompt(item: dict, workspace: Path) -> str:
    """Use exact frozen bytes for released items; custom inputs use the renderer."""
    if research_input(item) is not None:
        prompt = (workspace / "TASK.md").read_bytes()
        verify_research_workspace(item, workspace, prompt)
        return prompt.decode("utf-8")
    from .prompts import prepared_local_prompt
    return prepared_local_prompt(item["context"], has_code=bool(item.get("code")))
