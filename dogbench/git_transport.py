"""Shared diff-filtering helpers + a thin `gh repo create` wrapper.

Library-only. The standalone fork CLI that used to live here was replaced by
`dogbench.cloud_mirrors.Mirror`; the remaining helpers are imported by `mirror.py`,
`run_candidate.py`, and `resolve_pr_inputs.py`.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# Files with these extensions are always documentation
DOC_TEXT_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc", ".ipynb"}

# Files whose stem matches these names are documentation
DOC_HINT_FILENAMES = {"readme", "changelog", "contributing", "release", "security", "code_of_conduct"}

# Top-level (or any-level) directories whose contents are examples / docs scaffolding,
# not library code. Any file path whose first segment matches one of these is excluded.
EXAMPLE_DIRS = {
    "docs_src", "doc_src",          # fastapi-style code-in-docs
    "examples", "example",
    "notebooks", "notebook",
    "samples", "sample",
    "demos", "demo",
    "cookbook", "tutorials",
    "docs", "doc",                  # generic docs trees
    "open-api", "openapi",          # generated API/spec docs (reviewable signal)
}

PR_URL_RE = re.compile(r"https?://github\.com/([\w.\-]+/[\w.\-]+)/pull/(\d+)", re.IGNORECASE)


def is_non_code_file(path: str) -> bool:
    """Return True if this file should be excluded from the code-only PR."""
    normalized = path.strip("/").lower()
    parts = normalized.split("/")
    filename = parts[-1]
    stem = Path(filename).stem.lower()
    suffix = Path(filename).suffix.lower()

    if stem in DOC_HINT_FILENAMES or filename.startswith("readme."):
        return True
    if suffix in DOC_TEXT_EXTENSIONS:
        return True

    # Example / docs scaffolding directories (check every path segment)
    for part in parts[:-1]:
        if part in EXAMPLE_DIRS:
            return True

    return False


def filter_diff_to_code_only(full_diff: str) -> str:
    """Given a full unified diff (with `diff --git` headers), return only the
    hunks for non-doc, non-example files."""
    chunks: list[str] = []
    current: list[str] = []

    for line in full_diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                chunks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("".join(current))

    kept: list[str] = []
    excluded: list[str] = []
    for chunk in chunks:
        file_path = None
        for line in chunk.splitlines():
            if line.startswith("+++ b/"):
                file_path = line[6:].rstrip("\n")
                break
            if line.startswith("+++ /dev/null"):
                file_path = None
                break
        if file_path is None:
            for line in chunk.splitlines():
                if line.startswith("diff --git "):
                    parts = line.split(" b/", 1)
                    if len(parts) == 2:
                        file_path = parts[1].rstrip("\n")
                    break
        if file_path and is_non_code_file(file_path):
            excluded.append(file_path)
        else:
            kept.append(chunk)

    if excluded:
        print(f"[fork] Excluded {len(excluded)} non-code files from diff:")
        for f in excluded:
            print(f"[fork]   - {f}")
    print(f"[fork] Kept {len(kept)} code file(s) in diff")
    return "".join(kept)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run` with `check=True`, token-masked logging."""
    display = [c if "x-access-token" not in c.lower() else "https://x-access-token:***@..." for c in cmd]
    print(f"[fork] $ {' '.join(display)}")
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        print(f"[fork] command failed rc={exc.returncode}: {' '.join(display)}")
        if exc.stdout:
            print("[fork] stdout:")
            print(exc.stdout.rstrip())
        if exc.stderr:
            print("[fork] stderr:")
            print(exc.stderr.rstrip())
        raise


def create_remote_repo(new_repo: str, visibility: str, description: str, *, token: str | None = None) -> str:
    cmd = ["gh", "repo", "create", new_repo, f"--{visibility}", "--description", description]
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    result = run(cmd, capture_output=True, text=True, env=env)
    url = result.stdout.strip()
    print(f"[fork] Created repo: {url}")
    return url
