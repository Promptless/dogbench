"""Shared mirror repo primitives.

A `Mirror` is a private GitHub repo under an explicitly supplied owner
whose `main` is reset to whatever SHA we point it at. Horizontal evaluations
can share one mirror per source repo across agents. Production vertical runs
use one dedicated mirror per upstream GitHub org per candidate source, so
agents or model tuples stay in separate lanes while each lane can process
multiple docs/code repos from the same org.

**Mirror naming is a UUID** — the GitHub repo name itself must NOT identify
the upstream project (e.g., `eval-mirror-fastapi` would leak `fastapi` to an
agent that calls `gh repo view`). The mapping from source repo or
`org:<github_org>#<candidate_source>` → mirror UUID is persisted in
the explicitly configured mirror index so subsequent runs reuse the same
neutral mirror.

The mirror is kept in two local clones under WORK_DIR (paths are local-only,
not visible to the agent, so they may use the UUID for clarity). Org-scoped
mirrors keep one source clone per repo:
  <WORK_DIR>/<mirror_uuid>/orig or orig__<owner>__<repo> — source clone(s)
  <WORK_DIR>/<mirror_uuid>/mirror                       — clone of the mirror repo

`Mirror.set_to_sha(sha)` works forward AND backward — by default it replaces
`main` with a neutral orphan snapshot of that SHA in the source clone, then
force-pushes. Mintlify can opt into a linear snapshot mode that commits the
next snapshot on top of the previous mirror `main`, letting GitHub/Mintlify see
the incremental tree diff while still keeping neutral commit messages.

In production vertical runs, mirrors should be scoped to one UUID per
`GitHub org + candidate source`, not one UUID per docs repo. In that mode the
mirror keeps one local source clone per repo under `orig__<owner>__<repo>/`,
then assembles the right docs/code trees into the single per-source mirror for
each item.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .git_transport import (
    create_remote_repo,
    filter_diff_to_code_only,
    is_non_code_file,
    run,
)

# Explicit operator storage. No research checkout paths or account defaults.
_MIRROR_PATHS: dict[str, Path] = {}


def configure_mirrors(*, index_path: Path, mirror_overlays_root: Path,
                      source_overlays_root: Path) -> None:
    """Set controller-only storage before using the mirror index or overlays.

    Empty overlay directories are valid. This function performs no I/O and does
    not create accounts, repositories, directories, or connections.
    """
    _MIRROR_PATHS.update(
        index=Path(index_path).expanduser().resolve(),
        mirror_overlays=Path(mirror_overlays_root).expanduser().resolve(),
        source_overlays=Path(source_overlays_root).expanduser().resolve(),
    )


def _configured_path(name: str) -> Path:
    try:
        return _MIRROR_PATHS[name]
    except KeyError:
        raise RuntimeError("call configure_mirrors() with explicit index and overlay paths first") from None


MINTLIFY_CONFIG_PATHS = (
    Path("docs.json"),
    Path("mint.json"),
    Path("docs") / "docs.json",
    Path("docs") / "mint.json",
    Path("docs") / "mintlify" / "docs.json",
    Path("site") / "docs.json",
    Path("site") / "mint.json",
)
DOC_DISCOVERY_SKIP_DIRS = {
    ".git",
    ".github",
    ".tox",
    ".venv",
    "__pycache__",
    "_build",
    "_code_repo",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
}


def _load_mirror_index() -> dict[str, str]:
    if not _configured_path("index").exists():
        return {}
    try:
        return json.loads(_configured_path("index").read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_mirror_index(idx: dict[str, str]) -> None:
    _configured_path("index").parent.mkdir(parents=True, exist_ok=True)
    _configured_path("index").write_text(json.dumps(idx, indent=2, sort_keys=True))


def _mirror_index_key(source_repo: str, agent: str | None) -> str:
    """Index key for a mirror.

    `agent=None` → the bare source repo (the shared mirror used by the
    horizontal "one PR, all agents" runner; preserves the compatibility flat entries).
    `agent="codex"` → a per-source mirror `"<source_repo>#codex"`, the legacy
    vertical-run key. Production vertical runs should prefer the org-scoped key
    (`org:<github_org>#codex`) so multiple docs repos in one GitHub org are
    assembled into the same per-source mirror. '#' never appears in a repo slug
    or agent name.
    """
    return source_repo if agent is None else f"{source_repo}#{agent.strip().lower()}"


def _mirror_org_index_key(github_org: str, agent: str) -> str:
    """Index key for the production vertical runner.

    A benchmark org can have multiple docs repos (for example Mautic has user
    and developer docs). Candidate sources should see one persistent mirror per
    upstream GitHub org, not one per docs repo, so cross-repo tasks can be
    assembled into the same neutral UUID mirror.
    """
    return f"org:{github_org.strip()}#{agent.strip().lower()}"


def get_mirror_name_for(source_repo: str, agent: str | None = None) -> str | None:
    """Look up the UUID mirror name for a (source repo, agent), or None if none
    assigned yet. `agent=None` is the shared mirror."""
    return _load_mirror_index().get(_mirror_index_key(source_repo, agent))


def get_org_mirror_name_for(github_org: str, agent: str) -> str | None:
    """Look up the UUID mirror name for a (GitHub org, agent), if assigned."""
    return _load_mirror_index().get(_mirror_org_index_key(github_org, agent))


def assign_mirror_name(source_repo: str, agent: str | None = None) -> str:
    """Generate a fresh UUID mirror name and persist the mapping. Idempotent —
    returns the existing assignment if one already exists. `agent=None` is the
    shared mirror; pass an agent for its own dedicated mirror."""
    idx = _load_mirror_index()
    key = _mirror_index_key(source_repo, agent)
    if key in idx:
        return idx[key]
    name = str(uuid.uuid4())
    idx[key] = name
    _save_mirror_index(idx)
    return name


def assign_org_mirror_name(github_org: str, agent: str) -> str:
    """Generate or reuse a UUID mirror name for a (GitHub org, agent)."""
    idx = _load_mirror_index()
    key = _mirror_org_index_key(github_org, agent)
    if key in idx:
        return idx[key]
    name = str(uuid.uuid4())
    idx[key] = name
    _save_mirror_index(idx)
    return name


def _q(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command quietly, raising on nonzero. Returns the completed process."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _maybe(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, do NOT raise on nonzero."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _copy_git_tree_objects(
    source_repo: Path,
    target_repo: Path,
    tree_oid: str,
) -> None:
    """Copy exactly one Git tree graph without importing source history.

    Materializing tracked files through a worktree is not byte preserving:
    checkout filters and ``eol`` attributes can rewrite a historical blob
    before it is added to the synthetic snapshot.  Pack the frozen tree and
    its reachable blobs directly into the target object database instead.
    """
    pack = subprocess.Popen(
        ["git", "-C", str(source_repo), "pack-objects", "--stdout", "--revs"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pack.stdin is not None
    assert pack.stdout is not None
    assert pack.stderr is not None
    index = subprocess.Popen(
        ["git", "-C", str(target_repo), "index-pack", "--stdin"],
        stdin=pack.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pack.stdout.close()
    pack.stdin.write(f"{tree_oid}\n".encode("ascii"))
    pack.stdin.close()
    _index_stdout, index_stderr = index.communicate()
    pack_stderr = pack.stderr.read()
    pack.stderr.close()
    pack_rc = pack.wait()
    if pack_rc != 0 or index.returncode != 0:
        details = (pack_stderr + index_stderr).decode("utf-8", "replace")[:400]
        raise RuntimeError(f"failed to copy frozen tree objects: {details}")


def _delete_local_branch_conflicts(repo_dir: Path, branch_name: str) -> None:
    """Delete local branches that conflict with creating ``branch_name``.

    Git stores branch refs as files/directories. A legacy branch named
    ``candidate/code`` prevents creating ``candidate/code/<item>`` because the
    file at ``refs/heads/candidate/code`` blocks the directory. Old mirror
    clones can retain those local and remote-tracking refs even after remote
    branch cleanup, so remove the target branch and all prefix refs before
    creating a nested name.
    """
    prefixes: list[str] = []
    parts = branch_name.split("/")
    for i in range(1, len(parts)):
        prefixes.append("/".join(parts[:i]))
    for name in [branch_name, *prefixes]:
        _maybe(["git", "-C", str(repo_dir), "branch", "-D", name])
        _maybe(["git", "-C", str(repo_dir), "update-ref", "-d", f"refs/remotes/origin/{name}"])


@dataclass
class Mirror:
    source_repo: str         # e.g. "tiangolo/fastapi"
    mirror_repo: str         # e.g. "YOUR_ORG/neutral-mirror-uuid"
    orig_dir: Path
    mirror_dir: Path
    token: str

    # ─────────────────────────────────────────────────────────────────────
    # Git committer identity for mirror-side commits
    # ─────────────────────────────────────────────────────────────────────
    #
    # The mirror's commits show up in places agents can see — `git log` on
    # their clone, Mintlify's activity dashboard ("doc-bench / snapshot / 406
    # files added"), GitHub's commit list. A literal "doc-bench" name leaks
    # the pipeline's identity. Using the mirror's own UUID instead makes the
    # committer name self-referential and uninformative.

    @property
    def committer_name(self) -> str:
        return self.mirror_repo.split("/", 1)[-1]

    @property
    def committer_email(self) -> str:
        return f"{self.committer_name}@noreply.invalid"

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def ensure(
        cls,
        source_repo: str,
        token: str,
        *,
        org: str,
        work_dir: Path,
        mirror_name: str | None = None,
        agent: str | None = None,
        mirror_scope: str = "source_repo",
        mirror_github_org: str | None = None,
    ) -> "Mirror":
        """Create the mirror repo + local clones if they don't exist yet.
        Idempotent — safe to call repeatedly.

        If `mirror_name` is None, the name is read from (or assigned to)
        the explicitly configured mirror index so repeated runs reuse the
        same UUID-named mirror. By default the key is (source repo, agent).
        With `mirror_scope="github_org"`, the key is (GitHub org, agent), so
        multiple docs repos in the same upstream org are assembled into one
        per-agent mirror.

        `agent=None` resolves the shared per-repo mirror (horizontal "one PR,
        all agents" mode). Pass an agent (e.g. "codex") to resolve that agent's
        OWN dedicated mirror — the vertical "one agent over a repo's 25 PRs"
        mode, where each agent's repo is private to it so no agent can see
        another's PRs and no destructive recreate is needed between runs.
        """
        if not org or not re.fullmatch(r"[A-Za-z0-9_.-]+", org):
            raise ValueError("org must be an explicit GitHub owner")
        if not token:
            raise ValueError("an explicit controller GitHub token is required")
        work_dir = Path(work_dir).expanduser().resolve()
        if mirror_scope not in {"source_repo", "github_org"}:
            raise ValueError(f"unknown mirror_scope={mirror_scope!r}")
        if mirror_scope == "github_org" and not agent:
            raise ValueError("mirror_scope='github_org' requires an agent")

        if mirror_name is None:
            if mirror_scope == "github_org":
                github_org = mirror_github_org or source_repo.split("/", 1)[0]
                mirror_name = (get_org_mirror_name_for(github_org, agent or "")
                               or assign_org_mirror_name(github_org, agent or ""))
            else:
                mirror_name = (get_mirror_name_for(source_repo, agent)
                               or assign_mirror_name(source_repo, agent))
        mirror_repo = f"{org}/{mirror_name}"
        base = work_dir / mirror_name
        if mirror_scope == "github_org":
            orig_dir = base / f"orig__{source_repo.replace('/', '__')}"
        else:
            orig_dir = base / "orig"
        mirror_dir = base / "mirror"
        base.mkdir(parents=True, exist_ok=True)

        # 1. Ensure the GitHub mirror repo exists (private). The description
        # is neutral — agents that inspect the repo via `gh repo view` must
        # not learn the upstream source from the mirror's metadata.
        chk = subprocess.run(["gh", "repo", "view", mirror_repo], capture_output=True, text=True, check=False, env={**os.environ, "GH_TOKEN": token})
        if chk.returncode != 0:
            create_remote_repo(mirror_repo, "private", "internal project", token=token)

        # 2. Ensure the local clone of the ORIGINAL source repo exists.
        if not (orig_dir / ".git").exists():
            print(f"[mirror] cloning source {source_repo} → {orig_dir}")
            run([
                "git", "clone",
                f"https://x-access-token:{token}@github.com/{source_repo}.git",
                str(orig_dir),
            ])
        else:
            _maybe(["git", "-C", str(orig_dir), "fetch", "--all", "--quiet", "--tags"])

        # 3. Ensure the local clone of the MIRROR repo exists. If the remote
        # mirror is empty, the initial clone will warn but still produce a
        # working tree; we handle that on first push in set_to_sha.
        if not (mirror_dir / ".git").exists():
            print(f"[mirror] cloning mirror {mirror_repo} → {mirror_dir}")
            r = _maybe([
                "git", "clone",
                f"https://x-access-token:{token}@github.com/{mirror_repo}.git",
                str(mirror_dir),
            ])
            if r.returncode != 0 or not (mirror_dir / ".git").exists():
                # Empty repo — git clone may exit 0 but produce an empty dir
                # OR exit nonzero. Initialize ourselves and add origin.
                mirror_dir.mkdir(parents=True, exist_ok=True)
                run(["git", "-C", str(mirror_dir), "init", "-b", "main"])
                run([
                    "git", "-C", str(mirror_dir), "remote", "add", "origin",
                    f"https://x-access-token:{token}@github.com/{mirror_repo}.git",
                ])
            # Stable identity so commits don't fail; use the mirror's own
            # UUID rather than "doc-bench" so the committer name doesn't
            # leak that this repo is part of a pipeline.
            run(["git", "-C", str(mirror_dir), "config", "user.email",
                 f"{mirror_name}@noreply.invalid"])
            run(["git", "-C", str(mirror_dir), "config", "user.name", mirror_name])
        else:
            # Refresh credentials in the remote URL — token may have changed.
            run([
                "git", "-C", str(mirror_dir), "remote", "set-url", "origin",
                f"https://x-access-token:{token}@github.com/{mirror_repo}.git",
            ])

        return cls(
            source_repo=source_repo,
            mirror_repo=mirror_repo,
            orig_dir=orig_dir,
            mirror_dir=mirror_dir,
            token=token,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hard reset: delete and recreate the mirror repo
    # ─────────────────────────────────────────────────────────────────────

    def recreate(self) -> "Mirror":
        """Delete the GitHub mirror repo (which also nukes all closed PRs and
        their content), then recreate it as an empty repo. The UUID-named
        mapping in the explicitly configured mirror index is preserved so all per-mirror
        overlays and dashboard pointings (e.g. Mintlify) still apply.

        Use between agent runs in a batch when you want each agent to face a
        virgin mirror with no prior PR history to crib from.
        """
        env = os.environ.copy()
        env["GH_TOKEN"] = self.token
        del_res = subprocess.run(
            ["gh", "repo", "delete", self.mirror_repo, "--yes"],
            capture_output=True, text=True, env=env, check=False,
        )
        if del_res.returncode != 0:
            print(f"[mirror] warn: gh repo delete failed: {del_res.stderr[:200]}")
        # Wipe the local mirror clone — its remote no longer exists.
        if self.mirror_dir.exists():
            shutil.rmtree(self.mirror_dir)
        # Recreate the GitHub repo and re-clone locally.
        return Mirror.ensure(
            source_repo=self.source_repo,
            token=self.token,
            org=self.mirror_repo.split("/", 1)[0],
            work_dir=self.mirror_dir.parents[1],  # parent of <uuid>/mirror
            mirror_name=self.mirror_repo.split("/", 1)[1],
        )

    def reset_to_base(self, sha: str) -> str:
        """Non-destructive sibling of recreate(): return the mirror to a virgin
        state WITHOUT deleting the GitHub repo.

        Closes open PRs, deletes every non-main branch, then hard-resets main to
        the tree at `sha`. Composes the existing cleanup_pending() + set_to_sha()
        primitives — the same per-agent isolation run_candidate already performs,
        minus recreate()'s repo delete.

        Why prefer this over recreate() in a batch: deleting the GitHub repo
        changes its identity and orphans anything bound to it — most importantly
        a Promptless doc collection (keyed to the repo URL). That collection's
        analysis is expensive (minutes) and is meant to be provisioned ONCE per
        repo and reused across all ~25 of that repo's PRs via the
        operator-managed collection cache. recreate() would force a fresh
        analysis every run; reset_to_base() keeps the repo (and the collection)
        alive. To advance to the next PR, just call it again with the next base
        SHA — no repo churn.

        Caveat vs recreate(): closed PRs from prior runs remain in the repo's PR
        history (a PR can't be hard-deleted via the API; its branch is removed).
        main is clean with no open PRs/branches, which is what verdict
        extraction and per-agent isolation depend on.

        Returns the new HEAD SHA on the mirror.
        """
        self.cleanup_pending()
        return self.set_to_sha(sha)

    # ─────────────────────────────────────────────────────────────────────
    # Fetch a SHA into the source clone (e.g. PR head from a fork)
    # ─────────────────────────────────────────────────────────────────────

    def ensure_sha(self, sha: str) -> None:
        """Make `sha` available in `orig_dir`. For PRs from forks the head
        SHA isn't reachable from any branch of the upstream; in that case
        `git fetch origin <sha>` works on GitHub because GitHub enables
        uploadpack.allowReachableSHA1InWant. We also try the PR refspec as a
        fallback for older servers."""
        chk = _maybe(["git", "-C", str(self.orig_dir), "cat-file", "-e", f"{sha}^{{commit}}"])
        if chk.returncode == 0:
            return
        r = _maybe(["git", "-C", str(self.orig_dir), "fetch", "origin", sha])
        if r.returncode == 0:
            chk = _maybe(["git", "-C", str(self.orig_dir), "cat-file", "-e", f"{sha}^{{commit}}"])
            if chk.returncode == 0:
                return
        # Last resort: try fetching all PR heads. Expensive but works.
        _maybe([
            "git", "-C", str(self.orig_dir), "fetch", "origin",
            "+refs/pull/*/head:refs/remotes/origin/pr/*",
        ])
        chk = _maybe(["git", "-C", str(self.orig_dir), "cat-file", "-e", f"{sha}^{{commit}}"])
        if chk.returncode != 0:
            raise RuntimeError(f"could not fetch {sha} into {self.orig_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Core: set mirror main to any SHA
    # ─────────────────────────────────────────────────────────────────────

    def set_to_sha(
        self,
        sha: str,
        *,
        linear_history: bool = False,
        sanitize_for_managed_github: bool = True,
    ) -> str:
        """Hard-reset the mirror's `main` to match the tree at `sha` in the
        source repo. Force-pushes. Works whether the target SHA is ahead of,
        behind, or unrelated to the current mirror main.

        With ``linear_history=True``, commit the new snapshot on top of the
        current mirror ``main`` when one exists. Mintlify uses GitHub push
        history to decide how much to sync; keeping a parent chain lets it see
        the real tree diff instead of treating every update as a disconnected
        full snapshot.

        ``sanitize_for_managed_github`` is for a GitHub-hosted vessel that a
        managed agent can inspect.  It strips active GitHub automation,
        applies managed-only overlays, and scrubs direct upstream lookup
        identifiers.  A local brokered agent uses an inert, no-egress,
        no-GitHub-credential snapshot instead, so ``False`` preserves the
        exact frozen source tree (including ``.github`` and metadata files).

        Returns the new HEAD SHA on the mirror.
        """
        # 0. Verify SHA exists in source clone (fetch if missing).
        chk = _maybe(["git", "-C", str(self.orig_dir), "cat-file", "-e", f"{sha}^{{commit}}"])
        if chk.returncode != 0:
            print(f"[mirror] {sha} not in source clone; fetching…")
            _maybe(["git", "-C", str(self.orig_dir), "fetch", "origin", sha])
            chk = _maybe(["git", "-C", str(self.orig_dir), "cat-file", "-e", f"{sha}^{{commit}}"])
            if chk.returncode != 0:
                raise RuntimeError(f"SHA {sha} not found in {self.source_repo}")

        linear_parent = False
        if linear_history:
            _maybe(["git", "-C", str(self.mirror_dir), "fetch", "--quiet", "origin", "main"])
            origin_main = _maybe([
                "git", "-C", str(self.mirror_dir),
                "rev-parse", "--verify", "origin/main^{commit}",
            ])
            local_main = _maybe([
                "git", "-C", str(self.mirror_dir),
                "rev-parse", "--verify", "main^{commit}",
            ])
            start_point = "origin/main" if origin_main.returncode == 0 else (
                "main" if local_main.returncode == 0 else None
            )
            if start_point:
                checkout = _maybe([
                    "git", "-C", str(self.mirror_dir),
                    "checkout", "--quiet", "--force", "-B", "main", start_point,
                ])
                if checkout.returncode != 0:
                    raise RuntimeError(
                        "git checkout main for linear snapshot failed: "
                        f"{checkout.stderr[:400]}"
                    )
                linear_parent = True
            else:
                print("[mirror] no existing main found; creating first snapshot without parent")

        # 1. Copy the tree from orig@sha into the mirror working dir. We use
        # `git archive | tar -x` because the mirror's history is independent
        # of the source repo's — we don't want to import source commits, just
        # the tree state. Clear the mirror working dir first (preserve .git).
        mode = "linear snapshot" if linear_parent else "orphan snapshot"
        print(f"[mirror] resetting {self.mirror_repo} main → {self.source_repo}@{sha[:10]} ({mode})")
        for entry in self.mirror_dir.iterdir():
            if entry.name == ".git":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        # A local broker must retain the exact frozen Git tree. Do not pass
        # that tree through checkout-index followed by git add: repository
        # attributes can normalize historical blobs (for example CRLF to LF)
        # and silently change the resulting tree. Import only the source tree
        # graph, create a neutral parentless commit from it, and push that
        # commit to the local bare origin. The upstream commit and its history
        # remain unreachable from the agent-visible branch.
        if not sanitize_for_managed_github and not linear_parent:
            source_tree_oid = _q([
                "git", "-C", str(self.orig_dir),
                "rev-parse", f"{sha}^{{tree}}",
            ]).stdout.strip()
            _copy_git_tree_objects(
                self.orig_dir, self.mirror_dir, source_tree_oid,
            )
            _q([
                "git", "-C", str(self.mirror_dir),
                "checkout", "--quiet", "--orphan", "_tmp_reset",
            ])
            commit = _q([
                "git", "-C", str(self.mirror_dir),
                "-c", f"user.email={self.committer_email}",
                "-c", f"user.name={self.committer_name}",
                "commit-tree", source_tree_oid, "-m", "snapshot",
            ]).stdout.strip()
            _q([
                "git", "-C", str(self.mirror_dir),
                "reset", "--quiet", "--hard", commit,
            ])
            _maybe(["git", "-C", str(self.mirror_dir), "branch", "-D", "main"])
            _q(["git", "-C", str(self.mirror_dir), "branch", "-m", "main"])
            committed_tree_oid = _q([
                "git", "-C", str(self.mirror_dir),
                "rev-parse", "HEAD^{tree}",
            ]).stdout.strip()
            if committed_tree_oid != source_tree_oid:
                raise RuntimeError(
                    "exact local snapshot tree changed while creating neutral commit"
                )
            _q([
                "git", "-C", str(self.mirror_dir),
                "push", "origin", "main", "--force",
            ])
            print(f"[mirror] main now at {commit[:10]}")
            return commit

        # Populate from the source tree's index, rather than ``git archive``.
        # Archive obeys ``export-ignore`` attributes, which silently omits
        # tracked files and makes an exact local broker snapshot fail its tree
        # provenance check. checkout-index materializes every tracked entry
        # with its Git mode and without importing source history into the
        # synthetic mirror.
        source_index = self.mirror_dir / ".git" / "docbench-source.index"
        source_index.unlink(missing_ok=True)
        source_env = os.environ.copy()
        source_env["GIT_INDEX_FILE"] = str(source_index)
        source_tree = subprocess.run(
            ["git", "-C", str(self.orig_dir), "ls-tree", "-r", "-z", sha],
            check=True,
            capture_output=True,
        ).stdout
        gitlinks: list[tuple[str, str]] = []
        for entry in source_tree.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, _kind, object_id = metadata.split()
            if mode == b"160000":
                gitlinks.append((object_id.decode("ascii"), raw_path.decode("utf-8")))
        try:
            subprocess.run(
                [
                    "git", "-c", "core.sparseCheckout=false", "-C",
                    str(self.orig_dir), "read-tree", sha,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=source_env,
            )
            subprocess.run(
                [
                    "git", "-c", "core.sparseCheckout=false", "-C",
                    str(self.orig_dir), "checkout-index",
                    "--all", "--force", "--ignore-skip-worktree-bits",
                    f"--prefix={self.mirror_dir}/",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=source_env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"failed to materialize tree from {sha} into mirror") from exc
        finally:
            source_index.unlink(missing_ok=True)

        if sanitize_for_managed_github:
            # Regenerate a Mintlify `docs.json` from the mirror's current
            # `mkdocs.yml`, if there is one. The managed Mintlify agent needs
            # docs.json to know the nav structure; the upstream tree may not
            # ship one.
            self._regen_mintlify_docs_json()

            # A GitHub-hosted vessel must never run upstream automation. In
            # particular `.github/dependabot.yml` can create dependency PRs,
            # which pollutes history and can expose upstream details. Removing
            # the complete directory also prevents inherited workflows/actions
            # from running. Local agents use an inert snapshot and retain it.
            gh_dir = self.mirror_dir / ".github"
            if gh_dir.exists():
                shutil.rmtree(gh_dir)

            # Apply stable per-source transport overlays first, followed by
            # any per-mirror overlay. The source-scoped layer survives the
            # fresh UUID mirror allocation used by isolated managed-agent
            # retries; the mirror-scoped layer remains available for one-off
            # converted docs configurations. These integration-specific
            # changes must not alter the exact local source snapshot.
            mirror_name = self.mirror_repo.split("/", 1)[-1]
            source_overlay_name = self.source_repo.replace("/", "__")
            overlay_dirs = (
                _configured_path("source_overlays") / source_overlay_name,
                _configured_path("mirror_overlays") / mirror_name,
            )
            for overlay_dir in overlay_dirs:
                if not overlay_dir.is_dir():
                    continue
                for src in overlay_dir.rglob("*"):
                    if src.is_file():
                        rel = src.relative_to(overlay_dir)
                        dest = self.mirror_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                print(f"[mirror] applied overlay from {overlay_dir}")

            # Scrub direct upstream lookup identifiers from text-only metadata
            # files for managed GitHub agents. Source code remains untouched.
            _scrub_upstream_identifiers(
                self.mirror_dir, self.source_repo, self.mirror_repo,
            )

        # 2. Commit as a single snapshot. Most agents use a fresh orphan main
        # so their mirror history stays one-commit simple. Mintlify can opt into
        # a linear parent chain so GitHub/Mintlify observe the incremental tree
        # diff instead of a disconnected full snapshot on every PR.
        if not linear_parent:
            _q(["git", "-C", str(self.mirror_dir), "checkout", "--quiet", "--orphan", "_tmp_reset"])
        # The synthetic orphan index starts empty. Some paths tracked by the
        # source (notably generated artifacts) also match .gitignore, so a
        # normal add would silently omit them and change the snapshot tree.
        _q(["git", "-C", str(self.mirror_dir), "add", "-A", "--force"])
        # checkout-index cannot materialize a gitlink as a normal file. Restore
        # those entries explicitly so the synthetic local snapshot has the
        # exact source tree, including uninitialized submodules.
        for object_id, path in gitlinks:
            _q([
                "git", "-C", str(self.mirror_dir), "update-index", "--add",
                "--cacheinfo", f"160000,{object_id},{path}",
            ])
        # Neutral message — the source repo and SHA must not leak to the
        # agent via `git log` on the mirror.
        msg = "snapshot"
        commit = _maybe([
            "git", "-C", str(self.mirror_dir),
            "-c", f"user.email={self.committer_email}",
            "-c", f"user.name={self.committer_name}",
            "commit", "-m", msg, "--allow-empty",
        ])
        if commit.returncode != 0:
            raise RuntimeError(f"git commit failed: {commit.stderr[:400]}")

        # Replace local main for orphan snapshots; linear snapshots are already
        # committed directly on main. Force-push either way because the mirror is
        # intentionally a synthetic snapshot lane.
        if not linear_parent:
            _maybe(["git", "-C", str(self.mirror_dir), "branch", "-D", "main"])
            _q(["git", "-C", str(self.mirror_dir), "branch", "-m", "main"])
        _q(["git", "-C", str(self.mirror_dir), "push", "origin", "main", "--force"])

        head = _q(["git", "-C", str(self.mirror_dir), "rev-parse", "HEAD"]).stdout.strip()
        print(f"[mirror] main now at {head[:10]}")
        return head

    def mintlify_config_path(self) -> Path | None:
        """Return the first known Mintlify config path in the mirror tree."""
        for rel in MINTLIFY_CONFIG_PATHS:
            path = self.mirror_dir / rel
            if path.is_file():
                return rel
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Split-repo: overlay the code repo's tree for full-code-access items
    # ─────────────────────────────────────────────────────────────────────

    def overlay_external_tree(
        self, ext_repo: str, ext_sha: str, *, subdir: str = "_code_repo",
    ) -> str:
        """Overlay the FULL tree of another repo (the code repo of a split
        code+docs item) into the mirror under `subdir`, identity-scrubbed,
        then commit + force-push `main`.

        For split-repo items the docs live here (this mirror) but the code
        change lives in `ext_repo`. Handing the agent only the inline diff
        means it can't read the surrounding implementation. Overlaying the
        code repo at its base SHA under a reference-only `subdir/` gives every
        agent (local OR cloud — they all clone this one mirror) full browsable
        code access. The subdir is reference-only: the prompt says don't edit
        it and `fetch_pr_diff` drops any changes under it from the candidate.

        Call AFTER `set_to_sha(docs_base)`. Returns the new mirror HEAD.
        """
        # 1. Local clone of the external repo, cached + reused across agents/PRs.
        # Keyed PER code-repo: a single mirror can host items whose code lives in
        # different repos (e.g. doc-detective/common, /core, /doc-detective all
        # documented in the github.io docs mirror), so a fixed "code_src" path
        # would reuse the FIRST repo's clone and fail to find later repos' SHAs.
        code_src = self.orig_dir.parent / f"code_src__{ext_repo.replace('/', '__')}"
        if not (code_src / ".git").exists():
            shutil.rmtree(code_src, ignore_errors=True)
            url = f"https://x-access-token:{self.token}@github.com/{ext_repo}.git"
            run(["git", "clone", url, str(code_src)])
        chk = _maybe(["git", "-C", str(code_src), "cat-file", "-e", f"{ext_sha}^{{commit}}"])
        if chk.returncode != 0:
            _maybe(["git", "-C", str(code_src), "fetch", "origin", ext_sha])
            chk = _maybe(["git", "-C", str(code_src), "cat-file", "-e", f"{ext_sha}^{{commit}}"])
            if chk.returncode != 0:
                raise RuntimeError(f"overlay: SHA {ext_sha} not found in {ext_repo}")

        # 2. Wipe + repopulate the overlay subdir from `git archive`.
        dest = self.mirror_dir / subdir
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(code_src), "archive", ext_sha], stdout=subprocess.PIPE,
        )
        extract = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
        archive.stdout.close()  # type: ignore[union-attr]
        extract.communicate()
        archive.wait()
        if archive.returncode != 0 or extract.returncode != 0:
            raise RuntimeError(f"overlay archive/extract failed for {ext_repo}@{ext_sha}")
        # Drop .github inside the overlay (Dependabot / workflow noise + dep names).
        shutil.rmtree(dest / ".github", ignore_errors=True)

        # 3. Scrub the EXTERNAL repo's identity across the whole tree (the
        # overlay carries `ext_repo`'s slug/hostname in configs, go.mod, etc.).
        _scrub_upstream_identifiers(self.mirror_dir, ext_repo, self.mirror_repo)

        # 4. Commit + force-push as a follow-up snapshot on main.
        # Linked-code inputs may intentionally include generated/minified files
        # that match the documentation repository's ignore rules. They are
        # canonical model input, so stage them even when ignored.
        run(["git", "-C", str(self.mirror_dir), "add", "-A", "--force"])
        commit = _maybe([
            "git", "-C", str(self.mirror_dir),
            "-c", f"user.email={self.committer_email}",
            "-c", f"user.name={self.committer_name}",
            "commit", "-m", "snapshot", "--allow-empty",
        ])
        if commit.returncode != 0:
            raise RuntimeError(f"overlay commit failed: {commit.stderr[:400]}")
        run(["git", "-C", str(self.mirror_dir), "push", "origin", "main", "--force"])
        head = _q(["git", "-C", str(self.mirror_dir), "rev-parse", "HEAD"]).stdout.strip()
        print(f"[mirror] overlaid {ext_repo}@{ext_sha[:10]} → {subdir}/  (main now {head[:10]})")
        return head

    # ─────────────────────────────────────────────────────────────────────
    # Mintlify docs.json regeneration
    # ─────────────────────────────────────────────────────────────────────

    # Common mkdocs.yml locations across projects we mirror. First hit wins.
    _MKDOCS_PATHS = (
        Path("mkdocs.yml"),
        Path("mkdocs.yaml"),
        Path("site") / "mkdocs.yml",
        Path("site") / "mkdocs.yaml",
        Path("docs") / "mkdocs.yml",
        Path("docs") / "mkdocs.yaml",
        Path("docs") / "en" / "mkdocs.yml",
        Path("docs") / "en" / "mkdocs.yaml",
    )
    _DOCS_WALK_PATHS = (
        Path("docs"),
        Path("doc"),
        Path("content"),
        Path("documentation"),
        Path("site") / "docs",
        Path("website") / "docs",
        Path("docs") / "content",
        Path("doc") / "user" / "content",
    )

    def _is_discovery_skipped(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.mirror_dir)
        except ValueError:
            return True
        return any(part in DOC_DISCOVERY_SKIP_DIRS for part in rel.parts)

    def _find_mkdocs_path(self) -> Path | None:
        known = [
            self.mirror_dir / rel for rel in self._MKDOCS_PATHS
            if (self.mirror_dir / rel).is_file()
        ]
        if known:
            return known[0]

        candidates: list[Path] = []
        for path in self.mirror_dir.rglob("mkdocs.y*ml"):
            if not path.is_file() or self._is_discovery_skipped(path):
                continue
            rel = path.relative_to(self.mirror_dir)
            if len(rel.parts) > 5:
                continue
            candidates.append(path)
        if not candidates:
            return None

        preferred_names = {"docs", "doc", "documentation", "site", "website"}

        def score(path: Path) -> tuple[int, int, str]:
            rel = path.relative_to(self.mirror_dir)
            has_preferred = any(part in preferred_names for part in rel.parts[:-1])
            return (0 if has_preferred else 1, len(rel.parts), rel.as_posix())

        return sorted(candidates, key=score)[0]

    def _markdown_count(self, root: Path, *, limit: int = 300) -> int:
        count = 0
        for path in root.rglob("*"):
            if self._is_discovery_skipped(path):
                continue
            if path.is_file() and path.suffix.lower() in {".md", ".mdx", ".rst"}:
                count += 1
                if count >= limit:
                    return count
        return count

    def _find_docs_walk_dir(self) -> Path:
        candidates: dict[Path, int] = {}
        for rel in self._DOCS_WALK_PATHS:
            path = self.mirror_dir / rel
            if path.is_dir() and not self._is_discovery_skipped(path):
                count = self._markdown_count(path)
                if count:
                    candidates[path] = count

        likely_names = {
            "content",
            "doc",
            "docs",
            "documentation",
            "guide",
            "guides",
            "help",
            "learn",
            "manual",
            "pages",
            "site",
            "website",
        }
        for path in self.mirror_dir.rglob("*"):
            if not path.is_dir() or self._is_discovery_skipped(path):
                continue
            rel = path.relative_to(self.mirror_dir)
            if len(rel.parts) > 4:
                continue
            count = self._markdown_count(path)
            if count:
                candidates.setdefault(path, count)

        if not candidates:
            return self.mirror_dir

        def score(item: tuple[Path, int]) -> tuple[int, int, str]:
            path, count = item
            rel = path.relative_to(self.mirror_dir)
            preferred = rel in self._DOCS_WALK_PATHS or rel.as_posix() in {
                "site/docs",
                "website/docs",
                "docs",
                "doc",
                "content",
            }
            name_hint = any(part.lower() in likely_names for part in rel.parts)
            return (0 if preferred else 1, 0 if name_hint else 1, -count, len(rel.parts), rel.as_posix())

        return sorted(candidates.items(), key=score)[0][0]

    def _regen_mintlify_docs_json(self) -> None:
        """Ensure the mirror has a Mintlify config.

        Native docs.json/mint.json wins. If a mkdocs config exists, use its
        nav. Otherwise synthesize a minimal docs.json by walking discovered
        documentation content, regardless of the docs framework.
        """
        mkdocs_path = self._find_mkdocs_path()
        try:
            from .mkdocs_to_mintlify import (
                build_docs_json,
                parse_mkdocs_yml,
                walk_docs_dir,
            )
        except ImportError as exc:  # PyYAML missing, etc.
            print(f"[mirror] skipping docs.json regen ({exc})")
            return

        # If the project already ships a Mintlify config, leave it alone.
        for rel in MINTLIFY_CONFIG_PATHS:
            candidate = self.mirror_dir / rel
            if candidate.is_file():
                print(f"[mirror] keeping native docs.json at {rel}")
                return

        if mkdocs_path is not None:
            try:
                mkdocs = parse_mkdocs_yml(mkdocs_path)
                configured_docs_dir = mkdocs.get("docs_dir")
                if isinstance(configured_docs_dir, str) and configured_docs_dir.strip():
                    docs_dir = mkdocs_path.parent / configured_docs_dir
                else:
                    docs_dir = mkdocs_path.parent / "docs"
                if not docs_dir.is_dir():
                    docs_dir = mkdocs_path.parent
                try:
                    page_prefix = docs_dir.relative_to(self.mirror_dir).as_posix()
                except ValueError:
                    page_prefix = ""
                docs_json = build_docs_json(
                    mkdocs,
                    name="Project Docs",
                    page_prefix=page_prefix,
                    fallback_docs_dir=docs_dir,
                    fallback_repo_root=self.mirror_dir,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[mirror] docs.json regen failed: {exc}")
                return
            out_path = self.mirror_dir / "docs.json"
            out_path.write_text(json.dumps(docs_json, indent=2) + "\n", encoding="utf-8")
            rel = mkdocs_path.relative_to(self.mirror_dir)
            print(f"[mirror] regenerated root docs.json from {rel}")
            return

        docs_dir = self._find_docs_walk_dir()
        nav = walk_docs_dir(docs_dir, repo_root=self.mirror_dir)
        if not nav:
            docs_json = {
                "$schema": "https://mintlify.com/docs.json",
                "name": "Project Docs",
                "theme": "mint",
                "colors": {"primary": "#0a7e8c"},
                "navigation": {"pages": []},
            }
            out_path = self.mirror_dir / "docs.json"
            out_path.write_text(json.dumps(docs_json, indent=2) + "\n", encoding="utf-8")
            print("[mirror] synthesized empty docs.json fallback")
            return
        docs_json = {
            "$schema": "https://mintlify.com/docs.json",
            "name": "Project Docs",
            "theme": "mint",
            "colors": {"primary": "#0a7e8c"},
            "navigation": nav,
        }
        out_path = self.mirror_dir / "docs.json"
        out_path.write_text(json.dumps(docs_json, indent=2) + "\n", encoding="utf-8")
        rel = docs_dir.relative_to(self.mirror_dir)
        print(f"[mirror] synthesized docs.json by walking {rel} (no mkdocs.yml present)")

    # ─────────────────────────────────────────────────────────────────────
    # Pre-run cleanup
    # ─────────────────────────────────────────────────────────────────────

    def cleanup_pending(self) -> None:
        """Close any open PRs on the mirror and delete non-main remote
        branches. Call this before each agent run so leftover state from a
        previous agent can't contaminate verdict extraction (e.g., an
        already-existing docs PR getting attributed to the next agent).

        Cleanup is an isolation boundary, so it must fail closed.  In
        particular, do not use ``_maybe`` here: silently ignoring a failed
        branch deletion can expose the next item to a prior agent's output.
        """
        def sanitized(detail: str) -> str:
            return detail.replace(self.token, "***") if self.token else detail

        def remote_branches() -> set[str]:
            result = subprocess.run(
                [
                    "git", "ls-remote", "--heads",
                    f"https://x-access-token:{self.token}@github.com/{self.mirror_repo}.git",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "mirror cleanup could not list remote branches: "
                    f"{sanitized(result.stderr.strip())[:500]}"
                )
            branches: set[str] = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                    branches.add(parts[1].removeprefix("refs/heads/"))
            return branches

        # Close open PRs
        env = os.environ.copy()
        env["GH_TOKEN"] = self.token
        listing = subprocess.run(
            ["gh", "pr", "list", "--repo", self.mirror_repo, "--state", "open",
             "--json", "number", "--limit", "50"],
            capture_output=True, text=True, env=env, check=False,
        )
        if listing.returncode != 0:
            raise RuntimeError(
                "mirror cleanup could not list open PRs: "
                f"{sanitized(listing.stderr.strip())[:500]}"
            )
        if listing.stdout.strip():
            import json
            for pr in json.loads(listing.stdout):
                closed = subprocess.run(
                    [
                        "gh", "pr", "close", str(pr["number"]),
                        "--repo", self.mirror_repo, "--delete-branch",
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )
                if closed.returncode != 0:
                    raise RuntimeError(
                        f"mirror cleanup could not close PR {pr['number']}: "
                        f"{sanitized(closed.stderr.strip())[:500]}"
                    )

        # GitHub branch deletion can be briefly eventually consistent. Retry
        # the delete + verification cycle, but never allow the item to proceed
        # while any prior branch remains.
        failures: list[str] = []
        for attempt in range(3):
            stale = sorted(remote_branches() - {"main"})
            if not stale:
                return
            failures.clear()
            for branch in stale:
                deleted = subprocess.run(
                    [
                        "git", "push", "--delete",
                        f"https://x-access-token:{self.token}@github.com/{self.mirror_repo}.git",
                        branch,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if deleted.returncode != 0:
                    failures.append(
                        f"{branch}: {sanitized(deleted.stderr.strip())[:300]}"
                    )
            remaining = sorted(remote_branches() - {"main"})
            if not remaining:
                return
            if attempt < 2:
                time.sleep(1)

        details = "; ".join(failures) if failures else "remote refs remained after deletion"
        raise RuntimeError(
            "mirror cleanup failed closed; non-main branches remain: "
            f"{remaining!r} ({details})"
        )

    def cleanup_managed_output_branches(
        self,
        *,
        prefixes: tuple[str, ...],
        attempts: int = 5,
    ) -> None:
        """Delete stale managed-agent output refs immediately before dispatch.

        ``cleanup_pending`` runs at the start of vessel preparation. GitHub can
        briefly report an output branch as deleted and then expose it again
        while the associated PR close settles. A second, narrowly scoped sweep
        prevents that stale ref from failing the final mirror attestation after
        the current run's approved input branches have already been created.

        Only branches with an explicitly supplied managed-output prefix are
        touched, so the current docs-base and synthetic code-input branches are
        preserved.
        """
        if not prefixes:
            return

        def sanitized(detail: str) -> str:
            return detail.replace(self.token, "***") if self.token else detail

        def matching_remote_branches() -> list[str]:
            result = subprocess.run(
                [
                    "git", "ls-remote", "--heads",
                    f"https://x-access-token:{self.token}@github.com/{self.mirror_repo}.git",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "managed output cleanup could not list remote branches: "
                    f"{sanitized(result.stderr.strip())[:500]}"
                )
            branches = []
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) != 2 or not parts[1].startswith("refs/heads/"):
                    continue
                branch = parts[1].removeprefix("refs/heads/")
                if branch.startswith(prefixes):
                    branches.append(branch)
            return sorted(branches)

        failures: list[str] = []
        remaining: list[str] = []
        for attempt in range(max(1, attempts)):
            stale = matching_remote_branches()
            if not stale:
                return
            failures.clear()
            for branch in stale:
                deleted = subprocess.run(
                    [
                        "git", "push", "--delete",
                        f"https://x-access-token:{self.token}@github.com/{self.mirror_repo}.git",
                        branch,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if deleted.returncode != 0:
                    failures.append(
                        f"{branch}: {sanitized(deleted.stderr.strip())[:300]}"
                    )
            remaining = matching_remote_branches()
            if not remaining:
                return
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 5))

        details = "; ".join(failures) if failures else "remote refs remained after deletion"
        raise RuntimeError(
            "managed output cleanup failed closed; matching branches remain: "
            f"{remaining!r} ({details})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Code-only PR
    # ─────────────────────────────────────────────────────────────────────

    def open_code_only_pr(
        self,
        base_sha: str,
        head_sha: str,
        *,
        base_branch: str = "main",
        branch_name: str = "candidate/code",
        doc_pattern: str | None = None,
        pr_file_allowlist: set[str] | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> str | None:
        """Generate the base→head diff from `orig`, strip doc/example files,
        apply it on a fresh branch in the mirror, push, and open a PR
        targeting `main`. Mirror `main` must already be at base_sha (call
        set_to_sha(base_sha) first).

        `doc_pattern` is an optional regex; lines whose filename matches are
        excluded. If None, falls back to the heuristic `is_non_code_file()`
        used by fork_repo_at_base.

        Returns the PR URL, or None if there's no code diff to apply.
        """
        # Determine the CODE files changed base→head (exclude docs/tests/etc).
        name_status_command = [
            "git", "-C", str(self.orig_dir), "diff", "--name-status",
        ]
        if pr_file_allowlist is not None:
            # The canonical allowlist is authoritative. Disable similarity-
            # based rename inference so an unrelated base-divergence deletion
            # cannot be paired with an allowlisted addition and materialized
            # into the synthetic agent-input PR.
            name_status_command.append("--no-renames")
        name_status_command.append(f"{base_sha}..{head_sha}")
        name_status = _q(name_status_command).stdout
        def _is_allowed(path: str) -> bool:
            return pr_file_allowlist is None or path in pr_file_allowlist

        def _is_code(path: str) -> bool:
            # A canonical file-level allowlist is authoritative.  Do not let
            # the legacy static docs heuristic or directory regex silently
            # remove paths after they have been classified and approved.
            if pr_file_allowlist is not None:
                return path in pr_file_allowlist
            return _is_code_path(path, doc_pattern)

        writes: list[str] = []   # take head version
        deletes: list[str] = []  # remove from tree
        for line in name_status.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            code = parts[0][0]
            if code == "R" and len(parts) >= 3:      # rename: drop old, take new
                if _is_allowed(parts[1]) or _is_allowed(parts[2]):
                    deletes.append(parts[1])
                if _is_allowed(parts[2]) and _is_code(parts[2]):
                    writes.append(parts[2])
            elif code == "D":
                if _is_allowed(parts[1]) and _is_code(parts[1]):
                    deletes.append(parts[1])
            else:                                    # A, M, C, T
                if _is_allowed(parts[-1]) and _is_code(parts[-1]):
                    writes.append(parts[-1])

        if not writes and not deletes:
            print("[mirror] no code-only diff to apply; skipping PR.")
            return None

        # Fresh branch off the pinned prepared base.
        run(["git", "-C", str(self.mirror_dir), "fetch", "origin", base_branch])
        run(["git", "-C", str(self.mirror_dir), "checkout", "-B", base_branch, f"origin/{base_branch}"])
        _delete_local_branch_conflicts(self.mirror_dir, branch_name)
        run(["git", "-C", str(self.mirror_dir), "checkout", "-b", branch_name])

        # Materialize each changed code file's HEAD version onto the scrubbed
        # base (tree overlay), then re-scrub — rather than applying an
        # *unscrubbed* base→head patch onto the *scrubbed* tree, which mismatched
        # context on scrubbed files (package.json/README → .rej) and, when every
        # changed file was scrubbed, yielded an empty diff (a total miss, e.g.
        # doc-detective#244). The scrub is idempotent on already-scrubbed files.
        for path in deletes:
            dest = self.mirror_dir / path
            if dest.exists():
                dest.unlink()
        for path in writes:
            res = subprocess.run(
                ["git", "-C", str(self.orig_dir), "show", f"{head_sha}:{path}"],
                capture_output=True,
            )
            if res.returncode != 0:
                continue
            dest = self.mirror_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(res.stdout)

        _scrub_upstream_identifiers(self.mirror_dir, self.source_repo, self.mirror_repo)

        # Preserve ignored generated artifacts from the linked implementation
        # PR; the canonical changed-path set, not this repo's .gitignore, is
        # authoritative for the synthetic input PR.
        run(["git", "-C", str(self.mirror_dir), "add", "-A", "--force"])
        staged = _q(["git", "-C", str(self.mirror_dir), "diff", "--cached", "--name-only"]).stdout
        if not staged.strip():
            print("[mirror] nothing staged after overlay; no PR opened.")
            return None

        # Title and body MUST be caller-provided when they should look natural
        # to the agent — defaults here are neutral and never leak the source
        # repo or upstream SHAs. Pass a generated title/body via run_candidate.
        commit_title = title or "update implementation"
        run([
            "git", "-C", str(self.mirror_dir),
            "-c", f"user.email={self.committer_email}",
            "-c", f"user.name={self.committer_name}",
            "commit", "-m", commit_title,
        ])
        run(["git", "-C", str(self.mirror_dir), "push", "origin", branch_name, "--force"])

        env = os.environ.copy()
        env["GH_TOKEN"] = self.token
        pr_body = body or "Update implementation."
        res = subprocess.run(
            ["gh", "pr", "create", "--repo", self.mirror_repo,
             "--base", base_branch, "--head", branch_name,
             "--title", commit_title, "--body", pr_body],
            capture_output=True, text=True, env=env, check=False,
        )
        if res.returncode != 0:
            print(f"[mirror] gh pr create failed:\n{res.stderr[:600]}")
            return None
        return res.stdout.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Overlay code-change PR (split-repo: code change as a browsable PR)
    # ─────────────────────────────────────────────────────────────────────

    def open_overlay_code_pr(
        self,
        ext_repo: str,
        base_sha: str,
        head_sha: str,
        *,
        base_branch: str = "main",
        subdir: str = "_code_repo",
        branch_name: str = "candidate/code-change",
        pr_file_allowlist: set[str] | None = None,
        doc_pattern: str | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> str | None:
        """Represent a split-repo code change (ext_repo base→head) as a real PR
        on THIS mirror, with the changed files materialized under `subdir/`.

        Used instead of pasting a huge diff into the prompt: the agent browses a
        normal PR (and the `subdir/` tree) rather than reading inline diff text —
        which devin rejects outright above 30k chars, and which buries the task
        for everyone on large changes. Mirror `main` must already be at the docs
        base with `subdir/` overlaid at base_sha (call set_to_sha +
        overlay_external_tree(ext_repo, base_sha) first). Returns the PR URL, or
        None if there's no diff. Leaves the mirror checked out on `main`.
        """
        # Same per-repo cache key as overlay_external_tree (one mirror can host
        # code from multiple repos — a fixed path would read the wrong clone).
        code_src = self.orig_dir.parent / f"code_src__{ext_repo.replace('/', '__')}"  # cached by overlay_external_tree
        chk = _maybe(["git", "-C", str(code_src), "cat-file", "-e", f"{head_sha}^{{commit}}"])
        if chk.returncode != 0:
            _maybe(["git", "-C", str(code_src), "fetch", "origin", head_sha])

        name_status_command = [
            "git", "-C", str(code_src), "diff", "--name-status",
        ]
        if pr_file_allowlist is not None:
            name_status_command.append("--no-renames")
        name_status_command.append(f"{base_sha}..{head_sha}")
        name_status = _q(name_status_command).stdout
        def _is_allowed(path: str) -> bool:
            return pr_file_allowlist is None or path in pr_file_allowlist

        def _is_code(path: str) -> bool:
            if pr_file_allowlist is not None:
                return path in pr_file_allowlist
            return _is_code_path(path, doc_pattern)

        writes: list[str] = []
        deletes: list[str] = []
        for line in name_status.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            code = parts[0][0]
            if code == "R" and len(parts) >= 3:
                if (_is_allowed(parts[1]) or _is_allowed(parts[2])) and _is_code(parts[1]):
                    deletes.append(parts[1])
                if _is_allowed(parts[2]) and _is_code(parts[2]):
                    writes.append(parts[2])
            elif code == "D":
                if _is_allowed(parts[1]) and _is_code(parts[1]):
                    deletes.append(parts[1])
            else:
                if _is_allowed(parts[-1]) and _is_code(parts[-1]):
                    writes.append(parts[-1])
        if not writes and not deletes:
            print("[mirror] no external code diff to apply; skipping overlay PR.")
            return None

        run(["git", "-C", str(self.mirror_dir), "fetch", "origin", base_branch])
        run(["git", "-C", str(self.mirror_dir), "checkout", "-B", base_branch, f"origin/{base_branch}"])
        _delete_local_branch_conflicts(self.mirror_dir, branch_name)
        run(["git", "-C", str(self.mirror_dir), "checkout", "-b", branch_name])

        for path in deletes:
            dest = self.mirror_dir / subdir / path
            if dest.exists():
                dest.unlink()
        for path in writes:
            res = subprocess.run(
                ["git", "-C", str(code_src), "show", f"{head_sha}:{path}"],
                capture_output=True,
            )
            if res.returncode != 0:
                continue
            dest = self.mirror_dir / subdir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(res.stdout)

        _scrub_upstream_identifiers(self.mirror_dir, ext_repo, self.mirror_repo)

        # The linked implementation PR can include ignored generated assets
        # (for example a checked-in dist bundle). Force-stage the exact
        # canonical path set into the managed input PR.
        run(["git", "-C", str(self.mirror_dir), "add", "-A", "--force"])
        staged = _q(["git", "-C", str(self.mirror_dir), "diff", "--cached", "--name-only"]).stdout
        if not staged.strip():
            print("[mirror] nothing staged for overlay code PR.")
            run(["git", "-C", str(self.mirror_dir), "checkout", "main"])
            return None

        commit_title = title or "update implementation"
        run(["git", "-C", str(self.mirror_dir),
             "-c", f"user.email={self.committer_email}",
             "-c", f"user.name={self.committer_name}",
             "commit", "-m", commit_title])
        run(["git", "-C", str(self.mirror_dir), "push", "origin", branch_name, "--force"])

        env = os.environ.copy()
        env["GH_TOKEN"] = self.token
        res = subprocess.run(
            ["gh", "pr", "create", "--repo", self.mirror_repo,
             "--base", base_branch, "--head", branch_name,
             "--title", commit_title, "--body", body or "Update implementation."],
            capture_output=True, text=True, env=env, check=False,
        )
        run(["git", "-C", str(self.mirror_dir), "checkout", "main"])
        if res.returncode != 0:
            print(f"[mirror] overlay code PR create failed:\n{res.stderr[:600]}")
            return None
        return res.stdout.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Clone for the agent
    # ─────────────────────────────────────────────────────────────────────

    def clone_for_agent(self, target: Path, *, branch: str = "main") -> Path:
        """Fresh clone of the mirror repo with token-embedded origin, ready
        for an agent to push branches from. Per the project's fresh-clone-
        per-PR discipline, this is a new directory per agent run."""
        if target.exists():
            shutil.rmtree(target)
        run([
            "git", "clone", "--single-branch", "--no-tags", "--branch", branch,
            f"https://x-access-token:{self.token}@github.com/{self.mirror_repo}.git",
            str(target),
        ])
        run(["git", "-C", str(target), "config", "user.email", self.committer_email])
        run(["git", "-C", str(target), "config", "user.name", self.committer_name])
        return target


# File extensions/names eligible for upstream-identifier scrubbing. We only
# touch text-only project metadata + prose — source code stays untouched so
# imports and runtime behavior aren't broken. .py is intentionally excluded:
# `import fastapi` must keep working in the mirror, and the test files /
# package modules already had docs_src/ and docs/ stripped from the agent's
# code-only PR view, so they aren't a leak vector via tree scrub.
_SCRUB_EXTENSIONS = {
    ".md", ".mdx", ".rst",
    ".yml", ".yaml",
    ".toml",
    ".txt",
    ".cfg", ".ini",
    ".json",  # but skip *.lock.json / package-lock.json below
    ".html",
}
_SCRUB_BASENAMES = {
    "readme", "license", "licence", "authors", "notice", "contributing",
    "code_of_conduct", "changelog", "history", "maintainers", "credits",
}


def _is_scrubable(path: Path) -> bool:
    name = path.name.lower()
    if name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                "poetry.lock", "uv.lock", "cargo.lock", "gemfile.lock"}:
        return False
    if path.suffix.lower() in _SCRUB_EXTENSIONS:
        return True
    # Files like LICENSE, AUTHORS with no extension.
    stem = path.stem.lower()
    if stem in _SCRUB_BASENAMES or name in _SCRUB_BASENAMES:
        return True
    return False


def _scrub_upstream_identifiers(
    mirror_dir: Path, source_repo: str, mirror_repo: str,
) -> None:
    """Replace the upstream org and repo name with the mirror's org and
    UUID-name in every text-only file under `mirror_dir`. Skips source code
    (.py / .js / .ts / …) so `import <pkg>` and equivalent continue to work.

    Replacement strategy (surgical — only the things that let an agent
    actually find the upstream via the GitHub API):
      1. Full slug `<src_org>/<src_repo>` → `<mirror_org>/<uuid>`
      2. `<any-owner>/<src_repo>` → `<mirror_org>/<uuid>` (projects
         sometimes move orgs; e.g. tiangolo/fastapi -> fastapi/fastapi)
      3. Org hostname `<src_org>.<tld>` → `repo.<mirror_org>.com`

    We deliberately do NOT scrub the standalone `<src_org>` or `<src_repo>`
    tokens. Replacing the standalone org name produced awkward output for
    repos where the org IS the brand (Promptless, Helm, Mintlify): prose
    like "Promptless treats the task as ..." became "the configured mirror organization
    treats the task as ...". The standalone name alone isn't searchable to
    the GitHub API — slugs and hostnames are — so leaving it intact
    preserves readability without opening the lookup path.
    """
    src_org, _, src_repo = source_repo.partition("/")
    mirror_org, _, mirror_uuid = mirror_repo.partition("/")
    if not src_org or not src_repo:
        return

    full_slug_re = re.compile(re.escape(source_repo), re.IGNORECASE)
    # Projects sometimes move orgs (e.g. fastapi moved from `tiangolo/fastapi`
    # to `fastapi/fastapi`); the codebase ends up with multiple owner names
    # on the same repo. Match `<github-username>/<src_repo>` so both landings
    # become the mirror identity rather than `<uuid>/<uuid>` (which happens
    # if we let the standalone repo-name regex fire on each half).
    #
    # The owner pattern follows GitHub's actual username rules (alphanumeric
    # + single hyphens, must start with a letter or digit, length 1-39). No
    # underscores or dots — that excludes false matches against file paths
    # like `docs_src/fastapi/foo.py` where the segment before the slash has
    # an underscore.
    # `(?<!\.)` prevents matching domain segments like `com/fastapi` in
    # `https://github.com/fastapi/fastapi`. With the lookbehind, the regex
    # skips `com` (preceded by `.`) and locks onto `fastapi/fastapi`
    # (preceded by `/`), which is what we want.
    any_owner_slug_re = re.compile(
        rf"(?<!\.)\b[A-Za-z0-9][A-Za-z0-9-]{{0,38}}/{re.escape(src_repo)}\b",
        re.IGNORECASE,
    )
    # Replace any hostname that contains the source org as a label
    # (`fastapi.tiangolo.com`, `tiangolo.com`, `www.tiangolo.io`, etc.) with
    # a single neutral subdomain `repo.<mirror_org>.com`. Runs BEFORE the
    # standalone-org pattern so we don't half-rewrite a hostname like
    # `fastapi.the configured mirror organization.com` (which is what the bare-org regex
    # would otherwise produce).
    # IMPORTANT: only a real-TLD allowlist — NOT any 2–6 letter suffix. The
    # old `[a-z]{2,6}` matched FILE EXTENSIONS, so a config filename like
    # `.doc-detective.json` (or `doc-detective.config`) was mangled into
    # `.repo.<mirror>.com` and leaked into the docs the agent sees and its diff.
    # A hostname's TLD is one of these; `.json/.js/.ts/.md/.yaml/.config` are not.
    _REAL_TLDS = ("com|org|io|net|dev|ai|sh|app|co|info|xyz|gg|me|tv|cloud|tools|"
                  "page|site|run|fyi|build")
    org_domain_re = re.compile(
        rf"\b(?:[A-Za-z0-9-]+\.)?{re.escape(src_org)}\.(?:{_REAL_TLDS})\b",
        re.IGNORECASE,
    )
    def _replace_org_domain(_match: re.Match) -> str:
        return f"repo.{mirror_org}.com"

    def _replace_slug(_match: re.Match) -> str:
        return f"{mirror_org}/{mirror_uuid}"

    # This scrubber runs once for the base snapshot and again after the code
    # overlay.  Protect the already-scrubbed mirror identity so the broad
    # ``<any-owner>/<source-repo>`` rule cannot start inside a mirror URL/path
    # and rewrite it a second time (for example ``<mirror-uuid>/<repo>.yaml``).
    # Underscores make the sentinel ineligible for GitHub's owner syntax.
    mirror_identity_sentinel = "__DOCBENCH_PROTECTED_MIRROR_IDENTITY__"
    mirror_identity_re = re.compile(re.escape(mirror_repo), re.IGNORECASE)

    scrubbed = 0
    for dirpath, dirnames, filenames in os.walk(mirror_dir):
        # Prune .git before traversal. Checking path.parts after Path.rglob()
        # is too late: rglob has already entered Git's mutable object store,
        # where transient fan-out directories can disappear during iteration.
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for filename in filenames:
            path = Path(dirpath) / filename
            if not path.is_file():
                continue
            if not _is_scrubable(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            new = mirror_identity_re.sub(mirror_identity_sentinel, text)
            new = full_slug_re.sub(_replace_slug, new)
            new = any_owner_slug_re.sub(_replace_slug, new)
            new = org_domain_re.sub(_replace_org_domain, new)
            new = new.replace(mirror_identity_sentinel, mirror_repo)
            if new != text:
                path.write_text(new, encoding="utf-8")
                scrubbed += 1
    if scrubbed:
        print(f"[mirror] scrubbed upstream identifiers in {scrubbed} file(s)")


def _is_code_path(path: str, doc_pattern: str | None = None) -> bool:
    """Return whether ``path`` may be shown in a code-only PR.

    The static docs heuristic is always applied. ``doc_pattern`` is an
    additional classifier-derived strip rule, not a replacement; otherwise a
    missed classifier directory can leak obvious docs files such as ``*.md``.
    """
    if is_non_code_file(path):
        return False
    return not (re.search(doc_pattern, path) if doc_pattern else False)


def _filter_diff_by_regex(full_diff: str, pattern: re.Pattern, *, keep: bool) -> str:
    """Split a unified diff and keep/drop chunks based on whether the file
    path matches `pattern`. `keep=True` keeps only matching files; `keep=False`
    drops them."""
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

    out: list[str] = []
    for chunk in chunks:
        fp = None
        for line in chunk.splitlines():
            if line.startswith("+++ b/"):
                fp = line[6:].rstrip("\n")
                break
        if fp is None:
            for line in chunk.splitlines():
                if line.startswith("diff --git "):
                    parts = line.split(" b/", 1)
                    if len(parts) == 2:
                        fp = parts[1].rstrip("\n")
                    break
        if fp is None:
            continue
        matched = bool(pattern.search(fp))
        if matched == keep:
            out.append(chunk)
    return "".join(out)


def _filter_diff_by_paths(full_diff: str, paths: set[str]) -> str:
    """Keep only file chunks whose old or new path appears in ``paths``."""
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

    def _chunk_paths(chunk: str) -> set[str]:
        found: set[str] = set()
        for line in chunk.splitlines():
            if line.startswith("diff --git "):
                m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
                if m:
                    found.update(p for p in m.groups() if p and p != "/dev/null")
            elif line.startswith("--- a/") or line.startswith("+++ b/"):
                found.add(line[6:].rstrip("\n"))
        found.discard("/dev/null")
        return found

    return "".join(chunk for chunk in chunks if _chunk_paths(chunk) & paths)
