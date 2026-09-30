"""Read-only verification of operator-owned cloud-agent mirrors.

Mirror creation/provisioning is explicit operator setup. No account, organization,
local cache, hosted project or repository is selected automatically. The cloud
runner calls this check before it invokes an adapter that can write to a mirror.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from typing import Any


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        # Remote URL failures may contain credentials; do not echo subprocess text.
        raise RuntimeError(f"mirror Git verification failed ({args[0]})")
    return result.stdout.strip()


def _github_slug(url: str) -> str | None:
    if url.startswith("git@github.com:"):
        value = url[len("git@github.com:"):]
    else:
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
            return None
        value = parsed.path.lstrip("/")
    value = value.removesuffix(".git").rstrip("/")
    return value if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) else None


def validate_prepared_mirror(
    repo_dir: Path, mirror_repo: str, expected_base_sha: str,
    prepared_docs_dir: Path, base_branch: str = "main",
    prepared_code_dir: Path | None = None, code_path_prefix: str = "",
    managed_transport: bool = False, source_repo: str | None = None,
    code_source_repo: str | None = None, mirror_overlays_root: Path | None = None,
    source_overlays_root: Path | None = None,
) -> dict[str, Any]:
    """Validate local origin, clean pinned base and byte-identical prepared tree.

    The caller must separately check GitHub visibility/ownership and remote branch
    SHA using its explicit controller credential. This function does not contact
    GitHub and cannot establish what a remote currently contains.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", mirror_repo):
        raise ValueError("mirror_repo must be an explicit owner/repository")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_base_sha):
        raise ValueError("expected_base_sha must be an immutable 40-character SHA")
    _git(repo_dir, "check-ref-format", "--branch", base_branch)
    urls = _git(repo_dir, "remote", "get-url", "--all", "origin").splitlines()
    urls += _git(repo_dir, "remote", "get-url", "--push", "--all", "origin").splitlines()
    if not urls or any(_github_slug(url) != mirror_repo for url in urls):
        raise RuntimeError("configured mirror repository does not match every origin fetch/push URL")
    actual_base = _git(repo_dir, "rev-parse", f"refs/heads/{base_branch}^{{commit}}")
    if actual_base.lower() != expected_base_sha.lower():
        raise RuntimeError("configured mirror branch differs from the pinned base SHA")
    if _git(repo_dir, "rev-parse", "HEAD") != actual_base:
        raise RuntimeError("configured mirror checkout is not at the pinned base")
    if _git(repo_dir, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("configured mirror checkout must be clean")
    actual_tree = _git(repo_dir, "rev-parse", f"{actual_base}^{{tree}}")
    prepared_tree = _git(prepared_docs_dir, "rev-parse", "HEAD^{tree}")
    transport_details: dict[str, str] = {"transport_mode": "exact"}
    if code_path_prefix:
        prefix = PurePosixPath(code_path_prefix)
        if prefix.is_absolute() or ".." in prefix.parts or ".git" in prefix.parts or str(prefix) == ".":
            raise ValueError("code_path_prefix must be a safe relative directory")
    if managed_transport:
        if not source_repo or mirror_overlays_root is None or source_overlays_root is None:
            raise ValueError("managed transport requires source_repo and explicit overlay roots")
        for slug in (source_repo, code_source_repo):
            if slug is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", slug):
                raise ValueError("source repositories must use owner/repository form")
        details = _managed_expected_tree(prepared_docs_dir=prepared_docs_dir,
            prepared_code_dir=prepared_code_dir, source_repo=source_repo,
            code_source_repo=code_source_repo, mirror_repo=mirror_repo,
            code_path_prefix=code_path_prefix,
            mirror_overlays_root=Path(mirror_overlays_root).resolve(),
            source_overlays_root=Path(source_overlays_root).resolve())
        prepared_tree = details.pop('tree')
        transport_details = {"transport_mode": "managed", **details}
    elif code_path_prefix:
        prefix = PurePosixPath(code_path_prefix)
        if prefix.is_absolute() or ".." in prefix.parts or ".git" in prefix.parts or str(prefix) == ".":
            raise ValueError("code_path_prefix must be a safe relative directory")
        if prepared_code_dir is None:
            raise ValueError("code_path_prefix requires prepared_code_dir")
        # Recreate the original mirror's separate code-repository overlay from
        # immutable Git trees, including its pre-change code state. No checkout
        # filters, source history, or current working-tree files are imported.
        code_base_tree = _git(prepared_code_dir, "rev-parse", "HEAD~1^{tree}")
        with tempfile.TemporaryDirectory(prefix="dogbench-mirror-check-") as temporary:
            combined = Path(temporary)
            _git(combined, "init", "--quiet", "--bare")
            _copy_git_tree_objects(prepared_docs_dir, combined, prepared_tree)
            _copy_git_tree_objects(prepared_code_dir, combined, code_base_tree)
            _git(combined, "read-tree", prepared_tree)
            _git(combined, "read-tree", f"--prefix={str(prefix).rstrip('/')}/", code_base_tree)
            prepared_tree = _git(combined, "write-tree")
    if actual_tree != prepared_tree:
        raise RuntimeError("configured mirror tree differs from frozen prepared documentation")
    return {"ok": True, "mirror_repo": mirror_repo, "base_branch": base_branch,
            "base_sha": actual_base, "docs_tree": actual_tree, **transport_details}


# Frozen-tree transport helpers extracted from the original mirror implementation.

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


def _overlay_digest(root: Path) -> str:
    """Bind every file and its location in an operator-supplied overlay tree."""
    import hashlib
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError('managed transport overlay must not contain symlinks')
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode() + b'\0')
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _managed_expected_tree(
    *, prepared_docs_dir: Path, prepared_code_dir: Path | None,
    source_repo: str, code_source_repo: str | None, mirror_repo: str,
    code_path_prefix: str, mirror_overlays_root: Path, source_overlays_root: Path,
) -> dict[str, Any]:
    """Replay original managed transformations against local-only frozen clones."""
    # Require the cloud extra rather than silently omitting the source adapter's
    # Mintlify navigation transform when its optional parser is unavailable.
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('managed mirror verification requires the cloud extra (PyYAML)') from exc
    from . import cloud_mirrors
    before = {
        'mirror_overlay_sha256': _overlay_digest(mirror_overlays_root / mirror_repo.split('/', 1)[1]),
        'source_overlay_sha256': _overlay_digest(source_overlays_root / source_repo.replace('/', '__')),
    }
    previous_paths = dict(cloud_mirrors._MIRROR_PATHS)
    try:
        with tempfile.TemporaryDirectory(prefix='dogbench-managed-tree-') as temporary:
            root = Path(temporary)
            source = root/'source'; target = root/'mirror'; origin = root/'origin.git'
            subprocess.run(['git','clone','--quiet','--no-hardlinks',str(prepared_docs_dir),str(source)], check=True, capture_output=True)
            target.mkdir(); origin.mkdir()
            _git(target,'init','--quiet','--initial-branch=main')
            _git(origin,'init','--quiet','--bare','--initial-branch=main')
            _git(target,'config','core.hooksPath','/dev/null')
            _git(origin,'config','core.hooksPath','/dev/null')
            _git(target,'remote','add','origin',str(origin))
            cloud_mirrors.configure_mirrors(index_path=root/'index.json',
                mirror_overlays_root=mirror_overlays_root, source_overlays_root=source_overlays_root)
            mirror = cloud_mirrors.Mirror(source_repo,mirror_repo,source,target,'')
            mirror.set_to_sha(_git(source,'rev-parse','HEAD'),sanitize_for_managed_github=True)
            if code_path_prefix:
                if prepared_code_dir is None or not code_source_repo:
                    raise ValueError('managed code overlay requires frozen code and its source repository')
                cached = root / f"code_src__{code_source_repo.replace('/', '__')}"
                subprocess.run(['git','clone','--quiet','--no-hardlinks',str(prepared_code_dir),str(cached)],check=True,capture_output=True)
                mirror.overlay_external_tree(code_source_repo, _git(cached,'rev-parse','HEAD~1'),
                                             subdir=code_path_prefix.rstrip('/'))
            tree = _git(target,'rev-parse','HEAD^{tree}')
            code_details: dict[str, Any] = {}
            if prepared_code_dir is not None:
                # Same materialization as open_code_only_pr/open_overlay_code_pr:
                # canonical changed paths take their HEAD bytes, deletions remove
                # their base files, then identity scrubbing runs on the whole tree.
                names = _git(prepared_code_dir, 'diff', '--name-only', '--no-renames', 'HEAD~1..HEAD').splitlines()
                for name in names:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
                        raise ValueError('prepared code change contains an unsafe path')
                    result = subprocess.run(['git','-C',str(prepared_code_dir),'show',f'HEAD:{name}'],capture_output=True)
                    destination = target / code_path_prefix / name
                    if result.returncode:
                        if destination.exists():
                            destination.unlink()
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(result.stdout)
                cloud_mirrors._scrub_upstream_identifiers(target, code_source_repo or source_repo, mirror_repo)
                _git(target, 'add', '-A', '--force')
                code_head_tree = _git(target, 'write-tree')
                changed = _git(target, 'diff', '--name-only', '--no-renames', tree, code_head_tree).splitlines()
                def blobs(revision: str) -> dict[str, str | None]:
                    result = {}
                    for path in changed:
                        query = subprocess.run(['git','-C',str(target),'rev-parse','--verify',f'{revision}:{path}'],capture_output=True,text=True)
                        result[path] = query.stdout.strip() if query.returncode == 0 else None
                    return result
                code_details = {'code_base_blob_oids': blobs(tree),
                                'code_head_blob_oids': blobs(code_head_tree),
                                'code_paths': changed, 'code_head_tree': code_head_tree}
    finally:
        cloud_mirrors._MIRROR_PATHS.clear()
        cloud_mirrors._MIRROR_PATHS.update(previous_paths)
    after = {
        'mirror_overlay_sha256': _overlay_digest(mirror_overlays_root / mirror_repo.split('/', 1)[1]),
        'source_overlay_sha256': _overlay_digest(source_overlays_root / source_repo.replace('/', '__')),
    }
    if before != after:
        raise RuntimeError('managed transport overlays changed during verification')
    return {'tree':tree, **before, **code_details}
