"""Trusted source resolution and live GitHub attestations for public items."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class SourceAttestationError(RuntimeError):
    """Raised when a source PR cannot be bound to the frozen item identity."""


@dataclass(frozen=True)
class GitHubPullRequest:
    owner: str
    repo: str
    number: int

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_github_pr_url(value: str) -> GitHubPullRequest | None:
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "github.com"
        or len(parts) != 4
        or parts[2] != "pull"
        or not parts[3].isdigit()
    ):
        return None
    return GitHubPullRequest(parts[0], parts[1], int(parts[3]))


def _github_json(path: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dogbench-public-harness",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub host
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SourceAttestationError(f"GitHub attestation failed for {path}: {exc}") from exc


def _github_pages(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    separator = "&" if "?" in path else "?"
    while True:
        payload = _github_json(f"{path}{separator}per_page=100&page={page}")
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise SourceAttestationError(f"GitHub returned an invalid paginated response for {path}")
        rows.extend(payload)
        if len(payload) < 100:
            return rows
        page += 1


def _github_repo_slug(repo_url: str) -> str | None:
    parsed = urlparse(repo_url)
    if parsed.netloc.casefold() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def attest_github_component(
    *,
    source_url: str,
    repo_url: str,
    base_sha: str,
    head_sha: str,
    paths: list[str],
    frozen_research_base: bool = False,
) -> dict[str, Any]:
    pr = parse_github_pr_url(source_url)
    repo_slug = _github_repo_slug(repo_url)
    if pr is None or repo_slug is None:
        raise SourceAttestationError(
            "source_url and repo_url must be canonical https://github.com URLs; "
            "use --offline only for local fixtures or an already-attested mirror"
        )
    if pr.slug.casefold() != repo_slug.casefold():
        raise SourceAttestationError(
            f"source PR repository {pr.slug} does not match repo_url {repo_slug}"
        )

    metadata = _github_json(f"/repos/{pr.slug}/pulls/{pr.number}")
    if not isinstance(metadata, dict):
        raise SourceAttestationError(f"{source_url}: invalid PR metadata response")
    reported_base = str((metadata.get("base") or {}).get("sha") or "").lower()
    live_head = str((metadata.get("head") or {}).get("sha") or "").lower()
    if live_head != head_sha.lower():
        raise SourceAttestationError(
            f"{source_url}: head SHA drifted: expected {head_sha}, got {live_head or 'missing'}"
        )

    comparison = _github_json(
        f"/repos/{pr.slug}/compare/{base_sha.lower()}...{head_sha.lower()}"
    )
    if not isinstance(comparison, dict):
        raise SourceAttestationError(f"{source_url}: invalid compare response")
    merge_base = str((comparison.get("merge_base_commit") or {}).get("sha") or "").lower()
    if merge_base != base_sha.lower() and not frozen_research_base:
        raise SourceAttestationError(
            f"{source_url}: declared base is not the PR range merge base: "
            f"expected {base_sha}, got {merge_base or 'missing'}"
        )

    changed_paths: set[str] = set()
    for row in _github_pages(f"/repos/{pr.slug}/pulls/{pr.number}/files"):
        filename = row.get("filename")
        previous = row.get("previous_filename")
        if filename:
            changed_paths.add(str(filename))
        if previous:
            changed_paths.add(str(previous))
    missing = sorted(set(paths) - changed_paths)
    if missing:
        raise SourceAttestationError(
            f"{source_url}: declared paths are absent from the live PR: {', '.join(missing)}"
        )
    protected_prose = [
        str(value).strip()
        for value in (metadata.get("title"), metadata.get("body"))
        if isinstance(value, str) and value.strip()
    ]
    for endpoint in ("comments", "reviews"):
        for row in _github_pages(f"/repos/{pr.slug}/pulls/{pr.number}/{endpoint}"):
            body = row.get("body")
            if isinstance(body, str) and body.strip():
                protected_prose.append(body.strip())
    for row in _github_pages(f"/repos/{pr.slug}/issues/{pr.number}/comments"):
        body = row.get("body")
        if isinstance(body, str) and body.strip():
            protected_prose.append(body.strip())
    for row in _github_pages(f"/repos/{pr.slug}/pulls/{pr.number}/commits"):
        message = ((row.get("commit") or {}).get("message"))
        if isinstance(message, str) and message.strip():
            protected_prose.append(message.strip())

    return {
        "schema_version": "dogbench-github-source-attestation-v1",
        "ok": True,
        "repo": pr.slug,
        "pr_number": pr.number,
        "base_sha": base_sha.lower(),
        "github_merge_base_sha": merge_base,
        "base_policy": "frozen_research_binding" if frozen_research_base else "pr_merge_base",
        "github_reported_base_sha": reported_base,
        "head_sha": live_head,
        "declared_paths": sorted(paths),
        "live_changed_paths": sorted(changed_paths),
        # Trusted-controller-only material. The workspace builder consumes and
        # removes this field before persisting the attestation.
        "_protected_source_prose": protected_prose,
    }


def attest_item_sources(item: dict[str, Any], *, offline: bool) -> dict[str, Any]:
    if offline:
        return {
            "schema_version": "dogbench-source-attestations-v1",
            "ok": True,
            "mode": "offline_explicit",
            "components": [],
        }

    from .research_inputs import research_input

    frozen = research_input(item)
    components: list[dict[str, Any]] = []
    docs = item["docs"]
    if docs["head_sha"] is not None:
        components.append(
            attest_github_component(
                source_url=item["source_url"],
                repo_url=docs["repo_url"],
                base_sha=docs["base_sha"],
                head_sha=docs["head_sha"],
                paths=docs["paths"],
                frozen_research_base=frozen is not None,
            )
        )
    if item.get("code"):
        code = item["code"]
        components.append(
            attest_github_component(
                source_url=code["source_url"],
                repo_url=code["repo_url"],
                base_sha=code["base_sha"],
                head_sha=code["head_sha"],
                paths=code["paths"],
                frozen_research_base=frozen is not None,
            )
        )
    return {
        "schema_version": "dogbench-source-attestations-v1",
        "ok": True,
        "mode": "live_github",
        "components": components,
    }
