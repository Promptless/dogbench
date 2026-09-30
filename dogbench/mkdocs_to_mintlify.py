#!/usr/bin/env python3
"""Convert an mkdocs.yml `nav:` block into a Mintlify-compatible docs.json.

Mintlify uses docs.json (or mint.json) at its configured root to know the
documentation structure. mkdocs uses YAML; the nav section uses entries like:

  - features.md                       # untitled page
  - FastAPI: index.md                 # titled page
  - Learn:                            # group
    - learn/index.md
    - tutorial/index.md
    - Tutorial - User Guide:          # nested group
      - tutorial/first-steps.md

We map this to Mintlify's:

  {
    "$schema": "https://mintlify.com/docs.json",
    "name": <anonymous>,
    "theme": "mint",
    "colors": { "primary": "#0a7e8c" },
    "navigation": {
      "groups": [
        { "group": "Learn", "pages": ["learn/index", "tutorial/index",
                                       { "group": "Tutorial - User Guide",
                                         "pages": ["tutorial/first-steps"] }] }
      ]
    }
  }

Page strings drop the .md/.mdx extension.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any



def _strip_ext(p: str) -> str:
    return re.sub(r"\.(md|mdx|rst)$", "", p.strip("/"))


def _normalize_nav_entry(entry: Any) -> Any:
    """Convert one mkdocs nav entry to a Mintlify navigation entry.

    Returns either:
      - a string (page slug)
      - a dict {"group": ..., "pages": [...]}  (Mintlify group)
    """
    if isinstance(entry, str):
        return _strip_ext(entry)

    if isinstance(entry, dict):
        if len(entry) != 1:
            # Unexpected — multi-key dicts shouldn't appear in mkdocs nav.
            # Fall back to flattening: take the first key.
            key = next(iter(entry))
        else:
            key = next(iter(entry))
        value = entry[key]

        # Single string value → titled page. Mintlify doesn't have a per-page
        # title override at this level (titles come from frontmatter); we drop
        # the title and use just the page slug.
        if isinstance(value, str):
            return _strip_ext(value)

        # List value → group with pages.
        if isinstance(value, list):
            pages = [_normalize_nav_entry(e) for e in value]
            return {"group": key, "pages": pages}

    # Unknown shape — drop silently.
    return None


def _prefix_page_refs(entry: Any, prefix: str) -> Any:
    """Prefix page slugs in a converted navigation tree."""
    if not prefix:
        return entry
    if isinstance(entry, str):
        return f"{prefix.rstrip('/')}/{entry.lstrip('/')}"
    if isinstance(entry, dict):
        out = dict(entry)
        pages = out.get("pages")
        if isinstance(pages, list):
            out["pages"] = [_prefix_page_refs(page, prefix) for page in pages]
        return out
    return entry


def convert_nav(mkdocs_nav: list) -> list:
    """Walk top-level entries and convert. Strip Nones."""
    out = []
    for entry in mkdocs_nav:
        converted = _normalize_nav_entry(entry)
        if converted is None:
            continue
        out.append(converted)
    return out


def parse_mkdocs_yml(path: Path) -> dict:
    """Parse mkdocs.yml leniently. mkdocs.yml can contain python tags
    (e.g. !!python/name:foo) that yaml.safe_load rejects. We use a custom
    loader that treats unknown tags as null."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('Mintlify navigation conversion requires dogbench[cloud] (PyYAML)') from exc
    class PermissiveLoader(yaml.SafeLoader):
        pass

    def _unknown(loader, tag_suffix, node):  # noqa: ARG001
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return None

    PermissiveLoader.add_multi_constructor("", _unknown)
    PermissiveLoader.add_multi_constructor("tag:", _unknown)
    PermissiveLoader.add_multi_constructor("!", _unknown)

    text = path.read_text(encoding="utf-8")
    return yaml.load(text, Loader=PermissiveLoader)  # noqa: S506


def walk_docs_dir(docs_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Build a Mintlify-compatible `navigation` block by walking a docs/
    directory. Used when the project has no usable mkdocs `nav:` block —
    e.g., ruff's mkdocs.yml is just `INHERIT: mkdocs.generated.yml` and the
    generated file isn't checked in.

    Page slugs are stored as paths *relative to the repo root* (since
    Mintlify resolves them from the project root), with the .md/.mdx/.rst
    extension stripped.
    """
    if repo_root is None:
        repo_root = docs_dir.parent
    if not docs_dir.is_dir():
        return {}

    # Collect docs files keyed by their top-level dir under docs_dir.
    # Files directly under docs_dir become top-level pages.
    top_pages: list[str] = []
    by_group: dict[str, list[str]] = {}
    for md in sorted(docs_dir.rglob("*")):
        if not md.is_file():
            continue
        if md.suffix.lower() not in (".md", ".mdx", ".rst"):
            continue
        rel_to_root = md.relative_to(repo_root).as_posix()
        slug = _strip_ext(rel_to_root)
        rel_to_docs = md.relative_to(docs_dir)
        if len(rel_to_docs.parts) == 1:
            top_pages.append(slug)
        else:
            group = rel_to_docs.parts[0]
            by_group.setdefault(group, []).append(slug)

    nav: dict[str, Any] = {}
    if top_pages:
        nav["pages"] = top_pages
    groups = [
        {"group": g.replace("-", " ").replace("_", " ").title(), "pages": pages}
        for g, pages in sorted(by_group.items()) if pages
    ]
    if groups:
        nav["groups"] = groups
    return nav


def build_docs_json(
    mkdocs: dict,
    *,
    name: str = "Project Docs",
    page_prefix: str = "",
    fallback_docs_dir: Path | None = None,
    fallback_repo_root: Path | None = None,
) -> dict:
    """Turn an mkdocs config dict into a minimal Mintlify docs.json.

    Mintlify's navigation schema requires top-level entries to be split:
      - bare page slugs go in `navigation.pages` (strings)
      - groups (with their own pages list) go in `navigation.groups` (objects)
    Inside a group, both strings (pages) and nested group objects are allowed
    together. Only the top level needs the split.

    If the mkdocs config yields an empty navigation (e.g. `INHERIT:`-only
    configs like ruff's), and `fallback_docs_dir` is supplied, the nav is
    built by walking that directory instead. Mintlify rejects docs.json
    files whose `navigation` field has none of the required subfields
    (pages, groups, tabs, …), so we must produce at least one.
    """
    nav = mkdocs.get("nav") or []
    converted = [_prefix_page_refs(entry, page_prefix) for entry in convert_nav(nav)]

    top_pages: list[str] = []
    top_groups: list[dict] = []
    for entry in converted:
        if isinstance(entry, str):
            top_pages.append(entry)
        elif isinstance(entry, dict):
            top_groups.append(entry)

    navigation: dict[str, Any] = {}
    if top_pages:
        navigation["pages"] = top_pages
    if top_groups:
        navigation["groups"] = top_groups

    if not navigation and fallback_docs_dir is not None:
        navigation = walk_docs_dir(fallback_docs_dir, repo_root=fallback_repo_root)

    return {
        "$schema": "https://mintlify.com/docs.json",
        "name": name,                    # NEUTRAL — never the upstream name
        "theme": "mint",
        "colors": {"primary": "#0a7e8c"},
        "navigation": navigation,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="Path to mkdocs.yml")
    p.add_argument("--out", dest="out_path", type=Path, required=True,
                   help="Path to write docs.json")
    p.add_argument("--name", default="Project Docs",
                   help="Mintlify project name (default: 'Project Docs' — "
                        "MUST be anonymized, never the upstream's name)")
    args = p.parse_args()

    mkdocs = parse_mkdocs_yml(args.in_path)
    docs = build_docs_json(mkdocs, name=args.name)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"[mkdocs2mintlify] wrote {args.out_path}")
    print(f"[mkdocs2mintlify] top-level groups: "
          f"{[g.get('group') if isinstance(g, dict) else g for g in docs['navigation']['groups']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
