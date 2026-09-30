"""Load and validate the public DogBench item format."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class ItemValidationError(ValueError):
    """Raised when a benchmark item cannot be safely prepared."""


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ItemValidationError(f"file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ItemValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ItemValidationError(
                    f"{path}:{line_number}: each JSONL record must be an object"
                )
            records.append(record)
        return records

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ItemValidationError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        records = payload["items"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ItemValidationError(
            f"{path}: expected an item object, an array, or an object with an items array"
        )
    if not all(isinstance(record, dict) for record in records):
        raise ItemValidationError(f"{path}: every item must be an object")
    return records


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ItemValidationError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _FULL_SHA.fullmatch(value):
        raise ItemValidationError(f"{label} must be a full 40-character Git SHA")
    return value.lower()


def _paths(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ItemValidationError(f"{label} must be an array")
    paths: list[str] = []
    for index, raw_path in enumerate(value):
        if not isinstance(raw_path, str) or not raw_path:
            raise ItemValidationError(f"{label}[{index}] must be a non-empty string")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or raw_path in {".", ".."}
            or ".." in path.parts
            or ".git" in path.parts
            or "\x00" in raw_path
        ):
            raise ItemValidationError(f"{label}[{index}] must be a safe relative path")
        paths.append(raw_path)
    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        raise ItemValidationError(f"{label} contains duplicate paths: {', '.join(duplicates)}")
    return paths


def _repository_identity(value: str) -> str:
    """Return a stable comparison key for local and hosted Git repository URLs."""
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        identity = f"{parsed.netloc.casefold()}/{parsed.path.strip('/')}"
    else:
        identity = str(Path(value).expanduser().resolve())
    return identity.removesuffix(".git").rstrip("/")


def validate_item(record: dict[str, Any], *, index: int = 1) -> dict[str, Any]:
    """Return one normalized item after validating the public contract."""
    label = f"item {index}"
    instance_id = _required_string(record.get("instance_id"), f"{label}.instance_id")
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise ItemValidationError(
            f"{label}.instance_id may contain only letters, numbers, '.', '_' and '-'"
        )

    source_url = _required_string(record.get("source_url"), f"{instance_id}.source_url")
    context = _required_string(record.get("context"), f"{instance_id}.context")
    docs = record.get("docs")
    if not isinstance(docs, dict):
        raise ItemValidationError(f"{instance_id}.docs must be an object")
    if "head_sha" not in docs:
        raise ItemValidationError(f"{instance_id}.docs.head_sha is required (use null if hidden)")

    normalized_docs = {
        "repo_url": _required_string(
            docs.get("repo_url"), f"{instance_id}.docs.repo_url"
        ),
        "base_sha": _sha(docs.get("base_sha"), f"{instance_id}.docs.base_sha"),
        "head_sha": _sha(
            docs.get("head_sha"), f"{instance_id}.docs.head_sha", optional=True
        ),
        "paths": _paths(docs.get("paths"), f"{instance_id}.docs.paths"),
    }
    if normalized_docs["paths"] and normalized_docs["head_sha"] is None:
        raise ItemValidationError(
            f"{instance_id}.docs.head_sha is required when docs.paths is non-empty"
        )
    if (
        normalized_docs["head_sha"] is not None
        and normalized_docs["base_sha"] == normalized_docs["head_sha"]
        and normalized_docs["paths"]
    ):
        raise ItemValidationError(
            f"{instance_id}.docs.base_sha and docs.head_sha must differ"
        )

    if "code" not in record:
        raise ItemValidationError(f"{instance_id}.code is required (use null for docs-only)")
    raw_code = record["code"]
    normalized_code: dict[str, Any] | None
    if raw_code is None:
        normalized_code = None
    elif isinstance(raw_code, dict):
        normalized_code = {
            "source_url": _required_string(
                raw_code.get("source_url"), f"{instance_id}.code.source_url"
            ),
            "repo_url": _required_string(
                raw_code.get("repo_url"), f"{instance_id}.code.repo_url"
            ),
            "base_sha": _sha(
                raw_code.get("base_sha"), f"{instance_id}.code.base_sha"
            ),
            "head_sha": _sha(
                raw_code.get("head_sha"), f"{instance_id}.code.head_sha"
            ),
            "paths": _paths(raw_code.get("paths"), f"{instance_id}.code.paths"),
        }
        if not normalized_code["paths"]:
            raise ItemValidationError(
                f"{instance_id}.code.paths must not be empty when code is present"
            )
        if normalized_code["base_sha"] == normalized_code["head_sha"]:
            raise ItemValidationError(
                f"{instance_id}.code.base_sha and code.head_sha must differ"
            )
    else:
        raise ItemValidationError(f"{instance_id}.code must be an object or null")

    if (
        normalized_code is not None
        and _repository_identity(normalized_docs["repo_url"])
        == _repository_identity(normalized_code["repo_url"])
    ):
        overlap = sorted(set(normalized_docs["paths"]) & set(normalized_code["paths"]))
        if overlap:
            raise ItemValidationError(
                f"{instance_id} exposes held-out documentation as code: "
                + ", ".join(overlap)
            )

    return {
        "instance_id": instance_id,
        "source_url": source_url,
        "context": context,
        "docs": normalized_docs,
        "code": normalized_code,
    }


def load_items(path: Path) -> list[dict[str, Any]]:
    records = _read_records(path)
    if not records:
        raise ItemValidationError(f"{path}: no benchmark items found")
    items = [validate_item(record, index=index) for index, record in enumerate(records, 1)]
    duplicates = sorted(
        item_id
        for item_id, count in Counter(item["instance_id"] for item in items).items()
        if count > 1
    )
    if duplicates:
        raise ItemValidationError(f"duplicate instance_id values: {', '.join(duplicates)}")
    return items
