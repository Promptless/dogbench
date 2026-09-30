"""Load and validate DogBench prediction bundles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


COMPLETED = "completed"
FAILURE_STATUSES = {"error", "timeout", "invalid"}
VALID_STATUSES = {COMPLETED, *FAILURE_STATUSES}
VALID_DECISIONS = {"patch", "abstain"}


class PredictionValidationError(ValueError):
    """Raised when a prediction bundle violates the public schema."""


def _format_ids(values: Iterable[str], *, limit: int = 10) -> str:
    materialized = list(values)
    shown = ", ".join(materialized[:limit])
    remaining = len(materialized) - limit
    return f"{shown} (+{remaining} more)" if remaining > 0 else shown


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionValidationError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_records(path: Path, *, collection_key: str = "predictions") -> list[dict[str, Any]]:
    """Load records from JSONL, a JSON array, or a keyed JSON object."""
    if not path.is_file():
        raise PredictionValidationError(f"file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise PredictionValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise PredictionValidationError(
                    f"{path}:{line_number}: each JSONL record must be an object"
                )
            records.append(record)
        return records

    payload = _read_json(path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get(collection_key), list):
        records = payload[collection_key]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise PredictionValidationError(
            f"{path}: expected a JSON array, object, or {collection_key!r} array"
        )

    if not all(isinstance(record, dict) for record in records):
        raise PredictionValidationError(f"{path}: every record must be an object")
    return records


def load_item_ids(path: Path) -> set[str]:
    """Load item IDs from an index/JSONL file or a directory of item JSON files."""
    if path.is_dir():
        records: list[dict[str, Any]] = []
        for item_path in sorted(path.glob("*.json")):
            payload = _read_json(item_path)
            if isinstance(payload, dict) and any(
                key in payload for key in ("instance_id", "item_id", "id")
            ):
                records.append(payload)
    else:
        records = load_records(path, collection_key="items")

    item_ids: list[str] = []
    for index, record in enumerate(records, start=1):
        item_id = record.get("instance_id") or record.get("item_id") or record.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise PredictionValidationError(
                f"{path}: item record {index} has no non-empty instance_id, item_id, or id"
            )
        item_ids.append(item_id)

    duplicates = sorted(item_id for item_id, count in Counter(item_ids).items() if count > 1)
    if duplicates:
        raise PredictionValidationError(
            f"{path}: duplicate item IDs: {_format_ids(duplicates)}"
        )
    if not item_ids:
        raise PredictionValidationError(f"{path}: no benchmark items found")
    return set(item_ids)


def _validate_prediction(record: dict[str, Any], index: int) -> list[str]:
    label = f"record {index}"
    errors: list[str] = []

    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        errors.append(f"{label}: instance_id must be a non-empty string")
        instance_id = label
    else:
        label = instance_id

    status = record.get("status", COMPLETED)
    if not isinstance(status, str) or status not in VALID_STATUSES:
        errors.append(
            f"{label}: status must be one of {', '.join(sorted(VALID_STATUSES))}"
        )
        return errors

    decision = record.get("decision")
    patch = record.get("patch")
    if status == COMPLETED:
        if not isinstance(decision, str) or decision not in VALID_DECISIONS:
            errors.append(
                f"{label}: a completed prediction needs decision=patch or decision=abstain"
            )
        elif decision == "patch" and (not isinstance(patch, str) or not patch.strip()):
            errors.append(f"{label}: decision=patch requires a non-empty patch string")
        elif decision == "abstain" and patch not in (None, ""):
            errors.append(f"{label}: decision=abstain must not include a patch")
    else:
        if "decision" in record:
            errors.append(f"{label}: status={status} must not include a decision")
        if "patch" in record:
            errors.append(f"{label}: status={status} must not include a patch")
        error = record.get("error")
        if not isinstance(error, str) or not error.strip():
            errors.append(f"{label}: status={status} requires a non-empty error string")

    agent = record.get("agent")
    if agent is not None and not isinstance(agent, dict):
        errors.append(f"{label}: agent must be an object when provided")

    usage = record.get("usage")
    if usage is not None and not isinstance(usage, dict):
        errors.append(f"{label}: usage must be an object when provided")

    return errors


def validate_predictions(
    records: Iterable[dict[str, Any]],
    *,
    item_ids: set[str] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate predictions and return a compact bundle summary."""
    materialized = list(records)
    errors: list[str] = []
    ids: list[str] = []

    if not materialized:
        errors.append("prediction bundle is empty")

    for index, record in enumerate(materialized, start=1):
        errors.extend(_validate_prediction(record, index))
        instance_id = record.get("instance_id")
        if isinstance(instance_id, str) and instance_id.strip():
            ids.append(instance_id)

    duplicates = sorted(instance_id for instance_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate prediction IDs: {_format_ids(duplicates)}")

    predicted_ids = set(ids)
    missing: list[str] = []
    if item_ids is not None:
        unknown = sorted(predicted_ids - item_ids)
        if unknown:
            errors.append(
                f"prediction IDs not present in the item set: {_format_ids(unknown)}"
            )
        missing = sorted(item_ids - predicted_ids)
        if require_complete and missing:
            errors.append(
                f"missing predictions for {len(missing)} item(s): {_format_ids(missing)}"
            )

    if errors:
        raise PredictionValidationError("\n".join(errors))

    status_counts = Counter(record.get("status", COMPLETED) for record in materialized)
    decision_counts = Counter(
        record.get("decision")
        for record in materialized
        if record.get("status", COMPLETED) == COMPLETED
    )
    return {
        "schema_version": "dogbench-predictions-v1",
        "valid": True,
        "predictions": len(materialized),
        "items": len(item_ids) if item_ids is not None else None,
        "missing": len(missing),
        "statuses": dict(sorted(status_counts.items())),
        "decisions": dict(sorted(decision_counts.items())),
    }
