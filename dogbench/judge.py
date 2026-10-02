"""Canonical frozen-rubric judging and score aggregation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERRA_MODEL = "gpt-5.6-terra"



MAX_SCORE_ATTEMPTS = 20



CRITERION_WEIGHT = 1.0



P0_SCORE_CEILING = 60.0



SCORING_CONTRACT = "requirement-conditional-deduction-only-v1"



CRITERION_RE = re.compile(
    r"(?m)^(?:#{2,6}\s+|\d+\.\s+\*\*)"
    r"(C\d+[A-Za-z]?)\s*(?:[—–-]|:)\s+"
)



PRIORITY_RE = re.compile(r"(?m)^\s*\*\*Priority:\*\*\s*(P[0-3])\b")



CATEGORY_RE = re.compile(r"\*\*Category:\*\*\s*([^|\n]+)")



SCORING_RE = re.compile(
    r"\*\*Scoring:\*\*\s*(requirement|conditional|deduction[_ -]only)\b",
    re.IGNORECASE,
)



def now() -> str:
    return datetime.now(timezone.utc).isoformat()



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as h:
        json.dump(value, h, indent=2, sort_keys=True)
        h.write("\n")
        tmp = Path(h.name)
    tmp.replace(path)



def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value



def criterion_blocks(rubric: str) -> dict[str, dict[str, Any]]:
    matches = list(CRITERION_RE.finditer(rubric))
    result: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        criterion_id = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rubric)
        block = rubric[match.start():end]
        priority = PRIORITY_RE.search(block)
        category = CATEGORY_RE.search(block)
        scoring = SCORING_RE.search(block)
        if not priority or not category or not scoring or criterion_id in result:
            raise ValueError(f"invalid rubric metadata for {criterion_id}")
        scoring_type = scoring.group(1).lower().replace("-", "_").replace(" ", "_")
        result[criterion_id] = {
            "text": block.strip(),
            "priority": priority.group(1),
            "category": category.group(1).strip(),
            "scoring_type": scoring_type,
        }
    if not result:
        raise ValueError("rubric has no canonical criterion headings")
    return result



def score_schema(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = list(blocks)
    criterion_variants = []
    for criterion_id, block in blocks.items():
        scoring_type = block["scoring_type"]
        is_requirement = scoring_type == "requirement"
        is_conditional = scoring_type == "conditional"
        is_guardrail = scoring_type == "deduction_only"
        criterion_variants.append({
            "type": "object", "additionalProperties": False,
            "required": ["id", "scoring_type", "triggered", "verdict", "violated",
                         "triggering_patch_quote", "supporting_patch_quote",
                         "contradicting_patch_quote", "missing_requirement",
                         "reason"],
            "properties": {
                "id": {"type": "string", "enum": [criterion_id]},
                "scoring_type": {"type": "string", "enum": [scoring_type]},
                "triggered": ({"type": "boolean"} if is_conditional else {"type": "null"}),
                "triggering_patch_quote": {"type": ["string", "null"]},
                "supporting_patch_quote": {"type": ["string", "null"]},
                "contradicting_patch_quote": {"type": ["string", "null"]},
                "missing_requirement": {"type": ["string", "null"]},
                "verdict": (
                    {"type": "string", "enum": ["pass", "fail"]} if is_requirement else
                    {"type": ["string", "null"], "enum": ["pass", "fail", None]}
                    if is_conditional else {"type": "null"}
                ),
                "violated": ({"type": "boolean"} if is_guardrail else {"type": "null"}),
                "reason": {"type": "string"},
            },
        })
    return {
        "type": "object", "additionalProperties": False,
        "required": ["criteria", "summary"],
        "properties": {
            "criteria": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
                         "items": {"anyOf": criterion_variants}},
            "summary": {"type": "string"},
        },
    }



def score_prompt(rubric: str, patch: str, cell_id: str, correction: str = "") -> str:
    return f"""You are Terra scoring one anonymous documentation patch against one frozen rubric.

Evaluate every criterion exactly once and in rubric order. Evidence precedes verdict. Equivalent
wording counts. Ignore non-English, localized, and translated content entirely; never award or
deduct credit for its presence, absence, correctness, freshness, language, structure, links, or
synchronization. The rubric explicitly declares each criterion as requirement, conditional, or
deduction_only. For a requirement, return verdict Pass or Fail. For a conditional criterion,
decide triggered first; when triggered return Pass or Fail, and when not triggered return a null
verdict. For deduction_only, return violated true only when the patch actually contains the
prohibited content; otherwise return violated false. A not-triggered conditional and a clean
deduction-only guardrail earn no positive credit. Partial and N/A do not exist in this contract.
Never mark a guardrail violated without an exact prohibited patch quote, and never mark a
conditional triggered without an exact activating patch quote.
Do not use outside files, tools, or web research. Return strict JSON only.

For supporting_patch_quote and contradicting_patch_quote, copy a short contiguous substring
verbatim from PATCH; never paraphrase, normalize, or insert ellipses. A failed requirement must
name a missing requirement or quote contradicting text. A violated deduction-only guardrail must
put the exact prohibited text in contradicting_patch_quote. A triggered conditional must put the
exact activating text in triggering_patch_quote; otherwise that field must be null.
{correction}

CELL: {cell_id}
RUBRIC:\n<rubric>\n{rubric}\n</rubric>
PATCH:\n<patch>\n{patch}\n</patch>
"""



def codex_command(model: str, schema: Path, output: Path) -> list[str]:
    return ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
            "-c", 'model_reasoning_effort="high"', "--model", model,
            "--output-schema", str(schema), "--output-last-message", str(output), "-"]



def run_codex(prompt: Path, schema: Path, output: Path, log: Path, model: str) -> None:
    with prompt.open("rb") as source, log.open("wb") as destination:
        completed = subprocess.run(codex_command(model, schema, output), stdin=source,
                                   stdout=destination, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Codex exited {completed.returncode}; see {log}")



def validate_score(payload: dict[str, Any], blocks: dict[str, dict[str, Any]], patch: str) -> None:
    # Scorers quote authored text without unified-diff markers. Match only the
    # added lines or the post-change projection, excluding deleted content;
    # fuzzy or semantic-only matches remain invalid.
    added_view = " ".join("\n".join(
        line[1:] for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ).split())
    # Exact evidence may legitimately span unchanged context and newly added
    # lines. Build the authored post-patch projection by stripping the unified
    # diff marker from context/addition lines while excluding removals and diff
    # metadata. This remains exact, deterministic matching—not fuzzy quoting.
    authored_view = " ".join("\n".join(
        line[1:] for line in patch.splitlines()
        if ((line.startswith("+") and not line.startswith("+++"))
            or (line.startswith(" ") and not line.startswith(" @@")))
    ).split())
    # A scorer may quote the rendered JSON payload from a shell example rather
    # than its source-level shell escaping.  This remains a deterministic exact
    # projection: only escaped quote characters are unescaped; no fuzzy or
    # semantic matching is introduced.
    rendered_added_view = added_view.replace(r'\"', '"')
    rows = payload.get("criteria")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != list(blocks):
        raise ValueError("score must cover every criterion exactly once in rubric order")
    for row in rows:
        cid = row["id"]
        scoring_type = blocks[cid]["scoring_type"]
        if row.get("scoring_type") != scoring_type:
            raise ValueError(f"{cid}: scoring_type does not match rubric metadata")
        if scoring_type == "deduction_only":
            if (row.get("triggered") is not None or row.get("verdict") is not None
                    or not isinstance(row.get("violated"), bool)):
                raise ValueError(f"{cid}: guardrail requires violated boolean and null verdict")
            if row.get("triggering_patch_quote") is not None:
                raise ValueError(f"{cid}: guardrail cannot have a triggering patch quote")
            if row["violated"] and not row.get("contradicting_patch_quote"):
                raise ValueError(f"{cid}: violated guardrail lacks exact prohibited patch quote")
            if row.get("missing_requirement"):
                raise ValueError(f"{cid}: guardrail cannot be failed for omitted content")
        elif scoring_type == "requirement":
            verdict = row.get("verdict")
            if (row.get("triggered") is not None or verdict not in {"pass", "fail"}
                    or row.get("violated") is not None):
                raise ValueError(f"{cid}: requirement requires Pass/Fail only")
            if row.get("triggering_patch_quote") is not None:
                raise ValueError(f"{cid}: requirement cannot have a triggering patch quote")
        else:
            triggered = row.get("triggered")
            if not isinstance(triggered, bool) or row.get("violated") is not None:
                raise ValueError(f"{cid}: conditional requires triggered boolean")
            if triggered and row.get("verdict") not in {"pass", "fail"}:
                raise ValueError(f"{cid}: triggered conditional requires Pass/Fail")
            if not triggered and row.get("verdict") is not None:
                raise ValueError(f"{cid}: untriggered conditional requires null verdict")
            if triggered and not row.get("triggering_patch_quote"):
                raise ValueError(f"{cid}: triggered conditional lacks exact activating patch quote")
            if not triggered and row.get("triggering_patch_quote") is not None:
                raise ValueError(f"{cid}: untriggered conditional cannot have a triggering quote")
        if scoring_type != "deduction_only" and row.get("verdict") == "fail" and not (
                row.get("contradicting_patch_quote") or row.get("missing_requirement")):
            raise ValueError(f"{cid}: fail lacks contradiction or missing requirement")
        for field in ("triggering_patch_quote", "supporting_patch_quote",
                      "contradicting_patch_quote"):
            quote = row.get(field)
            normalized = " ".join(quote.split()) if quote else ""
            if (quote and normalized not in added_view and normalized not in rendered_added_view
                    and normalized not in authored_view):
                raise ValueError(f"{cid}: {field} is not an exact patch quote")



def numeric_score(payload: dict[str, Any], blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    requirement_credit = requirement_total = guardrail_penalty = 0.0
    blocking: list[str] = []
    for row in payload["criteria"]:
        block = blocks[row["id"]]
        weight = CRITERION_WEIGHT
        scoring_type = block["scoring_type"]
        active = scoring_type == "requirement" or (
            scoring_type == "conditional" and row["triggered"]
        )
        failed = row["violated"] if scoring_type == "deduction_only" else (
            active and row["verdict"] == "fail"
        )
        if scoring_type == "deduction_only":
            if row["violated"]:
                guardrail_penalty += weight
        elif active:
            requirement_total += weight
            if row["verdict"] == "pass":
                requirement_credit += weight
        if block["priority"] == "P0" and failed:
            blocking.append(row["id"])
    earned_after_penalty = max(0.0, requirement_credit - guardrail_penalty)
    uncapped_overall = (100 * earned_after_penalty / requirement_total
                        if requirement_total else 0.0)
    overall = min(uncapped_overall, P0_SCORE_CEILING) if blocking else uncapped_overall
    return {"scoring_contract": SCORING_CONTRACT,
            "aggregation_policy": "equal-criterion-weight-p0-ceiling-v1",
            "requirement_credit_weight": requirement_credit,
            "requirement_total_weight": requirement_total,
            "guardrail_penalty_weight": guardrail_penalty,
            "uncapped_overall": round(uncapped_overall, 1),
            "overall": round(overall, 1),
            "mergeable": not blocking, "blocking_p0": blocking}



def has_documentation_change(patch: str) -> bool:
    stripped = patch.strip()
    if not stripped or stripped.upper() in {
        "NO_DOC_CHANGES_NEEDED",
        "NO_DOCUMENTATION_CHANGES_NEEDED",
        "NO_CHANGES_NEEDED",
    }:
        return False
    if "diff --git " not in patch:
        return True
    return any(
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
        for line in patch.splitlines()
    )


def judge_patch(
    rubric: str,
    patch: str,
    cell_id: str,
    output_dir: Path,
    *,
    model: str = TERRA_MODEL,
    max_attempts: int = MAX_SCORE_ATTEMPTS,
) -> dict[str, Any]:
    """Run the canonical judge, retaining every attempt and validation error."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = criterion_blocks(rubric)
    schema = output_dir / "schema.json"
    atomic_json(schema, score_schema(blocks))
    correction = ""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_dir = output_dir / "attempts" / f"{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prompt, output, log = (attempt_dir / "prompt.txt", attempt_dir / "terra_score.json",
                               attempt_dir / "codex.log")
        prompt.write_text(score_prompt(rubric, patch, cell_id, correction), encoding="utf-8")
        try:
            run_codex(prompt, schema, output, log, model)
            payload = read_json(output)
            validate_score(payload, blocks, patch)
            payload["numeric"] = numeric_score(payload, blocks)
            atomic_json(output, payload)
            atomic_json(output_dir / "judgment.json", payload)
            atomic_json(output_dir / "judge-metadata.json", {
                "model": model, "reasoning_effort": "high", "attempt_count": attempt,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
                "prompt_sha256": sha256(prompt), "schema_sha256": sha256(schema),
                "output_sha256": sha256(output), "log_sha256": sha256(log),
                "scoring_contract": SCORING_CONTRACT,
            })
            return payload
        except Exception as exc:
            last_error = exc
            atomic_json(attempt_dir / "validation_error.json", {
                "schema_version": "dogbench-terra-score-attempt-error-v1",
                "created_at": now(), "error": repr(exc),
            })
            correction = ("CORRECTION FROM THE PREVIOUS REJECTED ATTEMPT: "
                          f"{exc}. Return a fully corrected result; do not defend the prior output. "
                          "For evidence quotes, copy the authored text verbatim from the unified diff, "
                          "omitting only the leading '+' marker; preserve literal backslashes and other "
                          "punctuation exactly. Never quote rubric, research, repository, or inferred "
                          "text in a patch-quote field. When a requirement Fail is caused only by omitted required "
                          "content, set contradicting_patch_quote to null and describe the omission in "
                          "missing_requirement instead of inventing a contradiction quote. For Pass or "
                          "a non-violated guardrail, prefer null over any supporting quote you cannot copy exactly from PATCH.")
    raise RuntimeError(f"score failed {max_attempts} validation attempts: {last_error}")


def _rubric(outcome: dict[str, Any], rubrics_dir: Path | None) -> str:
    text = outcome.get("rubric_markdown")
    if text is None and rubrics_dir is not None:
        name = outcome.get("rubric_path") or f"{outcome['instance_id']}.md"
        text = (rubrics_dir / Path(name).name).read_text(encoding="utf-8")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"missing rubric: {outcome['instance_id']}")
    expected = outcome.get("rubric_sha256")
    if expected is not None and hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError(f"rubric hash mismatch: {outcome['instance_id']}")
    return text


def _saved_judgment(
    saved: dict[str, Any], instance_id: str, patch_hash: str, rubric_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # A raw original JSON object is accepted for a single patch. An ID-keyed
    # mapping may contain raw objects or hash-bound {judgment, ...} envelopes.
    entry = saved if "criteria" in saved else saved.get(instance_id)
    if not isinstance(entry, dict):
        raise ValueError(f"missing saved judgment: {instance_id}")
    provided_hashes = {}
    for field, expected in (("patch_sha256", patch_hash), ("rubric_sha256", rubric_hash)):
        if field in entry and entry[field] != expected:
            raise ValueError(f"saved judgment {field} mismatch: {instance_id}")
        if field in entry:
            provided_hashes[field] = entry[field]
    payload = entry.get("judgment", entry)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid saved judgment: {instance_id}")
    binding_status = (
        "both_hashes_verified" if len(provided_hashes) == 2
        else "patch_only_verified" if "patch_sha256" in provided_hashes
        else "rubric_only_verified" if "rubric_sha256" in provided_hashes
        else "unverified"
    )
    return (deepcopy(payload),
            {"status": binding_status, "provided_hashes": provided_hashes},
            deepcopy(entry))


def score_predictions(
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    rubrics_dir: Path | None,
    output_dir: Path,
    *,
    saved_judgments: dict[str, Any] | None = None,
    model: str = TERRA_MODEL,
    max_attempts: int = MAX_SCORE_ATTEMPTS,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Score public prediction records; saved judgments never call a model.

    Live judging invokes the host Codex CLI and can incur model charges. Use a
    fresh output directory. Raw saved judgments are preserved as supplied; when
    an envelope provides patch/rubric hashes, those hashes must match. The input
    envelope and its binding status are retained separately from the judgment.
    Evidence validation alone cannot establish the provenance of an unhashed
    judgment.
    """
    from .predictions import validate_predictions

    by_id: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        instance_id = outcome.get("instance_id")
        if (not isinstance(instance_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", instance_id)
                or instance_id in by_id):
            raise ValueError(f"invalid or duplicate outcome ID: {instance_id!r}")
        if outcome.get("expected_outcome") not in {"patch", "abstain"}:
            raise ValueError(f"invalid expected outcome: {instance_id}")
        by_id[instance_id] = outcome
    validate_predictions(predictions, item_ids=set(by_id), require_complete=require_complete)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    if saved_judgments is not None and "criteria" in saved_judgments:
        patch_count = sum(
            p.get("status", "completed") == "completed" and p.get("decision") == "patch"
            and by_id[p["instance_id"]]["expected_outcome"] == "patch"
            and has_documentation_change(p.get("patch", "")) for p in predictions
        )
        if patch_count != 1:
            raise ValueError("a raw saved judgment requires exactly one scored patch")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for prediction in predictions:
        instance_id = prediction["instance_id"]
        outcome = by_id[instance_id]
        status = prediction.get("status", "completed")
        decision = prediction.get("decision") if status == "completed" else None
        patch = prediction.get("patch", "") or ""
        expected = outcome["expected_outcome"]
        correct = status == "completed" and decision == expected
        if decision == "patch" and not has_documentation_change(patch):
            correct = False
        row: dict[str, Any] = {
            "instance_id": instance_id, "expected_outcome": expected,
            "status": status, "decision": decision, "decision_correct": correct,
            "overall": 100.0 if correct and expected == "abstain" else 0.0,
            "mergeable": False, "blocking_p0": [],
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        }
        if correct and expected == "patch":
            rubric = _rubric(outcome, rubrics_dir)
            rubric_hash = hashlib.sha256(rubric.encode()).hexdigest()
            blocks = criterion_blocks(rubric)
            cell_dir = output_dir / "judgments" / instance_id
            if saved_judgments is not None:
                payload, binding, saved_input = _saved_judgment(
                    saved_judgments, instance_id, row["patch_sha256"], rubric_hash,
                )
                validate_score(payload, blocks, patch)
                atomic_json(cell_dir / "judgment.json", payload)
                saved_input_path = cell_dir / "saved-input.json"
                atomic_json(saved_input_path, saved_input)
                row["judgment_binding"] = {
                    **binding,
                    "saved_input_path": saved_input_path.relative_to(output_dir).as_posix(),
                    "saved_input_sha256": sha256(saved_input_path),
                }
                source = "saved_judgments"
            else:
                payload = judge_patch(rubric, patch, instance_id, cell_dir,
                                      model=model, max_attempts=max_attempts)
                source = "codex"
            row.update(numeric_score(payload, blocks))
            row.update(rubric_sha256=rubric_hash, judgment=payload,
                       judgment_source=source, judge_model=model if source == "codex" else None)
        scores.append(row)
    atomic_json(output_dir / "scores.json", {
        "schema_version": "dogbench-scores-v1", "scores": scores,
        "expected_ids": list(by_id),
    })
    return scores
