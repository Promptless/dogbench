"""Command-line interface for public DogBench utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .audit import audit_prediction_bundles
from .items import ItemValidationError, load_items
from .predictions import (
    PredictionValidationError,
    load_item_ids,
    load_records,
    validate_predictions,
)
from .workspaces import WorkspacePreparationError, prepare_items
from .verify import verify_output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dogbench",
        description="Prepare, run, validate, score, and report DogBench evaluations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a DogBench predictions JSONL or JSON file.",
    )
    validate.add_argument(
        "predictions",
        type=Path,
        nargs="+",
        help="One or more prediction bundles produced by evaluated agents.",
    )
    validate.add_argument(
        "--items",
        type=Path,
        help="DogBench construction records used for mandatory contamination checks.",
    )
    validate.add_argument(
        "--traces",
        type=Path,
        help=(
            "Trace root containing <bundle-name>/<instance-id>/ or "
            "<instance-id>/ files."
        ),
    )
    validate.add_argument(
        "--allow-missing-traces",
        action="store_true",
        help="Explicitly accept audit-incomplete candidates without primary traces.",
    )
    validate.add_argument(
        "--offline",
        action="store_true",
        help="Skip live GitHub PR re-attestation and record the exception.",
    )
    validate.add_argument(
        "--format-only",
        action="store_true",
        help="Only check prediction syntax; do not treat the result as canonical validation.",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Print the validation summary as JSON.",
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="Prepare anonymous local workspaces from DogBench items.",
    )
    prepare.add_argument("items", type=Path, help="DogBench item JSON or JSONL file.")
    prepare.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New workspaces are created under this directory.",
    )
    prepare.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Skip live GitHub PR attestation. Intended only for local fixtures or "
            "already-attested mirrors; the exception is recorded in the output."
        ),
    )
    prepare.add_argument(
        "--json",
        action="store_true",
        help="Print the preparation summary as JSON.",
    )

    verify = subparsers.add_parser(
        "verify",
        help="Verify prepared workspaces and attestations without network access.",
    )
    verify.add_argument("output", type=Path, help="Directory created by dogbench prepare.")
    verify.add_argument("--json", action="store_true", help="Print the report as JSON.")

    run = subparsers.add_parser("run", help="Run an agent on sealed prepared inputs.")
    run.add_argument("items", type=Path)
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--agent", choices=["claude", "codex", "opencode", "devin", "mintlify", "promptless"], required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--timeout", type=int, default=1800, help="Per-item timeout in seconds.")
    run.add_argument("--budget-usd", type=float, default=5.0)
    run.add_argument("--config", type=Path, help="Explicit provider/mirror configuration JSON.")
    run.add_argument("--json", action="store_true")

    score = subparsers.add_parser("score", help="Score saved predictions against frozen outcomes and rubrics.")
    score.add_argument("predictions", type=Path)
    score.add_argument("--outcomes", type=Path, required=True)
    score.add_argument("--rubrics", type=Path, help="Optional directory of frozen rubrics; outcomes may embed rubric_markdown.")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--judgments", type=Path, help="Reuse saved judgments JSON without calling the judge.")
    score.add_argument("--model", default="gpt-5.6-terra")
    score.add_argument("--max-attempts", type=int, default=20)
    score.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="Aggregate a saved score bundle without model calls.")
    report.add_argument("scores", type=Path)
    report.add_argument("--outcomes", type=Path, help="Expected task membership for coverage checks.")
    report.add_argument("--output", type=Path, help="Write a JSON report to this file.")
    report.add_argument("--json", action="store_true")

    return parser


def _run_validate(args: argparse.Namespace) -> int:
    if args.format_only:
        if args.traces or args.allow_missing_traces or args.offline:
            raise PredictionValidationError(
                "--format-only cannot be combined with contamination-audit options"
            )
        item_ids = load_item_ids(args.items) if args.items else None
        validations = [
            validate_predictions(
                load_records(path),
                item_ids=item_ids,
                require_complete=item_ids is not None,
            )
            for path in args.predictions
        ]
        summary = {
            "schema_version": "dogbench-format-validation-v1",
            "valid": True,
            "canonical": False,
            "warning": "contamination controls were explicitly disabled",
            "bundles": validations,
        }
        exit_code = 0
    else:
        if args.items is None:
            raise PredictionValidationError(
                "canonical validation requires --items; use --format-only only for syntax checks"
            )
        items = load_items(args.items)
        item_ids = {item["instance_id"] for item in items}
        bundles = []
        validations = []
        for path in args.predictions:
            records = load_records(path)
            validations.append(
                validate_predictions(records, item_ids=item_ids, require_complete=True)
            )
            bundles.append((path, records))
        contamination = audit_prediction_bundles(
            items=items,
            bundles=bundles,
            trace_root=args.traces.resolve() if args.traces else None,
            allow_missing_traces=args.allow_missing_traces,
            offline=args.offline,
        )
        summary = {
            "schema_version": "dogbench-canonical-validation-v1",
            "valid": contamination["ok"],
            "canonical": not args.offline and not args.allow_missing_traces,
            "bundles": validations,
            "contamination": contamination,
        }
        exit_code = 0 if contamination["ok"] else 3
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        prediction_count = sum(row["predictions"] for row in validations)
        print(f"Validated {prediction_count} DogBench prediction(s)")
        if args.format_only:
            print("Scope: format only; contamination controls explicitly disabled")
        else:
            print(
                f"Contamination gate: {contamination['quarantined']} "
                f"of {contamination['candidates']} candidate(s) quarantined"
            )
            print(f"Batch decision: {contamination['decision']}")
    return exit_code


def _run_prepare(args: argparse.Namespace) -> int:
    summary = prepare_items(
        load_items(args.items), args.output.resolve(), offline=args.offline
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Prepared {summary['prepared']} DogBench workspace(s) in {summary['output']}")
        for item in summary["items"]:
            print(f"- {item['instance_id']}: {item['path']}")
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    summary = verify_output(args.output.resolve())
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Verified {summary['verified']} sealed DogBench workspace(s)")
        for item in summary["items"]:
            print(f"- {item['instance_id']}: {item['status']} ({item['workspace']})")
    return 0


def _run_agent(args: argparse.Namespace) -> int:
    from .execution import run_prepared

    if args.timeout <= 0 or args.budget_usd <= 0:
        raise ValueError("--timeout and --budget-usd must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("--config must contain a JSON object")
    summary = run_prepared(
        args.items.resolve(), args.prepared.resolve(), args.output.resolve(),
        agent=args.agent, model=args.model, timeout_s=args.timeout,
        budget_usd=args.budget_usd, config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else f"Run artifacts: {args.output.resolve()}")
    if "exit_code" in summary:
        return int(summary["exit_code"])
    if str(summary.get("stopped") or "").startswith("CLAUDE_RATE_LIMIT:"):
        return 2
    if "contamination" in str(summary.get("stopped") or ""):
        return 3
    return 0 if summary["complete"] and not summary["failed"] else 1


def _run_score(args: argparse.Namespace) -> int:
    from .judge import score_predictions
    from .reporting import build_report

    predictions = load_records(args.predictions)
    outcomes = load_records(args.outcomes)
    saved = json.loads(args.judgments.read_text(encoding="utf-8")) if args.judgments else None
    if saved is not None and not isinstance(saved, dict):
        raise ValueError("--judgments must contain a JSON object (a judgment or an instance_id mapping)")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    scores = score_predictions(
        predictions, outcomes, args.rubrics, args.output.resolve(),
        saved_judgments=saved, model=args.model, max_attempts=args.max_attempts,
    )
    report = build_report(scores, expected_ids=[row["instance_id"] for row in outcomes])
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else f"Scores and report: {args.output.resolve()}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from .reporting import build_report

    bundle = json.loads(args.scores.read_text(encoding="utf-8"))
    scores = bundle.get("scores") if isinstance(bundle, dict) else bundle
    expected_ids = bundle.get("expected_ids") if isinstance(bundle, dict) else None
    if args.outcomes:
        expected_ids = [row["instance_id"] for row in load_records(args.outcomes)]
    if not isinstance(scores, list):
        raise ValueError("score file must contain a list or an object with a scores list")
    report = build_report(scores, expected_ids=expected_ids)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else report["summary"], indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "prepare":
            return _run_prepare(args)
        if args.command == "verify":
            return _run_verify(args)
        if args.command == "run":
            return _run_agent(args)
        if args.command == "score":
            return _run_score(args)
        if args.command == "report":
            return _run_report(args)
    except (PredictionValidationError, ItemValidationError, WorkspacePreparationError, ValueError, OSError, RuntimeError) as exc:
        print(f"dogbench: {args.command} failed\n{exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2
