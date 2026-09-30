from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dogbench.cli import main
from dogbench.contamination import context_audit
from dogbench.sources import SourceAttestationError, attest_github_component


def write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_returncode(repo: Path, *args: str) -> int:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    ).returncode


def source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "source"
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(repo)], check=True
    )
    git(repo, "config", "user.name", "Source Maintainer")
    git(repo, "config", "user.email", "maintainer@example.com")
    (repo / "docs").mkdir()
    (repo / "src").mkdir()
    (repo / "docs" / "guide.md").write_text("old documentation\n", encoding="utf-8")
    (repo / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "source base")
    base = git(repo, "rev-parse", "HEAD")

    (repo / "docs" / "guide.md").write_text("human reference\n", encoding="utf-8")
    (repo / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "source change")
    return repo, base, git(repo, "rev-parse", "HEAD")


def test_validate_accepts_patch_abstention_and_operational_failure(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "instance_id": "task-patch",
                "decision": "patch",
                "patch": "diff --git a/docs/a.md b/docs/a.md\n",
            },
            {
                "instance_id": "task-abstain",
                "decision": "abstain",
                "reason": "No user-facing behavior changed.",
            },
            {
                "instance_id": "task-timeout",
                "status": "timeout",
                "error": "Agent exceeded the item timeout.",
            },
        ],
    )

    assert main(["validate", str(predictions), "--format-only", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["valid"] is True
    assert summary["canonical"] is False
    assert summary["bundles"][0]["decisions"] == {"abstain": 1, "patch": 1}
    assert summary["bundles"][0]["statuses"] == {"completed": 2, "timeout": 1}


def test_validate_rejects_patch_decision_without_patch(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"instance_id": "task-1", "decision": "patch"}])

    assert main(["validate", str(predictions), "--format-only"]) == 2
    assert "decision=patch requires a non-empty patch string" in capsys.readouterr().err


def test_validate_requires_items_unless_format_only(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"instance_id": "task-1", "decision": "abstain"}])

    assert main(["validate", str(predictions)]) == 2
    assert "canonical validation requires --items" in capsys.readouterr().err


def test_validate_quarantines_missing_trace_by_default(tmp_path, capsys) -> None:
    source, base, head = source_repo(tmp_path)
    item = {
        "instance_id": "trace-required",
        "source_url": "https://example.invalid/pull/1",
        "context": "Clarify existing behavior for users.",
        "docs": {
            "repo_url": str(source),
            "base_sha": base,
            "head_sha": head,
            "paths": ["docs/guide.md"],
        },
        "code": None,
    }
    items = tmp_path / "items.json"
    items.write_text(json.dumps(item), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [{"instance_id": "trace-required", "decision": "abstain"}],
    )

    assert (
        main(
            [
                "validate",
                str(predictions),
                "--items",
                str(items),
                "--offline",
                "--json",
            ]
        )
        == 3
    )
    report = json.loads(capsys.readouterr().out)
    assert (
        report["contamination"]["reports"][0]["hard_findings"][0]["code"]
        == "primary_trace_missing"
    )


def test_validate_checks_item_membership_and_completeness(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    items = tmp_path / "items.json"
    write_jsonl(
        predictions,
        [{"instance_id": "task-1", "decision": "abstain"}],
    )
    items.write_text(
        json.dumps({"items": [{"item_id": "task-1"}, {"id": "task-2"}]}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate",
                str(predictions),
                "--items",
                str(items),
                "--format-only",
            ]
        )
        == 2
    )
    assert "missing predictions for 1 item(s): task-2" in capsys.readouterr().err


def test_validate_rejects_operational_failure_disguised_as_decision(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "instance_id": "task-1",
                "status": "error",
                "decision": "abstain",
                "error": "Agent crashed.",
            }
        ],
    )

    assert main(["validate", str(predictions), "--format-only"]) == 2
    assert "status=error must not include a decision" in capsys.readouterr().err


def test_validate_rejects_patch_field_on_operational_failure(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "instance_id": "task-1",
                "status": "error",
                "patch": "",
                "error": "Agent crashed.",
            }
        ],
    )

    assert main(["validate", str(predictions), "--format-only"]) == 2
    assert "status=error must not include a patch" in capsys.readouterr().err


def test_validate_rejects_empty_bundle(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")

    assert main(["validate", str(predictions), "--format-only"]) == 2
    assert "prediction bundle is empty" in capsys.readouterr().err


def test_validate_reports_wrong_status_and_decision_types(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {"instance_id": "task-1", "status": ["completed"]},
            {"instance_id": "task-2", "decision": {"kind": "patch"}},
        ],
    )

    assert main(["validate", str(predictions), "--format-only"]) == 2
    errors = capsys.readouterr().err
    assert "task-1: status must be one of" in errors
    assert "task-2: a completed prediction needs decision=patch or decision=abstain" in errors


def test_prepare_builds_anonymous_docs_and_filtered_code_repositories(
    tmp_path, capsys
) -> None:
    source, base, head = source_repo(tmp_path)
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {
                "instance_id": "example-pr1",
                "source_url": "https://example.invalid/docs/pull/1",
                "context": "Document the behavior introduced by the implementation change.",
                "docs": {
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["docs/guide.md"],
                },
                "code": {
                    "source_url": "https://example.invalid/code/pull/2",
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["src/app.py"],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "workspaces"

    assert (
        main(
            [
                "prepare",
                str(items),
                "--output",
                str(output),
                "--offline",
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["prepared"] == 1
    assert summary["source_attestation"] == "offline_explicit"

    workspace = Path(summary["items"][0]["path"])
    docs = workspace / "docs"
    code = workspace / "code"
    assert (docs / "docs" / "guide.md").read_text() == "old documentation\n"
    assert (code / "src" / "app.py").read_text() == "VALUE = 2\n"
    assert (code / "docs" / "guide.md").read_text() == "old documentation\n"
    assert git(docs, "remote") == ""
    assert git(code, "remote") == ""
    assert git(docs, "log", "-1", "--format=%s") == "snapshot"
    assert git(code, "log", "-2", "--format=%s").splitlines() == ["change", "snapshot"]
    assert git(code, "diff", "--name-only", "HEAD~1..HEAD") == "src/app.py"
    assert git_returncode(docs, "cat-file", "-e", f"{base}^{{commit}}") != 0
    assert git_returncode(code, "cat-file", "-e", f"{head}^{{commit}}") != 0

    task = (workspace / "TASK.md").read_text()
    assert "Document the behavior" in task
    assert "example.invalid" not in task
    assert base not in task
    manifest = json.loads((workspace / "workspace.json").read_text())
    assert "instance_id" not in manifest
    assert manifest["code"]["path"] == "code"

    assert main(["verify", str(output), "--json"]) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["ok"] is True
    assert verification["items"][0]["instance_id"] == "example-pr1"


def test_prepare_supports_docs_only_items(tmp_path, capsys) -> None:
    source, base, head = source_repo(tmp_path)
    items = tmp_path / "items.jsonl"
    write_jsonl(
        items,
        [
            {
                "instance_id": "docs-only",
                "source_url": "https://example.invalid/issues/1",
                "context": "Clarify the existing behavior for users.",
                "docs": {
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["docs/guide.md"],
                },
                "code": None,
            }
        ],
    )

    assert (
        main(
            [
                "prepare",
                str(items),
                "--output",
                str(tmp_path / "out"),
                "--offline",
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    workspace = Path(summary["items"][0]["path"])
    assert (workspace / "docs").is_dir()
    assert not (workspace / "code").exists()
    assert json.loads((workspace / "workspace.json").read_text())["code"] is None


def test_prepare_supports_issue_without_reference_patch(tmp_path, capsys) -> None:
    source, base, _ = source_repo(tmp_path)
    items = tmp_path / "items.jsonl"
    write_jsonl(items, [{
        "instance_id": "issue-without-reference",
        "source_url": "https://example.invalid/issues/2",
        "context": "Explain when readers should use the existing setting.",
        "docs": {
            "repo_url": str(source), "base_sha": base,
            "head_sha": None, "paths": [],
        },
        "code": None,
    }])
    assert main([
        "prepare", str(items), "--output", str(tmp_path / "out"),
        "--offline", "--json",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    workspace = Path(summary["items"][0]["path"])
    assert (workspace / "docs").is_dir()
    assert not (workspace / "code").exists()
    assert main(["verify", str(tmp_path / "out"), "--json"]) == 0


def test_prepare_rejects_unsafe_instance_id(tmp_path, capsys) -> None:
    source, base, head = source_repo(tmp_path)
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {
                "instance_id": "../escape",
                "source_url": "https://example.invalid/issues/1",
                "context": "Clarify the behavior.",
                "docs": {
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["docs/guide.md"],
                },
                "code": None,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "prepare",
                str(items),
                "--output",
                str(tmp_path / "out"),
                "--offline",
            ]
        )
        == 2
    )
    assert "instance_id may contain only" in capsys.readouterr().err


def test_prepare_rejects_context_copied_from_human_gold(tmp_path, capsys) -> None:
    source, base, _ = source_repo(tmp_path)
    copied_line = "When rollback fails because a resource is missing, recreate it before retrying."
    (source / "docs" / "guide.md").write_text(copied_line + "\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "substantive human reference")
    head = git(source, "rev-parse", "HEAD")
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {
                "instance_id": "contaminated-context",
                "source_url": "https://example.invalid/issues/1",
                "context": copied_line,
                "docs": {
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["docs/guide.md"],
                },
                "code": None,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "prepare",
                str(items),
                "--output",
                str(tmp_path / "out"),
                "--offline",
            ]
        )
        == 2
    )
    assert "heldout_gold_line_overlap" in capsys.readouterr().err


def test_verify_detects_workspace_tampering(tmp_path, capsys) -> None:
    source, base, head = source_repo(tmp_path)
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {
                "instance_id": "tamper-check",
                "source_url": "https://example.invalid/issues/1",
                "context": "Clarify existing behavior for users.",
                "docs": {
                    "repo_url": str(source),
                    "base_sha": base,
                    "head_sha": head,
                    "paths": ["docs/guide.md"],
                },
                "code": None,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert (
        main(
            ["prepare", str(items), "--output", str(output), "--offline", "--json"]
        )
        == 0
    )
    workspace = Path(json.loads(capsys.readouterr().out)["items"][0]["path"])
    (workspace / "TASK.md").write_text("tampered\n", encoding="utf-8")

    assert main(["verify", str(output)]) == 2
    assert "artifact hash mismatch" in capsys.readouterr().err


def test_audit_quarantines_human_patch_copy(tmp_path, capsys) -> None:
    source, base, _ = source_repo(tmp_path)
    copied_line = (
        "When rollback fails because a resource is missing, recreate the resource "
        "and retry the rollback command after verifying the release state is healthy."
    )
    (source / "docs" / "guide.md").write_text(copied_line + "\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "long human reference")
    head = git(source, "rev-parse", "HEAD")
    items = tmp_path / "items.json"
    item = {
        "instance_id": "copied-gold",
        "source_url": "https://example.invalid/issues/1",
        "context": "Clarify existing behavior for users.",
        "docs": {
            "repo_url": str(source),
            "base_sha": base,
            "head_sha": head,
            "paths": ["docs/guide.md"],
        },
        "code": None,
    }
    items.write_text(json.dumps(item), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "instance_id": "copied-gold",
                "decision": "patch",
                "patch": (
                    "diff --git a/docs/guide.md b/docs/guide.md\n"
                    "--- a/docs/guide.md\n"
                    "+++ b/docs/guide.md\n"
                    f"+{copied_line}\n"
                ),
            }
        ],
    )

    assert (
        main(
            [
                "validate",
                str(predictions),
                "--items",
                str(items),
                "--offline",
                "--allow-missing-traces",
                "--json",
            ]
        )
        == 3
    )
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert report["contamination"]["decision"] == "stop"
    assert (
        report["contamination"]["reports"][0]["similarity_alerts"][0]["code"]
        == "human_patch_near_verbatim"
    )


def test_live_source_attestation_uses_immutable_merge_base(monkeypatch) -> None:
    base = "1" * 40
    head = "2" * 40

    def fake_github_json(path: str):
        if path.endswith("/pulls/7"):
            return {"base": {"sha": "3" * 40}, "head": {"sha": head}}
        if "/compare/" in path:
            return {"merge_base_commit": {"sha": base}}
        if "/files?" in path:
            return [{"filename": "src/app.py", "previous_filename": "src/old.py"}]
        if any(
            marker in path
            for marker in ("/comments?", "/reviews?", "/commits?")
        ):
            return []
        raise AssertionError(path)

    monkeypatch.setattr("dogbench.sources._github_json", fake_github_json)
    result = attest_github_component(
        source_url="https://github.com/example/project/pull/7",
        repo_url="https://github.com/example/project",
        base_sha=base,
        head_sha=head,
        paths=["src/app.py"],
    )

    assert result["base_sha"] == base
    assert result["github_reported_base_sha"] == "3" * 40


def test_live_source_attestation_rejects_wrong_merge_base(monkeypatch) -> None:
    base = "1" * 40
    head = "2" * 40

    def fake_github_json(path: str):
        if path.endswith("/pulls/7"):
            return {"base": {"sha": base}, "head": {"sha": head}}
        if "/compare/" in path:
            return {"merge_base_commit": {"sha": "4" * 40}}
        raise AssertionError(path)

    monkeypatch.setattr("dogbench.sources._github_json", fake_github_json)
    with pytest.raises(SourceAttestationError, match="not the PR range merge base"):
        attest_github_component(
            source_url="https://github.com/example/project/pull/7",
            repo_url="https://github.com/example/project",
            base_sha=base,
            head_sha=head,
            paths=["src/app.py"],
        )


def test_audit_rejects_candidate_edits_to_code_context(tmp_path, capsys) -> None:
    source, base, head = source_repo(tmp_path)
    item = {
        "instance_id": "unsafe-path",
        "source_url": "https://example.invalid/pull/1",
        "context": "Clarify existing behavior for users.",
        "docs": {
            "repo_url": str(source),
            "base_sha": base,
            "head_sha": head,
            "paths": ["docs/guide.md"],
        },
        "code": None,
    }
    items = tmp_path / "items.json"
    items.write_text(json.dumps(item), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "instance_id": "unsafe-path",
                "decision": "patch",
                "patch": (
                    "diff --git a/_code_repo/app.py b/_code_repo/app.py\n"
                    "--- a/_code_repo/app.py\n"
                    "+++ b/_code_repo/app.py\n"
                    "+changed\n"
                ),
            }
        ],
    )

    assert (
        main(
            [
                "validate",
                str(predictions),
                "--items",
                str(items),
                "--offline",
                "--allow-missing-traces",
                "--json",
            ]
        )
        == 3
    )
    report = json.loads(capsys.readouterr().out)
    assert (
        report["contamination"]["reports"][0]["hard_findings"][0]["code"]
        == "unsafe_candidate_path"
    )


def test_context_audit_rejects_source_pr_solution_prose() -> None:
    item = {
        "instance_id": "source-prose",
        "source_url": "https://github.com/example/project/pull/7",
        "context": "Explain how operators can safely retry rollback after restoring the missing resource.",
        "docs": {
            "repo_url": "https://github.com/example/project",
            "base_sha": "1" * 40,
            "head_sha": "2" * 40,
            "paths": ["docs/guide.md"],
        },
        "code": None,
    }
    report = context_audit(
        item["context"],
        item,
        "",
        protected_source_prose=[item["context"]],
    )

    assert report["ok"] is False
    assert report["findings"][0]["code"] == "source_pr_prose_overlap"
