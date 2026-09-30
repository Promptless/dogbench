"""Execute frozen prepared inputs using the original research adapters and rules."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, MutableMapping
from urllib.parse import urlparse

from .adapters import AgentResult
from .prompts import cap_prompt_for_devin

MODEL_CREDENTIAL_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "COHERE_API_KEY",
    "FIREWORKS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "XAI_API_KEY",
}



MODEL_CREDENTIAL_ENV_PREFIXES = ("AWS_", "BEDROCK_")



@contextmanager
def restricted_third_party_adapter_environment(
    environ: MutableMapping[str, str] = os.environ,
):
    """Temporarily remove credentials that could call a model provider."""
    denied = {
        key: value
        for key, value in tuple(environ.items())
        if key in MODEL_CREDENTIAL_ENV_KEYS
        or key.startswith(MODEL_CREDENTIAL_ENV_PREFIXES)
    }
    for key in denied:
        environ.pop(key, None)
    try:
        yield sorted(denied)
    finally:
        for key in tuple(environ):
            if key in MODEL_CREDENTIAL_ENV_KEYS or key.startswith(
                MODEL_CREDENTIAL_ENV_PREFIXES
            ):
                environ.pop(key, None)
        environ.update(denied)



CLAUDE_CANDIDATE_KEY_RE = re.compile(
    r"^claude__(claude|bedrock)-(opus|sonnet)-(\d+)-(\d+)(?:$|[-_])"
)



def configure_claude_candidate_contract(
    agent: str,
    candidate_key: str,
    model: str | None,
    environ: MutableMapping[str, str] = os.environ,
) -> dict[str, str] | None:
    """Make Claude routing an explicit property of the candidate key."""
    match = CLAUDE_CANDIDATE_KEY_RE.match(candidate_key)
    if agent != "claude":
        if match:
            raise ValueError(
                f"candidate key {candidate_key!r} requires --agent claude"
            )
        return None
    if not match:
        return None

    route, family, major, minor = match.groups()
    expected_model_token = f"claude-{family}-{major}-{minor}"
    effective_model = model or "claude-sonnet-4-6"
    if expected_model_token not in effective_model:
        raise ValueError(
            f"candidate key {candidate_key!r} requires model containing "
            f"{expected_model_token!r}, got {effective_model!r}"
        )

    if route == "bedrock":
        environ["DOCBENCH_CLAUDE_CODE_USE_BEDROCK"] = "1"
    else:
        environ.pop("DOCBENCH_CLAUDE_CODE_USE_BEDROCK", None)
    return {
        "route": route,
        "expected_model_token": expected_model_token,
        "requested_model": effective_model,
    }



def validate_claude_result_contract(
    contract: dict[str, str] | None,
    reported_model: str | None,
) -> list[str]:
    if not contract:
        return []
    actual = reported_model or ""
    errors: list[str] = []
    if contract["expected_model_token"] not in actual:
        errors.append(
            "reported Claude model does not match candidate key: "
            f"expected {contract['expected_model_token']!r}, got {actual!r}"
        )
    actual_route = (
        "bedrock"
        if re.search(r"(?:^|\.)anthropic\.claude-", actual)
        else "claude"
    )
    if actual and actual_route != contract["route"]:
        errors.append(
            "reported Claude route does not match candidate key: "
            f"expected {contract['route']!r}, got {actual_route!r} ({actual!r})"
        )
    return errors



def source_pr_number_from_item_id(item_id: str) -> int | None:
    """Infer the source GitHub PR number from canonical item ids."""
    tail = (item_id or "").rsplit("-pr", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])



def create_brokered_agent_workspace(
    trusted_clone: Path,
    agent_workspace: Path,
    *,
    branch: str,
) -> str:
    """Create a shallow, single-branch, remote-less checkout for the agent."""
    subprocess.run(
        [
            "git", "clone", "--quiet", "--no-hardlinks", "--depth", "1",
            "--single-branch", "--branch", branch,
            trusted_clone.resolve().as_uri(), str(agent_workspace),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(agent_workspace), "remote", "remove", "origin"],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(agent_workspace), "config",
            "core.hooksPath", "/dev/null",
        ],
        check=True,
    )
    base_oid = subprocess.run(
        ["git", "-C", str(agent_workspace), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git", "-C", str(agent_workspace), "update-ref",
            "refs/docbench/agent-base", base_oid,
        ],
        check=True,
    )
    return base_oid



def export_brokered_agent_patch(
    agent_workspace: Path,
    *,
    base_oid: str,
) -> tuple[str, list[str]]:
    """Export all tracked and untracked agent edits without executing hooks."""
    subprocess.run(
        [
            "git", "-C", str(agent_workspace), "-c",
            "core.hooksPath=/dev/null", "add", "-A",
        ],
        check=True,
    )
    names = subprocess.run(
        [
            "git", "-C", str(agent_workspace), "diff", "--cached",
            "--name-only", "--diff-filter=ACDMRTUXB", base_oid,
        ],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    forbidden = sorted(path for path in names if path == "_code_repo" or path.startswith("_code_repo/"))
    if forbidden:
        raise RuntimeError(
            "agent modified read-only code context: " + ", ".join(forbidden[:20])
        )
    patch = subprocess.run(
        [
            "git", "-C", str(agent_workspace), "diff", "--cached",
            "--binary", "--full-index", base_oid,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return patch, names



def verdict_from_brokered_patch(patch: str) -> str:
    """Derive the local-agent verdict only from its sealed workspace diff."""
    return "PATCH_READY" if patch.strip() else "NO_DOC_CHANGES_NEEDED"



def recover_brokered_patch_result(result: AgentResult, patch: str) -> bool:
    """Promote a sealed local patch after a recoverable process-level failure.

    Agent prose and exit status are not the candidate artifact for local lanes;
    the isolated workspace diff is.  Preserve the original process outcome for
    audit while allowing the normal provenance, scope, and contamination gates
    below to decide whether the patch is usable.
    """
    if result.ok or not patch.strip():
        return False
    model_contract = (result.extras or {}).get("candidate_model_contract") or {}
    if model_contract and model_contract.get("ok") is not True:
        return False
    prior = {
        "ok": result.ok,
        "error": result.error,
        "verdict": result.verdict,
    }
    extras = dict(result.extras or {})
    extras["recovered_after_agent_termination"] = prior
    result.extras = extras
    result.ok = True
    result.error = None
    result.verdict = "PATCH_READY"
    result.docs_pr_url = None
    return True



class ExecutionConfigurationError(ValueError):
    """Required frozen inputs or explicit deployment configuration are missing."""


class RateLimitedRun(RuntimeError):
    """The research runner leaves rate-limited attempts without a candidate."""


@contextmanager
def _temporary_environment(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _repo_slug(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return ""
    return "/".join(parsed.path.strip("/").split("/")[:2]).removesuffix(".git")


def _required(config: dict[str, Any], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ExecutionConfigurationError(f"run configuration requires {name}")
    return value


def _credential(config: dict[str, Any], name: str) -> str:
    key = _required(config, name)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ExecutionConfigurationError(f"{name} must name an environment variable")
    value = os.environ.get(key, "")
    if not value:
        raise ExecutionConfigurationError(f"credential environment variable is unset: {key}")
    return value


def _gh_json(route: str, token: str) -> Any:
    result = subprocess.run(
        ["gh", "api", route], env={**os.environ, "GH_TOKEN": token},
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def _cloud_item_config(config: dict[str, Any], item_id: str) -> dict[str, Any]:
    item_configs = config.get("items")
    if not isinstance(item_configs, dict) or not isinstance(item_configs.get(item_id), dict):
        raise ExecutionConfigurationError(f"cloud run requires config.items[{item_id!r}]")
    return {**{k: v for k, v in config.items() if k != "items"}, **item_configs[item_id]}


def _validate_cloud_input(
    item: dict[str, Any], workspace: Path, config: dict[str, Any], token: str,
) -> dict[str, Any]:
    """Validate an explicitly provisioned private mirror; never create one here."""
    from .mirror import validate_prepared_mirror
    repo = _required(config, "mirror_repo")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ExecutionConfigurationError("mirror_repo must be owner/repository")
    upstream = {_repo_slug(item['docs']['repo_url'])}
    if item.get("code"):
        upstream.add(_repo_slug(item['code']['repo_url']))
    if repo in upstream:
        raise ExecutionConfigurationError("cloud mirror must differ from every upstream repository")
    metadata = _gh_json(f"repos/{repo}", token)
    if not metadata.get("private"):
        raise ExecutionConfigurationError("cloud run requires an explicitly prepared private mirror")
    branch = _required(config, "base_branch")
    base = _required(config, "base_sha")
    clone = Path(_required(config, "clone_dir")).expanduser().resolve()
    transport_mode = config.get("transport_mode", "exact")
    if transport_mode not in {"exact", "managed"}:
        raise ExecutionConfigurationError("transport_mode must be exact or managed")
    transport_options = {}
    if transport_mode == "managed":
        transport_options = {
            "managed_transport": True,
            "source_repo": _repo_slug(item['docs']['repo_url']),
            "code_source_repo": _repo_slug(item['code']['repo_url']) if item.get('code') else None,
            "mirror_overlays_root": Path(_required(config, 'mirror_overlays_root')).expanduser().resolve(),
            "source_overlays_root": Path(_required(config, 'source_overlays_root')).expanduser().resolve(),
        }
    validation = validate_prepared_mirror(
        repo_dir=clone, mirror_repo=repo, expected_base_sha=base,
        prepared_docs_dir=workspace / "docs", base_branch=branch,
        prepared_code_dir=workspace / "code" if item.get("code") else None,
        code_path_prefix=config.get("code_pr_path_prefix", ""),
        **transport_options,
    )
    validation["ok"] = True
    remote_base = _gh_json(f"repos/{repo}/commits/{base}", token)
    if remote_base["commit"]["tree"]["sha"] != validation["docs_tree"]:
        raise ExecutionConfigurationError("remote mirror base tree differs from prepared documentation")
    # Require the named branch still to resolve to the attested base.
    from urllib.parse import quote
    branch_info = _gh_json(f"repos/{repo}/branches/{quote(branch, safe='')}", token)
    if branch_info["commit"]["sha"] != base:
        raise ExecutionConfigurationError("remote mirror base branch moved")
    if item.get("code"):
        url = _required(config, "code_pr_url")
        match = re.fullmatch(rf"https://github\.com/{re.escape(repo)}/pull/(\d+)", url)
        if not match:
            raise ExecutionConfigurationError("code_pr_url must be a PR in the configured private mirror")
        pr = _gh_json(f"repos/{repo}/pulls/{match[1]}", token)
        if pr.get("state") != "open" or pr["base"]["sha"] != base:
            raise ExecutionConfigurationError("prepared code PR must be open against the frozen mirror base")
        prefix = config.get("code_pr_path_prefix", "")
        if prefix not in ("", "_code_repo/"):
            raise ExecutionConfigurationError("code_pr_path_prefix must be empty or _code_repo/")
        code = workspace / "code"
        names = _git(code, "diff", "--name-only", "--no-renames", "HEAD^..HEAD").splitlines()
        expected = (
            set(validation["code_paths"]) if transport_mode == "managed"
            else {prefix + name for name in names}
        )
        file_rows = []
        page = 1
        while True:
            rows = _gh_json(f"repos/{repo}/pulls/{match[1]}/files?per_page=100&page={page}", token)
            file_rows.extend(rows)
            if len(rows) < 100:
                break
            page += 1
        actual = {r["filename"] for r in file_rows}
        actual.update(r["previous_filename"] for r in file_rows if r.get("previous_filename"))
        if actual != expected:
            raise ExecutionConfigurationError("mirror code PR path set differs from the prepared code change")
        for side, rev in (("base", "HEAD^"), ("head", "HEAD")):
            tree = _gh_json(f"repos/{repo}/git/trees/{pr[side]['sha']}?recursive=1", token)
            if tree.get("truncated"):
                raise ExecutionConfigurationError("cannot verify a truncated remote code tree")
            remote = {r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
            if transport_mode == "managed":
                expected_blobs = validation[f"code_{side}_blob_oids"]
                if side == "head" and tree.get("sha") != validation["code_head_tree"]:
                    raise ExecutionConfigurationError("mirror code PR head tree differs from original managed transport")
                for path, expected_blob in expected_blobs.items():
                    if remote.get(path) != expected_blob:
                        raise ExecutionConfigurationError(f"mirror code PR {side} content differs at {path}")
            else:
                for name in names:
                    found = subprocess.run(
                        ["git", "-C", str(code), "rev-parse", "--verify", f"{rev}:{name}"],
                        capture_output=True, text=True, check=False,
                    )
                    expected_blob = found.stdout.strip() if found.returncode == 0 else None
                    if remote.get(prefix + name) != expected_blob:
                        raise ExecutionConfigurationError(f"mirror code PR {side} content differs at {name}")
        validation["code_pr_verified"] = True
    elif config.get("code_pr_url"):
        raise ExecutionConfigurationError("docs-only item cannot receive an extra code PR")
    return validation


def _cloud_prompt(item: dict[str, Any], config: dict[str, Any]) -> str:
    from .prompts import (
        DOCUMENTATION_TASK_INSTRUCTION, MANAGED_REQUIRED_SAFETY_SUFFIX,
        _additional_context_block, trailing_instruction, build_docs_only_prompt,
    )
    branch = _required(config, "base_branch")
    work_branch = _required(config, "docs_work_branch")
    if item.get("code"):
        code_note = (
            "\nThe changed source files are under the `_code_repo/` directory "
            "(reference only — do NOT edit it)."
            if config.get("code_pr_path_prefix") == "_code_repo/" else ""
        )
        prompt = (
            f"{DOCUMENTATION_TASK_INSTRUCTION}\n\n"
            f"Provided code change: {_required(config, 'code_pr_url')}.{code_note}"
            f"{_additional_context_block(item['context'])}\n"
            f"{trailing_instruction(branch, work_branch, managed_safety=True)}"
        )
    else:
        prompt = build_docs_only_prompt(
            item['context'], f"https://github.com/{config['mirror_repo']}",
            docs_base_branch=branch, managed_safety=True,
        )
    # Original main adds the required managed suffix even if the template has it.
    return f"{prompt.rstrip()}\n\n{MANAGED_REQUIRED_SAFETY_SUFFIX}\n"


def _run_cloud_adapter(
    agent: str, model: str, item: dict[str, Any], workspace: Path,
    log_dir: Path, config: dict[str, Any], timeout_s: int,
) -> tuple[AgentResult, str, str, dict[str, Any]]:
    from . import adapters
    token = _credential(config, "github_token_env")
    attestation = _validate_cloud_input(item, workspace, config, token)
    prompt = _cloud_prompt(item, config)
    if agent == "devin":
        # The published wrapper does not resume prior attempts implicitly.
        prompt = cap_prompt_for_devin(prompt, config['mirror_repo'], config['base_branch'])
    (log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    baseline = adapters.list_mirror_pr_numbers(config['mirror_repo'], token)
    kwargs = dict(
        prompt=prompt, mirror_repo=config['mirror_repo'], token=token,
        timeout_s=timeout_s, log_dir=log_dir, model=model,
        baseline_pr_numbers=baseline, base_branch=config['base_branch'],
    )
    if agent == "devin":
        kwargs['devin_api_key'] = _credential(config, 'devin_api_key_env')
        with restricted_third_party_adapter_environment():
            result = adapters.run_devin(**kwargs)
    elif agent == "mintlify":
        kwargs['mintlify_token'] = _credential(config, 'mintlify_token_env')
        kwargs['project_id'] = _required(config, 'project_id')
        # Original adapter assumes the caller has pointed the project at mirror_repo.
        if config.get('project_mirror_confirmed') is not True:
            raise ExecutionConfigurationError('Mintlify requires project_mirror_confirmed=true')
        with restricted_third_party_adapter_environment():
            result = adapters.run_mintlify(**kwargs)
    else:
        kwargs.update(
            api_trigger_key=_credential(config, 'promptless_api_trigger_key_env'),
            runtime_base_url=_required(config, 'runtime_base_url'),
            code_pr_url=config.get('code_pr_url'),
            expected_docs_branch=config['docs_work_branch'],
            expected_base_oid=config['base_sha'],
            mirror_path=Path(config['clone_dir']).expanduser().resolve(),
            analysis_version=config.get('analysis_version'),
        )
        result = adapters.run_promptless(**kwargs)
    return result, "", prompt, attestation


def _prediction(candidate: dict[str, Any]) -> dict[str, Any]:
    """Map preserved candidate outcomes to the existing public transport schema."""
    row = {
        'instance_id': candidate['item_id'],
        'agent': {'name': candidate['agent'], 'model': candidate['model']},
        'usage': {'elapsed_s': candidate['elapsed_s'], 'cost_usd': candidate['cost_usd']},
    }
    if not candidate['ok']:
        row['status'] = 'invalid' if not candidate['contamination_gate']['ok'] else 'error'
        row['error'] = candidate.get('error') or 'agent did not complete successfully'
    elif candidate['docs_diff'].strip():
        row.update(status='completed', decision='patch', patch=candidate['docs_diff'])
    elif candidate['verdict'] == 'NO_DOC_CHANGES_NEEDED':
        row.update(status='completed', decision='abstain')
    else:
        row.update(status='error', error=candidate.get('error') or candidate['verdict'] or 'unknown agent verdict')
    return row


def _run_one(
    item: dict[str, Any], workspace: Path, output: Path, *, agent: str,
    model: str, timeout_s: int, budget_usd: float, config: dict[str, Any],
    candidates_root: Path,
) -> dict[str, Any]:
    from . import adapters
    from .trace_audit import audit_post_dispatch_agent_trace
    from .runtime_contamination import build_contamination_report, model_input_artifacts, scan_model_inputs
    log_dir = output / 'logs'
    log_dir.mkdir(parents=True)
    candidate_key = str(config.get('candidate_key') or agent)
    source_repo = _repo_slug(item['source_url']) or _repo_slug(item['docs']['repo_url'])
    pr_number = source_pr_number_from_item_id(item['instance_id'])
    head_shas = [c.get('head_sha') or '' for c in (item['docs'], item.get('code')) if c]
    from .research_inputs import local_execution_prompt, input_binding
    prompt = (
        local_execution_prompt(item, workspace)
        if agent in {'claude', 'codex', 'opencode'} else ''
    )
    binding = input_binding(item)
    if binding is not None:
        _write_json(log_dir / 'research_input_binding.json', binding)
    if agent not in {'claude','codex','opencode'}:
        cloud_config = _cloud_item_config(config, item['instance_id'])
        prompt = _cloud_prompt(item, cloud_config)
    (log_dir / 'prompt.txt').write_text(prompt, encoding='utf-8')
    prompt_audit = scan_model_inputs(
        [log_dir / 'prompt.txt'], item_id=item['instance_id'], source_repo=source_repo,
        source_pr_number=pr_number, source_head_shas=head_shas,
    )
    prompt_audit['errors'] = [f"protected model input: {r['kind']}" for r in prompt_audit['findings']]
    _write_json(log_dir / 'prompt_contamination_audit.json', prompt_audit)
    if not prompt_audit['ok']:
        raise ExecutionConfigurationError('; '.join(prompt_audit['errors']))
    with tempfile.TemporaryDirectory(prefix='workspace-') as temporary:
        root = Path(temporary)
        if agent in {'claude','codex','opencode'}:
            clone = root / 'agent-workspace'
            branch = _git(workspace / 'docs', 'symbolic-ref', '--short', 'HEAD')
            base_oid = create_brokered_agent_workspace(workspace / 'docs', clone, branch=branch)
            task_dir = root / 'task-inputs'
            task_dir.mkdir()
            if item.get('code'):
                shutil.copytree(workspace / 'code', task_dir / 'code_repo', symlinks=True)
            (task_dir / 'README.txt').write_text(
                'This directory contains only sanitized, read-only task inputs.\n', encoding='utf-8',
            )
            environment = {
                'DOCBENCH_REQUIRE_AGENT_FS_ISOLATION':'1',
                'DOCBENCH_PATCH_BROKER_MODE':'1',
                'DOCBENCH_AGENT_TASK_DIR':str(task_dir),
                'DOCBENCH_CLAUDE_CODE_USE_BEDROCK':os.environ.get('DOCBENCH_CLAUDE_CODE_USE_BEDROCK',''),
            }
            with _temporary_environment(environment):
                contract = configure_claude_candidate_contract(agent, candidate_key, model)
                kwargs = dict(
                    clone_dir=clone, prompt=prompt, mirror_repo='local/sanitized-snapshot',
                    token='', timeout_s=timeout_s, log_dir=log_dir,
                    baseline_pr_numbers=set(), base_branch=branch, base_sha=base_oid, model=model,
                )
                if agent=='claude':
                    kwargs['budget_usd']=budget_usd
                result = getattr(adapters, f'run_{agent}')(**kwargs)
            if (result.error or '').startswith('CLAUDE_RATE_LIMIT:'):
                (log_dir/'agent_raw_output.txt').write_text(result.raw_output, encoding='utf-8')
                raise RateLimitedRun(result.error)
            if agent=='claude':
                reported_model = str((result.extras or {}).get('reported_model') or result.model or '')
                errors = validate_claude_result_contract(contract, reported_model)
                if contract:
                    result.extras = {**(result.extras or {}), 'candidate_model_contract':{
                        **contract, 'reported_model':reported_model, 'ok':not errors, 'errors':errors,
                    }}
                if errors:
                    result.ok=False
                    result.error='; '.join(filter(None,[result.error,*errors]))
            patch, changed = export_brokered_agent_patch(clone, base_oid=base_oid)
            recovered = recover_brokered_patch_result(result,patch)
            (log_dir/'brokered_agent.patch').write_text(patch,encoding='utf-8')
            attestation = {
                'schema_version':'docbench-local-agent-sandbox-v1','ok':True,
                'backend':os.environ.get('DOCBENCH_AGENT_SANDBOX_BACKEND'),
                'agent_workspace_remote_count':0,'task_inputs_read_only':True,
                'candidate_logs_mounted':False,'trusted_publisher_mounted':False,
                'output_contract':'sealed_workspace_diff','agent_final_text_used_for_verdict':False,
                'patch_paths':changed,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),
                'recovered_after_agent_termination':recovered,
            }
            result.extras={**(result.extras or {}),'sandbox_attestation':attestation}
            result.docs_pr_url=None
            if result.ok:
                result.verdict=verdict_from_brokered_patch(patch)
            _write_json(log_dir/'sandbox_attestation.json',attestation)
        else:
            result,patch,prompt,attestation = _run_cloud_adapter(
                agent,model,item,workspace,log_dir,cloud_config,timeout_s,
            )
            _write_json(log_dir/'third_party_mirror_attestation.json',attestation)
        audit = audit_post_dispatch_agent_trace(
            log_dir,source_repo=source_repo,source_pr_number=pr_number,agent=agent,
            protected_host_paths=tuple(config.get('protected_host_paths',[])),
        )
        _write_json(log_dir/'post_dispatch_source_access_audit.json',audit)
        result.extras={**(result.extras or {}),'post_dispatch_source_access_audit':audit}
        if not audit['ok']:
            result.ok=False
            result.error='; '.join(filter(None,[result.error,*audit['errors']]))
        if agent not in {'claude','codex','opencode'}:
            patch = _finalize_cloud_result(result, cloud_config, log_dir)
        report=build_contamination_report(
            item_id=item['instance_id'],agent=agent,candidate_key=candidate_key,
            candidate_diff=patch,candidates_root=candidates_root,
            model_input_paths=model_input_artifacts(log_dir),source_repo=source_repo,
            source_pr_number=pr_number,source_head_shas=head_shas,
            prompt_audit=prompt_audit,trace_audit=audit,
            managed_mirror_attestation=attestation if agent not in {'claude','codex','opencode'} else None,
        )
        _write_json(log_dir/'contamination_report.json',report)
        if not report['ok']:
            result.ok=False
            codes=[r.get('code') for r in report['hard_findings']+report['similarity_alerts']]
            result.error='; '.join(filter(None,[result.error,'contamination gate failed: '+', '.join(filter(None,codes))]))
        candidate = {
            'schema_version':2,'item_id':item['instance_id'],'source_repo':source_repo,
            'agent':result.agent,'model':result.model,'candidate_key':candidate_key,
            'source':candidate_key,'verdict':result.verdict,'docs_pr_url':result.docs_pr_url,
            'docs_diff':patch,'docs_diff_sha256':hashlib.sha256(patch.encode()).hexdigest(),
            'elapsed_s':result.elapsed_s,'cost_usd':result.cost_usd,'ok':result.ok,
            'error':result.error,'input_contamination_audit':prompt_audit,
            'post_dispatch_source_access_audit':audit,'contamination_gate':report,
            'extras':result.extras,
        }
        _write_json(output/'candidate.json',candidate)
        (log_dir/'agent_raw_output.txt').write_text(result.raw_output,encoding='utf-8')
        return candidate


def run_prepared(
    items_path: Path, workspaces_path: Path, output_dir: Path, *, agent: str,
    model: str, timeout_s: int = 1800, budget_usd: float = 5.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run each frozen prepared item once; emit the public prediction transport.

    Local lanes require Linux and the configured original container isolation.
    Cloud lanes require per-item, pre-provisioned private mirrors in ``config``.
    Reference candidates, when supplied, stay on the controller and are never
    mounted into a local agent. No context or rubric is generated here.
    """
    from .items import load_items
    from .verify import verify_output
    from .predictions import validate_predictions
    config = dict(config or {})
    if agent not in {'claude','codex','opencode','devin','mintlify','promptless'}:
        raise ExecutionConfigurationError(f'unsupported agent: {agent}')
    if not isinstance(model,str) or not model.strip():
        raise ExecutionConfigurationError('an explicit model is required')
    if timeout_s <= 0 or budget_usd <= 0:
        raise ExecutionConfigurationError('timeout and budget must be positive')
    if agent in {'claude','codex','opencode'} and not sys.platform.startswith('linux'):
        raise ExecutionConfigurationError('local agent execution requires Linux container isolation')
    prepared=Path(workspaces_path).resolve()
    output=Path(output_dir).resolve()
    if output.exists():
        raise ExecutionConfigurationError('output directory already exists; use a fresh run directory')
    if output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ExecutionConfigurationError('output must be outside prepared workspaces')
    items=load_items(Path(items_path))
    verification=verify_output(prepared)
    index=json.loads((prepared/'index.json').read_text())
    rows={r['instance_id']:r for r in index['items']}
    if set(rows)!={i['instance_id'] for i in items}:
        raise ExecutionConfigurationError('prepared workspace IDs differ from requested item IDs')
    bindings=[]
    for item in items:
        row=rows[item['instance_id']]
        attestation_path=Path(row['attestation'])
        if not attestation_path.is_absolute():
            attestation_path=prepared/attestation_path
        attestation=json.loads(attestation_path.read_text())
        if attestation['item']!=item:
            raise ExecutionConfigurationError(f"prepared item bytes differ: {item['instance_id']}")
        workspace=Path(row['path'])
        if not workspace.is_absolute():
            workspace=prepared/workspace
        bindings.append((item,workspace))
        if agent not in {'claude','codex','opencode'}:
            cloud_config=_cloud_item_config(config,item['instance_id'])
            for field in ('mirror_repo','clone_dir','base_branch','base_sha','docs_work_branch','github_token_env'):
                _required(cloud_config,field)
    candidates_root=Path(config.get('candidate_reference_root') or output/'reference-candidates').resolve()
    if candidates_root.is_relative_to(prepared):
        raise ExecutionConfigurationError('candidate references must stay outside prepared workspaces')
    output.mkdir(parents=True)
    predictions=[]
    stopped=None
    exit_code=0
    contamination_count=0
    for item,workspace in bindings:
        try:
            candidate=_run_one(
                item,workspace,output/'items'/item['instance_id'],agent=agent,model=model,
                timeout_s=timeout_s,budget_usd=budget_usd,config=config,candidates_root=candidates_root,
            )
        except RateLimitedRun as exc:
            stopped=str(exc)
            exit_code=max(exit_code,2)
            break
        # Validation receives primary agent traces only, never controller records.
        from .trace_audit import PRIMARY_AGENT_TRACE_NAMES
        trace_output = output / 'traces' / item['instance_id']
        trace_output.mkdir(parents=True)
        for name in sorted(PRIMARY_AGENT_TRACE_NAMES):
            primary = output / 'items' / item['instance_id'] / 'logs' / name
            if primary.is_file():
                shutil.copy2(primary, trace_output / name)
        predictions.append(_prediction(candidate))
        if not candidate['contamination_gate']['ok']:
            contamination_count += 1
            exit_code = 3
        elif candidate['verdict'] == 'PATCH_BROKER_PENDING':
            exit_code = max(exit_code, 2)
        elif not candidate['ok'] or predictions[-1]['status'] != 'completed':
            exit_code = max(exit_code, 1)
        if candidate['contamination_gate']['decision']=='quarantine_and_stop':
            stopped='contamination gate requested a batch stop'
            break
    if predictions:
        validate_predictions(predictions,item_ids=set(rows),require_complete=False)
    prediction_path=output/'predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in predictions),encoding='utf-8')
    summary={
        'agent':agent,'model':model,'attempts_per_item':1,'items_requested':len(items),
        'items_completed':len(predictions),'failed':sum(r['status']!='completed' for r in predictions),
        'complete':len(predictions)==len(items),'stopped':stopped,
        'exit_code':exit_code,'contamination_count':contamination_count,
        'predictions':str(prediction_path),'prepared_verified':verification['verified'],
        'candidate_reference_root':str(candidates_root),
    }
    _write_json(output/'run.json',summary)
    return summary


def fetch_pr_diff(pr_url: str, token: str) -> str:
    """Get the raw unified diff for a docs PR on the mirror.

    Earlier versions filtered this down to files that looked like docs or
    docs-adjacent paths. That hid useful signal when an agent edited generated
    files, examples, configs, or other non-doc paths. For candidate outputs, keep
    the full PR diff and let later scoring decide whether those changes are
    appropriate.
    """

    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "diff", pr_url],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode == 0:
        raw = res.stdout
    else:
        raw = fetch_pr_diff_via_files_api(pr_url, token)
        if not raw:
            print(
                f"[run] warning: gh pr diff failed for docs PR "
                f"({res.stderr.strip()[:200]})"
            )
            return ""
    return raw


def fetch_pr_diff_via_files_api(pr_url: str, token: str) -> str:
    """Fetch a PR diff via the files API when GitHub refuses /pull.diff.

    GitHub's PR diff endpoint returns HTTP 406 once a PR exceeds 300 files. The
    files API still returns per-file patches, which is enough for the docs-diff
    filter and evaluator artifacts.
    """
    PR_URL_RE = re.compile(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)")

    m = PR_URL_RE.search(pr_url)
    if not m:
        return ""
    repo, num = m.group(1), m.group(2)
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        [
            "gh", "api", "--paginate", f"repos/{repo}/pulls/{num}/files",
            "--jq", ".[] | {filename,status,previous_filename,patch} | @json",
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0:
        return ""

    chunks: list[str] = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, str):
                item = json.loads(item)
        except json.JSONDecodeError:
            continue
        filename = item.get("filename")
        if not filename:
            continue
        patch = item.get("patch") or ""
        if not patch.strip():
            continue
        status = item.get("status")
        previous = item.get("previous_filename") or filename
        old_path = "/dev/null" if status == "added" else f"a/{previous}"
        new_path = "/dev/null" if status == "removed" else f"b/{filename}"
        chunks.append(
            f"diff --git a/{previous} b/{filename}\n"
            f"--- {old_path}\n"
            f"+++ {new_path}\n"
            f"{patch.rstrip()}\n"
        )
    return "".join(chunks)



def _finalize_cloud_result(result: AgentResult, config: dict[str, Any], log_dir: Path) -> str:
    """Original SHA-repair-diff sequence, restricted to the configured private mirror."""
    from .cloud_results import (
        validate_pr_sha_context, retarget_pr_to_expected_base_if_safe,
        republish_pr_after_protected_base_write_if_safe,
    )
    if not result.docs_pr_url:
        if result.verdict.startswith('UNKNOWN:') and not result.error:
            result.ok=False
            result.error=result.verdict
        return ''
    repo=config['mirror_repo']
    if not re.fullmatch(rf'https://github\.com/{re.escape(repo)}/pull/\d+',result.docs_pr_url):
        result.ok=False
        result.error='agent output PR is outside the configured private mirror'
        return ''
    token=_credential(config,'github_token_env')
    kwargs=dict(
        pr_url=result.docs_pr_url,token=token,expected_mirror_repo=repo,
        expected_base_ref=config['base_branch'],expected_base_oid=config['base_sha'],
    )
    validation=validate_pr_sha_context(**kwargs)
    if not validation['ok'] and retarget_pr_to_expected_base_if_safe(
        pr_url=result.docs_pr_url,token=token,expected_base_ref=config['base_branch'],
        expected_base_oid=config['base_sha'],validation=validation,
    ):
        validation=validate_pr_sha_context(**kwargs)
        validation['warnings'].append({
            'code':'pr_retargeted_to_expected_base',
            'message':"Agent opened the PR against the mirror default branch; the runner retargeted it after verifying its first commit was based on the exact pinned benchmark SHA.",
        })
    if not validation['ok'] and result.agent=='mintlify':
        replacement=republish_pr_after_protected_base_write_if_safe(
            pr_url=result.docs_pr_url,token=token,expected_mirror_repo=repo,
            expected_base_ref=config['base_branch'],expected_base_oid=config['base_sha'],
            expected_head_ref=config['docs_work_branch'],validation=validation,
        )
        if replacement:
            if not re.fullmatch(rf'https://github\.com/{re.escape(repo)}/pull/\d+',replacement):
                result.ok=False
                result.error='replacement PR is outside the configured private mirror'
                return ''
            result.docs_pr_url=replacement
            kwargs['pr_url']=replacement
            validation=validate_pr_sha_context(**kwargs)
            validation['warnings'].append({
                'code':'protected_base_write_republished',
                'message':"Mintlify wrote its unchanged commit onto the pinned base ref; the controller copied that commit to the work ref, restored the frozen base OID, and opened a replacement PR without altering agent content.",
            })
    result.extras={**(result.extras or {}),'sha_validation':validation}
    _write_json(log_dir/'sha_validation.json',validation)
    if not validation['ok']:
        result.ok=False
        message='docs PR SHA validation failed: '+'; '.join(e['code'] for e in validation['errors'])
        result.error=f'{result.error}; {message}' if result.error else message
    patch=fetch_pr_diff(result.docs_pr_url,token)
    if not patch.strip() and not result.error:
        result.ok=False
        result.error=f'docs PR produced no reviewable docs diff: {result.docs_pr_url}'
    return patch
