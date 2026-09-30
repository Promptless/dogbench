"""Adapters that run an agent against a doc-bench mirror clone and return a
uniform `AgentResult`.

All adapters share the same contract:
  - Caller has already set the mirror's `main` to the base SHA and (for
    code-containing tasks) opened a code-only PR on the mirror.
  - Adapter receives a fresh clone of the mirror, a task prompt, and metadata.
  - Adapter runs the agent headless and produces an `AgentResult` describing
    what happened, including the verdict (`DOCS_PR_URL: ...`,
    `NO_DOC_CHANGES_NEEDED`, or `UNKNOWN: ...`).

Adapters do NOT extract the diff themselves. Verdict + raw output are returned
and the driver resolves the diff from the mirror PR.
"""
from __future__ import annotations

import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codex_auth_lock import serialize_codex_auth
from .promptless_backend import PromptlessBackend, load_promptless_backend


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    agent: str                           # "claude" | "codex" | "devin"
    model: str
    ok: bool                             # process exit succeeded
    verdict: str                         # "DOCS_PR_URL: <url>" | "NO_DOC_CHANGES_NEEDED" | "UNKNOWN: ..."
    docs_pr_url: str | None              # parsed from verdict, when applicable
    elapsed_s: float
    raw_output: str = ""
    cost_usd: float | None = None
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _copy_agent_credential(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_opencode_candidate_policy(safe_home: Path) -> Path:
    """Write the final OpenCode config layer used by benchmark candidates."""
    path = safe_home / ".config/opencode/docbench-policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provider": {
                    "openrouter": {
                        "models": {
                            "moonshotai/kimi-k3": {},
                            "qwen/qwen3.8-max": {},
                        }
                    }
                },
                "permission": {
                    "external_directory": {
                        "/task/code_repo/**": "allow",
                    },
                    "edit": {
                        "/task/code_repo/**": "deny",
                    },
                    "webfetch": "deny",
                    "websearch": "deny",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _persist_refreshed_codex_auth(isolated_home: Path | None) -> bool:
    """Atomically retain a valid Codex token refresh from an isolated run.

    Candidate homes must remain disposable so agents cannot inspect one
    another's sessions or logs.  Codex ChatGPT auth is the exception: refresh
    tokens are rotating, so discarding a refreshed ``auth.json`` makes the
    host copy unusable for the next candidate.  Persist only that single file,
    after checking its schema and account identity against the host copy.
    """
    if isolated_home is None:
        return False
    source = isolated_home / ".codex/auth.json"
    configured_auth = os.environ.get("DOCBENCH_CODEX_AUTH_FILE", "").strip()
    if not configured_auth:
        return False
    destination = Path(configured_auth).expanduser().resolve()
    try:
        raw = source.read_bytes()
        if not raw or len(raw) > 1_000_000:
            return False
        candidate = json.loads(raw)
        if not isinstance(candidate, dict):
            return False
        tokens = candidate.get("tokens")
        if not isinstance(tokens, dict):
            return False
        required_tokens = ("access_token", "account_id", "id_token", "refresh_token")
        if any(not isinstance(tokens.get(key), str) or not tokens[key] for key in required_tokens):
            return False

        if destination.is_file():
            current = json.loads(destination.read_text(encoding="utf-8"))
            current_tokens = current.get("tokens") if isinstance(current, dict) else None
            if isinstance(current_tokens, dict):
                current_account = current_tokens.get("account_id")
                if current_account and current_account != tokens["account_id"]:
                    return False
            current_mode = current.get("auth_mode") if isinstance(current, dict) else None
            if current_mode and current_mode != candidate.get("auth_mode"):
                return False

        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".auth.json.docbench-", dir=str(destination.parent)
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, destination)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _broker_aws_role_credentials(env: dict[str, str]) -> dict[str, str]:
    """Resolve the host role to short-lived values without exposing IMDS."""
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        return env
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError("Bedrock sandbox requires AWS CLI credential export")
    result = subprocess.run(
        [aws, "configure", "export-credentials", "--format", "process"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not broker short-lived AWS credentials for Bedrock: "
            + result.stderr.strip()[:200]
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AWS credential export returned invalid JSON") from exc
    credential_env = env.copy()
    mapping = {
        "AccessKeyId": "AWS_ACCESS_KEY_ID",
        "SecretAccessKey": "AWS_SECRET_ACCESS_KEY",
        "SessionToken": "AWS_SESSION_TOKEN",
        "Expiration": "AWS_CREDENTIAL_EXPIRATION",
    }
    for source, destination in mapping.items():
        value = payload.get(source)
        if value:
            credential_env[destination] = str(value)
    if not credential_env.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError("AWS credential export omitted AccessKeyId")
    return credential_env


def isolate_local_agent_process(
    cmd: list[str],
    env: dict[str, str],
    clone_dir: Path,
) -> tuple[list[str], dict[str, str], Path | None]:
    """Run a local agent with only its checkout, clean home, and task inputs.

    The production backend is an unprivileged Docker container on an
    internal network.  Its only egress path is the model-provider proxy.
    Execution requires Linux and an explicitly configured Docker or Podman sandbox.
    """
    required = os.environ.get("DOCBENCH_REQUIRE_AGENT_FS_ISOLATION", "1") == "1"
    if not required:
        raise RuntimeError("public candidate execution requires filesystem isolation")
    if sys.platform != "linux":
        raise RuntimeError("agent filesystem isolation requires Linux")

    backend = os.environ.get("DOCBENCH_AGENT_SANDBOX_BACKEND", "docker")
    if backend in {"docker", "podman"}:
        runtime = shutil.which(backend)
        sudo = shutil.which("sudo")
        if not runtime or not sudo:
            raise RuntimeError(f"agent sandbox requires {backend} and sudo")

        if env.get("CLAUDE_CODE_USE_BEDROCK") == "1":
            env = _broker_aws_role_credentials(env)

        isolation_root_override = env.get("DOCBENCH_AGENT_ISOLATION_ROOT_OVERRIDE", "").strip()
        isolation_root = (
            Path(isolation_root_override).resolve()
            if isolation_root_override
            else clone_dir.parent / ".agent-isolation"
        )
        safe_home = isolation_root / "home"
        safe_home.mkdir(parents=True, exist_ok=True)
        credential_paths = {
            "DOCBENCH_CLAUDE_AUTH_FILE": ".claude/.credentials.json",
            "DOCBENCH_CODEX_AUTH_FILE": ".codex/auth.json",
            "DOCBENCH_OPENCODE_AUTH_FILE": ".local/share/opencode/auth.json",
            "DOCBENCH_OPENCODE_CONFIG_FILE": ".config/opencode/opencode.jsonc",
        }
        for variable, relative_path in credential_paths.items():
            configured = os.environ.get(variable, "").strip()
            if configured:
                source = Path(configured).expanduser().resolve()
                if not source.is_file():
                    raise RuntimeError(f"configured credential/config file does not exist: {variable}")
                _copy_agent_credential(source, safe_home / relative_path)
        _write_opencode_candidate_policy(safe_home)
        (safe_home / ".gitconfig").write_text(
            "[user]\n\tname = docbench-agent\n\temail = agent@localhost\n"
            "[core]\n\thooksPath = /dev/null\n",
            encoding="utf-8",
        )

        image_key = (
            "DOCBENCH_QWEN_SANDBOX_IMAGE"
            if any("qwen/qwen3.8-max" in value.lower() for value in cmd)
            else "DOCBENCH_AGENT_SANDBOX_IMAGE"
        )
        required_settings = (image_key, "DOCBENCH_AGENT_SANDBOX_NETWORK", "DOCBENCH_AGENT_EGRESS_PROXY")
        missing = [key for key in required_settings if not os.environ.get(key, "").strip()]
        if missing:
            raise RuntimeError("agent sandbox configuration missing: " + ", ".join(missing))
        image = os.environ[image_key]
        network = os.environ["DOCBENCH_AGENT_SANDBOX_NETWORK"]
        proxy = os.environ["DOCBENCH_AGENT_EGRESS_PROXY"]
        task_dir_text = os.environ.get("DOCBENCH_AGENT_TASK_DIR")
        task_dir = Path(task_dir_text).resolve() if task_dir_text else None
        if task_dir is not None and not task_dir.is_dir():
            raise RuntimeError(f"agent task directory does not exist: {task_dir}")
        output_dir_text = env.get("DOCBENCH_AGENT_OUTPUT_DIR")
        output_dir = Path(output_dir_text).resolve() if output_dir_text else None
        if output_dir is not None and not output_dir.is_dir():
            raise RuntimeError(f"agent output directory does not exist: {output_dir}")

        clone_root = clone_dir.resolve()

        def _container_arg(value: str) -> str:
            # Relative arguments already resolve against the container's
            # /workspace working directory.  Treating every string as a host
            # path corrupts ordinary flags and shell programs (for example,
            # ``bash -lc 'gh api ...'`` became
            # ``bash /workspace/-lc /workspace/gh api ...``).
            if not Path(value).is_absolute():
                return value
            try:
                path = Path(value).resolve()
                relative = path.relative_to(clone_root)
            except (OSError, ValueError):
                return value
            return str(Path("/workspace") / relative)

        container_cmd = [_container_arg(value) for value in cmd]
        # Host and image package managers install the same CLI at different
        # absolute prefixes. Resolve the executable from the pinned image PATH.
        container_cmd[0] = Path(cmd[0]).name
        cidfile_override = os.environ.get(
            "DOCBENCH_AGENT_CONTAINER_CIDFILE_OVERRIDE", "",
        ).strip()
        cidfile = (
            Path(cidfile_override).resolve()
            if cidfile_override
            else isolation_root / "container.cid"
        )
        cidfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            cidfile.unlink()
        except FileNotFoundError:
            pass
        wrapped = [
            "run", "--rm",
        ]
        # Claude receives its prompt over stdin. Rubric Codex calls use the
        # explicit `codex exec -` stdin form because their prompts can exceed
        # argv limits. Candidate Codex and OpenCode receive positional prompts;
        # keeping stdin open for those can make them wait forever for EOF.
        executable = Path(cmd[0]).name
        if executable == "claude" or (executable == "codex" and cmd[-1:] == ["-"]):
            wrapped.append("--interactive")
        wrapped += [
            "--cidfile", str(cidfile),
            "--network", network,
            "--read-only",
            "--cap-drop", "all",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "1024",
            "--user", "1000:1000",
            "--workdir", "/workspace",
            "--volume", f"{clone_root}:/workspace:rw,Z",
            "--volume", f"{safe_home.resolve()}:/home/agent:rw,Z",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=1073741824",
        ]
        if task_dir is not None:
            wrapped += ["--volume", f"{task_dir}:/task:ro,Z"]
        if output_dir is not None:
            # Rubric audit agents may write their required artifacts here. It
            # is deliberately the only writable location besides checkout.
            wrapped += ["--volume", f"{output_dir}:/outputs:rw,Z"]

        container_env = {
            "HOME": "/home/agent",
            "CODEX_HOME": "/home/agent/.codex",
            "XDG_CONFIG_HOME": "/home/agent/.config",
            "XDG_DATA_HOME": "/home/agent/.local/share",
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "TMP": "/tmp",
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
            "OPENCODE_CONFIG": "/home/agent/.config/opencode/docbench-policy.json",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_DISABLE_SHARE": "true",
            "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER": "true",
            "OPENCODE_PURE": "true",
            "DISABLE_AUTOUPDATER": "1",
            "DO_NOT_TRACK": "1",
        }
        if env.get("DOCBENCH_RUBRIC_RUNTIME_POLICY") != "rubric-research":
            container_env.update(
                {
                    "HTTP_PROXY": proxy,
                    "HTTPS_PROXY": proxy,
                    "ALL_PROXY": proxy,
                    "NO_PROXY": "localhost,127.0.0.1",
                }
            )
        # Bedrock routes need short-lived AWS credentials. Rubric research is
        # the sole policy allowed to receive a dedicated GitHub API token;
        # candidate/doc-patch agents remain sealed from GitHub credentials.
        for key, value in env.items():
            if key.startswith("AWS_") or key in {
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX",
                "OPENROUTER_API_KEY",
            } or (
                env.get("DOCBENCH_RUBRIC_RUNTIME_POLICY") == "rubric-research"
                and key in {"GH_TOKEN", "GITHUB_TOKEN"}
            ):
                container_env[key] = value
        env_file = isolation_root / "container.env"
        for key, value in container_env.items():
            if "\n" in value or "\r" in value:
                raise RuntimeError(f"container environment value contains a newline: {key}")
        env_file.write_text(
            "".join(f"{key}={value}\n" for key, value in sorted(container_env.items())),
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        wrapped += ["--env-file", str(env_file)]
        wrapped += [image, *container_cmd]

        # Keep credential values out of both argv and the launcher process
        # environment. The short-lived protected env file is deleted whenever
        # the runtime wrapper exits.
        launcher = isolation_root / "sandbox-container-launcher.sh"
        quoted_env_file = shlex.quote(str(env_file))
        quoted_cidfile = shlex.quote(str(cidfile))
        quoted_sudo = shlex.quote(sudo)
        quoted_runtime = shlex.quote(runtime)
        launcher.write_text(
            "#!/bin/sh\n"
            "cleanup() {\n"
            f"  if [ -s {quoted_cidfile} ]; then\n"
            f"    {quoted_sudo} -n {quoted_runtime} rm -f \"$(cat {quoted_cidfile})\" "
            ">/dev/null 2>&1 || true\n"
            "  fi\n"
            f"  rm -f {quoted_cidfile} {quoted_env_file}\n"
            "}\n"
            "trap cleanup EXIT\n"
            "trap 'exit 129' HUP\n"
            "trap 'exit 130' INT\n"
            "trap 'exit 143' TERM\n"
            f"{shlex.quote(sudo)} -n {shlex.quote(runtime)} \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o700)

        launcher_env = env.copy()
        for key in tuple(launcher_env):
            if (
                key.startswith("AWS_")
                or key in {
                    "GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS", "SSH_AUTH_SOCK",
                    "GIT_SSH", "GIT_SSH_COMMAND", "OPENROUTER_API_KEY",
                }
            ):
                launcher_env.pop(key, None)
        # subprocess.run() kills only the launcher on timeout. Docker otherwise
        # leaves the detached container alive, so retain just enough host-side
        # metadata for the adapter to force-remove that exact container.
        launcher_env["DOCBENCH_AGENT_CONTAINER_CIDFILE"] = str(cidfile)
        launcher_env["DOCBENCH_AGENT_CONTAINER_RUNTIME"] = runtime
        launcher_env["DOCBENCH_AGENT_CONTAINER_SUDO"] = sudo
        return [str(launcher), *wrapped], launcher_env, safe_home

    raise RuntimeError(f"unsupported agent sandbox backend: {backend}; use docker or podman")


# ─────────────────────────────────────────────────────────────────────────────
# Shared verdict extraction
# ─────────────────────────────────────────────────────────────────────────────

# Must require a concrete GitHub PR URL so we don't match placeholders like
# `<url>` or hallucinated paths such as `/pull/NEW_PR`.
_DOCS_PR_URL_RE = re.compile(
    r"\bDOCS_PR_URL:\s*"
    r"(https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/(\d+))\b"
)
_NO_CHANGES_RE = re.compile(r"\bNO_DOC_CHANGES_NEEDED\b")
# Agents that correctly resist a no-op often end with a prose "no docs needed"
# conclusion instead of the literal marker (and make no PR). This is ONLY
# consulted when no new PR was produced (see extract_verdict_from_repo), so a
# match there means a genuine no-op decision, not docs that also discuss other
# files — making these patterns safe from false positives.
_NO_DOC_PROSE_RE = re.compile(
    r"no (?:additional |separate |further |new )*"
    r"(?:doc|documentation)\w* (?:change|update|edit|PR)\w* "
    r"(?:are |is )?(?:needed|required|necessary|warranted)"
    r"|no (?:additional |separate |further |new )?"
    r"docs? (?:are |is )?(?:needed|required|necessary|warranted)"
    r"|no (?:additional |separate |further |new )?"
    r"(?:doc(?:umentation)?|docs?|doc-only) PR (?:is|was) "
    r"(?:needed|required|necessary|warranted)"
    r"|no (?:additional |separate |further |new )?"
    r"docs? PR is (?:needed|required|necessary|warranted)"
    r"|no (?:additional |separate |further |new )?"
    r"PR (?:is|was) (?:needed|required|necessary|warranted)"
    r"|(?:doc|documentation)\w* (?:is|are) already (?:up[- ]to[- ]date|accurate|correct)"
    r"|does not (?:require|warrant|need) (?:a |any )?(?:doc|documentation)"
    r"|PR already contains (?:all )?(?:the )?"
    r"(?:doc|documentation)\w* (?:change|update|edit)\w* (?:required|needed)"
    r"|(?:PR(?: #\d+)?|candidate PR|existing PR) (?:already )?(?:contains|includes|updates|covers) "
    r"(?:the )?(?:relevant |needed |required )?(?:doc|documentation|docs?)"
    r"|(?:doc|documentation|docs?) (?:already )?(?:appear to be |are )?"
    r"(?:updated|covered|present) in (?:the )?(?:PR|candidate PR|existing PR) (?:itself)?"
    r"|I (?:did not|didn[’']t) open (?:a |an |the )?"
    r"(?:separate |duplicate |new |additional )?(?:doc|documentation|docs?) PR "
    r"because (?:I )?(?:didn[’']t|did not|don[’']t|do not|couldn[’']t|could not|found no|find no)"
    r"|I (?:did not|didn[’']t) open (?:a |an |the )?"
    r"(?:separate |duplicate |new |additional )?(?:doc|documentation|docs?) PR"
    r"|I (?:did not|didn[’']t) open (?:a |an |the )?duplicate PR"
    r"|I (?:do not|don[’']t) think (?:a |an |the )?"
    r"(?:separate |duplicate |new |additional )?(?:doc|documentation|docs?) PR "
    r"(?:is|was)? ?(?:needed|required|necessary|warranted)"
    r"|no (?:separate |new |additional )?(?:docs?-only |documentation )?gap "
    r"(?:that )?(?:needs|requires|warrants) (?:a )?(?:follow[- ]?up|PR|change|update)"
    r"|documentation assessment[:\s*]+\**\s*no\b",
    re.I,
)


def _parse_claude_stream_result(stdout: str) -> tuple[str, float | None]:
    """Return the final assistant/result text from Claude Code stream-json."""
    result_text = ""
    assistant_text = ""
    cost_usd: float | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "assistant":
            msg = obj.get("message")
            if isinstance(msg, dict):
                text_parts = [
                    part.get("text") or ""
                    for part in (msg.get("content") or [])
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text = "\n".join(part for part in text_parts if part)
                if text:
                    assistant_text = text
        elif obj.get("type") == "result":
            text = obj.get("result") or obj.get("content") or ""
            if text:
                result_text = text
            cost_usd = obj.get("total_cost_usd") or obj.get("cost_usd") or cost_usd
    return result_text or assistant_text or stdout, cost_usd


def _parse_claude_reported_model(stdout: str) -> str | None:
    """Read the provider model Claude Code reports in its init event."""
    for line in (stdout or "").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(obj, dict)
            and obj.get("type") == "system"
            and obj.get("subtype") == "init"
            and isinstance(obj.get("model"), str)
        ):
            return obj["model"]
    return None


def _parse_claude_rate_limit(stdout: str) -> str | None:
    """Return a control-plane rate-limit message from Claude Code stream-json."""
    message = ""
    reset_at: int | None = None
    saw_rate_limit = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "rate_limit_event":
            info = obj.get("rate_limit_info") or {}
            if info.get("status") == "rejected":
                saw_rate_limit = True
                try:
                    reset_at = int(info.get("resetsAt"))
                except (TypeError, ValueError):
                    reset_at = None
        if obj.get("type") == "result" and obj.get("is_error") and obj.get("api_error_status") == 429:
            saw_rate_limit = True
            if obj.get("result"):
                message = str(obj["result"])
    if not saw_rate_limit:
        return None
    reset_suffix = ""
    if reset_at:
        reset_iso = datetime.fromtimestamp(reset_at, timezone.utc).isoformat()
        reset_suffix = f"; resets_at={reset_iso}"
    return f"CLAUDE_RATE_LIMIT: {message or 'Claude Code session limit'}{reset_suffix}"


EXCLUDED_BRANCH_PREFIXES: tuple[str, ...] = (
    "candidate/code",     # our own code-only PR
    "dependabot/",        # GitHub auto-bot, even with .github stripped these may slip in
    "renovate/",          # other dep-bot
)


def _secret_redactions() -> list[tuple[str, str]]:
    redactions: list[tuple[str, str]] = []
    sensitive = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "COOKIE")
    for key, value in os.environ.items():
        if not value or len(value) < 12:
            continue
        if any(part in key.upper() for part in sensitive):
            redactions.append((value, f"***REDACTED:{key}***"))
    # Common token shapes that can appear in native CLI logs even when the
    # exact value is not present in this process environment.
    redactions.extend([
        (r"gh[pousr]_[A-Za-z0-9_]{20,}", "***REDACTED:GITHUB_TOKEN***"),
        (r"sk-[A-Za-z0-9_-]{20,}", "***REDACTED:API_KEY***"),
    ])
    return redactions


def _redact_trace_text(text: str) -> str:
    for needle, replacement in _secret_redactions():
        if not needle:
            continue
        if needle.startswith(("gh", "sk-")) and "[" in needle:
            text = re.sub(needle, replacement, text)
        else:
            text = text.replace(needle, replacement)
    return text


def _native_trace_patterns(agent: str, home_dir: Path | None = None) -> list[str]:
    home = str(home_dir or Path.home())
    if agent == "claude":
        return [
            f"{home}/.claude/projects/**/*.jsonl",
        ]
    if agent == "codex":
        return [
            f"{home}/.codex/session_index.jsonl",
            f"{home}/.codex/sessions/**/*.jsonl",
            f"{home}/.codex/archived_sessions/**/*.jsonl",
            f"{home}/.codex/log/**/*.log",
        ]
    if agent == "opencode":
        return [
            f"{home}/.local/share/opencode/log/**/*.log",
        ]
    return []


def snapshot_native_agent_traces(
    agent: str,
    log_dir: Path | None,
    *,
    started_at: float,
    max_files: int = 60,
    max_bytes_per_file: int = 5_000_000,
    session_ids: set[str] | None = None,
    home_dir: Path | None = None,
) -> list[str]:
    """Copy native CLI session/log artifacts modified during this run.

    Candidate logs already capture the harness stdout/stderr. This supplements
    them with the agent tool's own session store, which is often where the
    richest trajectory lives. The copy is best-effort and redacts obvious secret
    values before writing into the candidate log directory.
    """
    if not log_dir:
        return []
    patterns = _native_trace_patterns(agent, home_dir)
    if not patterns:
        return []

    dest_root = log_dir / "native_sessions"
    copied: list[str] = []
    cutoff = started_at - 10.0
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for path in Path("/").glob(pattern[1:]) if pattern.startswith("/") else Path().glob(pattern):
            try:
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                if resolved.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for src in candidates[:max_files]:
        rel_label = "__".join(part for part in src.parts if part not in ("/", ""))
        dest = dest_root / rel_label
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = src.read_bytes()
        except OSError:
            continue
        truncated = False
        if len(raw) > max_bytes_per_file:
            raw = raw[-max_bytes_per_file:]
            truncated = True
        text = raw.decode("utf-8", errors="replace")
        if agent == "opencode":
            # OpenCode appends concurrent sessions to one global log. Keep
            # only lines tied to this invocation; opencode.events.jsonl holds
            # the complete per-run trajectory.
            if not session_ids:
                continue
            text = "\n".join(
                line for line in text.splitlines()
                if any(session_id in line for session_id in session_ids)
            )
            if not text:
                continue
            text += "\n"
        if truncated:
            text = f"[trace truncated to last {max_bytes_per_file} bytes]\n{text}"
        dest.write_text(_redact_trace_text(text), encoding="utf-8")
        copied.append(str(dest.relative_to(log_dir)))

    if copied:
        (log_dir / "native_sessions_manifest.json").write_text(
            json.dumps({
                "agent": agent,
                "started_at": started_at,
                "copied": copied,
                "redacted": True,
                "max_files": max_files,
                "max_bytes_per_file": max_bytes_per_file,
            }, indent=2),
            encoding="utf-8",
        )
    return copied


def _text_from_jsonl_events(raw: str) -> str:
    """Best-effort final assistant text extraction from JSONL event streams."""
    chunks: list[str] = []
    final_text = ""
    for line in (raw or "").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        event_type = str(obj.get("type") or obj.get("event") or "").lower()
        for key in ("result", "text", "content", "message", "output"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                if event_type in {"result", "final", "complete", "completed", "assistant"}:
                    final_text = val
                chunks.append(val)
                break
            if isinstance(val, dict):
                nested = val.get("content") or val.get("text") or val.get("message")
                if isinstance(nested, str) and nested.strip():
                    if event_type in {"result", "final", "complete", "completed", "assistant"}:
                        final_text = nested
                    chunks.append(nested)
                    break
        part = obj.get("part")
        if isinstance(part, dict):
            nested = part.get("text") or part.get("content") or part.get("message")
            if isinstance(nested, str) and nested.strip():
                if event_type in {"text", "message", "assistant", "result", "final"}:
                    final_text = nested
                chunks.append(nested)
    return final_text or "\n".join(chunks)


def _opencode_tool_event(
    obj: dict[str, Any],
) -> tuple[str | None, str | None, bool, bool]:
    """Extract tool details, including OpenCode lifecycle interruptions."""
    if obj.get("type") == "tool":
        state = obj.get("state") if isinstance(obj.get("state"), dict) else {}
        input_data = state.get("input") if isinstance(state.get("input"), dict) else {}
        metadata = (
            state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        )
        return (
            str(obj.get("tool") or "") or None,
            str(input_data.get("url") or "") or None,
            str(state.get("status") or "").lower() == "error",
            metadata.get("interrupted") is True,
        )

    # Some OpenCode versions nest tool events under a message/part payload.
    for key in ("part", "data", "message"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            tool, url, failed, interrupted = _opencode_tool_event(nested)
            if tool or url:
                return tool, url, failed, interrupted
    return None, None, False, False


def _opencode_terminal_error(stdout: str) -> str | None:
    """Extract a terminal provider/process error from OpenCode JSONL."""
    terminal: str | None = None
    for line in (stdout or "").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "error":
            continue
        error = obj.get("error") if isinstance(obj.get("error"), dict) else {}
        data = error.get("data") if isinstance(error.get("data"), dict) else {}
        message = data.get("message") or error.get("message") or obj.get("message")
        if isinstance(message, str) and message.strip():
            terminal = message.strip()
    return terminal


def _opencode_loop_guard_reason(
    line: str,
    *,
    invalid_count: int,
    failed_webfetch_by_url: dict[str, int],
    accumulated_cost_usd: float,
    max_invalid_tool_calls: int,
    max_cost_usd: float,
) -> tuple[str | None, int, dict[str, int], float]:
    """Return a kill reason when OpenCode is clearly stuck in a tool loop."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None, invalid_count, failed_webfetch_by_url, accumulated_cost_usd
    if not isinstance(obj, dict):
        return None, invalid_count, failed_webfetch_by_url, accumulated_cost_usd

    part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
    event_cost = part.get("cost")
    if isinstance(event_cost, (int, float)) and event_cost > 0:
        accumulated_cost_usd += float(event_cost)
        if accumulated_cost_usd >= max_cost_usd:
            return (
                f"opencode cost guard: ${accumulated_cost_usd:.4f} >= ${max_cost_usd:.2f}",
                invalid_count,
                failed_webfetch_by_url,
                accumulated_cost_usd,
            )

    tool, url, failed, interrupted = _opencode_tool_event(obj)
    # Qwen/OpenCode reports malformed calls as either `invalid` or `unknown`,
    # depending on the OpenCode build.  Count both so a bad-call loop cannot
    # consume an entire lane budget. OpenCode also emits synthetic `unknown`
    # tool events when it interrupts sibling calls at a step boundary; those
    # lifecycle placeholders are not model-issued malformed calls.
    if tool in {"invalid", "unknown"} and not interrupted:
        invalid_count += 1
        if invalid_count >= max_invalid_tool_calls:
            return (
                f"opencode loop guard: {invalid_count} invalid tool calls",
                invalid_count,
                failed_webfetch_by_url,
                accumulated_cost_usd,
            )

    if tool == "webfetch" and failed and url:
        failed_webfetch_by_url[url] = failed_webfetch_by_url.get(url, 0) + 1
        if failed_webfetch_by_url[url] >= 8:
            return (
                f"opencode loop guard: {failed_webfetch_by_url[url]} failed webfetch calls to {url}",
                invalid_count,
                failed_webfetch_by_url,
                accumulated_cost_usd,
            )

    return None, invalid_count, failed_webfetch_by_url, accumulated_cost_usd


def _opencode_step_metrics(line: str) -> tuple[bool, int | None]:
    """Return whether an event finished a model step and its reported token load."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obj, dict) or obj.get("type") != "step_finish":
        return False, None
    part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
    tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
    total = tokens.get("total")
    return True, int(total) if isinstance(total, (int, float)) and total >= 0 else None


def _run_opencode_with_loop_guard(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout_s: int,
    cwd: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    """Run OpenCode while terminating pathological tool loops early."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        bufsize=1,
        start_new_session=True,
    )
    stdout_parts: list[str] = []
    invalid_count = 0
    failed_webfetch_by_url: dict[str, int] = {}
    accumulated_cost_usd = 0.0
    completed_steps = 0
    max_invalid_tool_calls = max(
        1, int(env.get("DOCBENCH_OPENCODE_MAX_INVALID_TOOL_CALLS", "25")),
    )
    # Qwen 3.8 Max can enter a repeat-invalid-tool loop through OpenCode. Keep
    # the tighter circuit breaker model-specific so other OpenCode lanes retain
    # their configured tolerance.
    if any("qwen/qwen3.8-max" in arg.lower() for arg in cmd):
        max_invalid_tool_calls = min(max_invalid_tool_calls, 3)
    max_cost_usd = max(
        0.01, float(env.get("DOCBENCH_OPENCODE_MAX_COST_USD", "5.0")),
    )
    max_steps = max(1, int(env.get("DOCBENCH_OPENCODE_MAX_STEPS", "80")))
    max_step_tokens = max(
        1, int(env.get("DOCBENCH_OPENCODE_MAX_STEP_TOKENS", "250000")),
    )
    guard_reason: str | None = None
    deadline = time.monotonic() + timeout_s

    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    while True:
        if time.monotonic() > deadline:
            _terminate_agent_process_group(proc)
            # The Docker client can exit while the detached container still
            # owns its stdout pipe. Remove the exact CID before communicate(),
            # otherwise this timeout path can hang until an outer watchdog
            # kills the entire candidate job.
            _cleanup_agent_container(env)
            stdout, _ = proc.communicate()
            stdout_parts.append(stdout or "")
            raise subprocess.TimeoutExpired(cmd, timeout_s, output="".join(stdout_parts), stderr="")

        ready = sel.select(timeout=0.5)
        line = proc.stdout.readline() if ready else ""
        if line:
            stdout_parts.append(line)
            step_finished, step_tokens = _opencode_step_metrics(line)
            if step_finished:
                completed_steps += 1
                if completed_steps >= max_steps:
                    guard_reason = f"opencode step guard: {completed_steps} completed steps"
                elif step_tokens is not None and step_tokens >= max_step_tokens:
                    guard_reason = (
                        f"opencode token guard: {step_tokens} tokens in one step "
                        f">= {max_step_tokens}"
                    )
            if guard_reason:
                _terminate_agent_process_group(proc)
                _cleanup_agent_container(env)
                stdout, _ = proc.communicate()
                stdout_parts.append(stdout or "")
                return (
                    subprocess.CompletedProcess(cmd, proc.returncode if proc.returncode is not None else -15, "".join(stdout_parts), ""),
                    guard_reason,
                )
            guard_reason, invalid_count, failed_webfetch_by_url, accumulated_cost_usd = _opencode_loop_guard_reason(
                line,
                invalid_count=invalid_count,
                failed_webfetch_by_url=failed_webfetch_by_url,
                accumulated_cost_usd=accumulated_cost_usd,
                max_invalid_tool_calls=max_invalid_tool_calls,
                max_cost_usd=max_cost_usd,
            )
            if guard_reason:
                _terminate_agent_process_group(proc)
                _cleanup_agent_container(env)
                stdout, _ = proc.communicate()
                stdout_parts.append(stdout or "")
                return (
                    subprocess.CompletedProcess(cmd, proc.returncode if proc.returncode is not None else -15, "".join(stdout_parts), ""),
                    guard_reason,
                )
            continue

        if proc.poll() is not None:
            _terminate_agent_process_group(proc, grace_s=1.0)
            stdout, _ = proc.communicate()
            stdout_parts.append(stdout or "")
            return subprocess.CompletedProcess(cmd, proc.returncode, "".join(stdout_parts), ""), None

        time.sleep(0.1)


def _terminate_agent_process_group(
    proc: subprocess.Popen[str], *, grace_s: float = 10.0,
) -> None:
    """Terminate an isolated CLI wrapper and every descendant holding its pipes."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if proc.poll() is None:
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    # The wrapper may have exited while a container/docker descendant still
    # owns stdout. Kill the group even after the direct child was reaped.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait()


def list_mirror_pr_numbers(
    mirror_repo: str,
    token: str,
    *,
    strict: bool = False,
    state: str = "all",
) -> list[int]:
    """Return PR numbers on the mirror in the requested GitHub state.

    The default ``all`` state records a pre-run baseline for verdict filtering.
    Persistent Promptless mirrors also use ``open`` to attest that cleanup left
    only the current run's approved input PRs visible as active work.
    """
    if state not in {"all", "open", "closed", "merged"}:
        raise ValueError(f"unsupported PR state: {state}")
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "list", "--repo", mirror_repo, "--state", state,
         "--json", "number", "--limit", "200"],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0:
        if strict:
            raise RuntimeError(
                "could not verify mirror PR history: " + (res.stderr or "gh pr list failed")[:300]
            )
        return []
    if not res.stdout.strip():
        if strict:
            raise RuntimeError("could not verify mirror PR history: empty gh response")
        return []
    try:
        return [int(p["number"]) for p in json.loads(res.stdout)]
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        if strict:
            raise RuntimeError("could not parse mirror PR history") from exc
        return []


def _verified_agent_pr_url(
    mirror_repo: str,
    token: str,
    agent_text: str,
    baseline_pr_numbers: set[int],
) -> str | None:
    """Return a DOCS_PR_URL from agent text only if it is real and new.

    The preferred path is still `gh pr list` on the mirror. This fallback only
    exists for cases where listing failed or GitHub's search/indexing lags, so
    it must be stricter than a plain regex match.
    """
    m = _DOCS_PR_URL_RE.search(agent_text or "")
    if not m:
        return None
    url, owner, repo, number_s = m.groups()
    if f"{owner}/{repo}".lower() != mirror_repo.lower():
        return None
    try:
        number = int(number_s)
    except ValueError:
        return None
    if number in baseline_pr_numbers:
        return None

    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "view", url, "--json", "number,url"],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    if int(data.get("number", -1)) != number:
        return None
    return data.get("url") or url


def extract_verdict_from_repo(
    mirror_repo: str,
    token: str,
    *,
    baseline_pr_numbers: set[int] | None = None,
    exclude_branch_prefixes: tuple[str, ...] = EXCLUDED_BRANCH_PREFIXES,
    agent_text: str = "",
) -> tuple[str, str | None]:
    """Determine the verdict by listing PRs on the mirror.

    A PR is treated as the agent's docs PR only if:
      - its number is NOT in `baseline_pr_numbers` (it appeared during this run), AND
      - its head branch does NOT start with any of `exclude_branch_prefixes`.

    Among the remaining PRs, the most recently created is selected. Falls
    back to scanning `agent_text` for the marker lines.

    Returns (verdict_line, docs_pr_url_or_None).
    """
    baseline = baseline_pr_numbers or set()
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "list", "--repo", mirror_repo, "--state", "all",
         "--json", "number,headRefName,url,createdAt", "--limit", "100"],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
        try:
            prs = json.loads(res.stdout)
        except json.JSONDecodeError:
            prs = []
        candidates = []
        for p in prs:
            if int(p.get("number", -1)) in baseline:
                continue
            head = p.get("headRefName") or ""
            if any(head.startswith(prefix) for prefix in exclude_branch_prefixes):
                continue
            candidates.append(p)
        candidates.sort(key=lambda p: p.get("createdAt") or "", reverse=True)
        if candidates:
            url = candidates[0]["url"]
            return f"DOCS_PR_URL: {url}", url

    pr_url = _verified_agent_pr_url(mirror_repo, token, agent_text, baseline)
    if pr_url:
        return f"DOCS_PR_URL: {pr_url}", pr_url
    if _NO_CHANGES_RE.search(agent_text or "") or _NO_DOC_PROSE_RE.search(agent_text or ""):
        return "NO_DOC_CHANGES_NEEDED", None
    return "UNKNOWN: agent did not emit a verdict and no new PR found", None


def extract_brokered_verdict(agent_text: str) -> tuple[str, None]:
    """Parse a local-agent conclusion without consulting GitHub.

    The trusted parent inspects and publishes workspace changes after the
    sandbox exits.  Until then, any non-no-op response is deliberately only a
    pending patch verdict.
    """
    if _NO_CHANGES_RE.search(agent_text or "") or _NO_DOC_PROSE_RE.search(agent_text or ""):
        return "NO_DOC_CHANGES_NEEDED", None
    return "PATCH_BROKER_PENDING", None


# Branch-name prefix the fallback opens PRs from. Must NOT match
# EXCLUDED_BRANCH_PREFIXES, so extract_verdict_from_repo picks it up as the
# agent's docs PR. A per-agent suffix keeps concurrent/sequential runs on the
# same shared mirror from colliding on one branch (which would make a later
# agent's `gh pr create` fail and resolve to an earlier agent's PR).
_FALLBACK_BRANCH_PREFIX = "agent-docs"


def _should_recover_pr(verdict: str, pr_url: str | None) -> bool:
    return pr_url is None and verdict != "NO_DOC_CHANGES_NEEDED"


def ensure_pr_from_clone(
    clone_dir: Path,
    mirror_repo: str,
    token: str,
    *,
    branch_suffix: str = "recovered",
    base_branch: str = "main",
    base_sha: str | None = None,
    log_dir: Path | None = None,
) -> str | None:
    """Open a docs PR from work a local CLI agent left in `clone_dir` but never
    pushed. Local agents (codex, claude) reliably edit + sometimes commit the
    docs, then skip `git push` / `gh pr create` — leaving the harness with no
    PR to extract a diff from (verdict UNKNOWN) even though real work exists.

    We're not judging whether the agent remembered the mechanical push step, so
    this completes it on the agent's behalf: commit any uncommitted changes,
    push a branch, and open a PR against main. The diff is exactly the agent's
    doc edits relative to the base SHA the clone was made from — comparable to
    every other agent's PR diff.

    Returns the new PR URL, or None when the clone has no changes vs origin/main
    (a genuine no-op) or the push/PR fails.
    """
    git = ["git", "-C", str(clone_dir)]
    branch = f"{_FALLBACK_BRANCH_PREFIX}/{branch_suffix}"
    env = os.environ.copy()
    env["GH_TOKEN"] = token

    def _run(cmd: list[str], **kw):
        return subprocess.run(cmd, capture_output=True, text=True, env=env,
                              check=False, **kw)

    # 1. Stage + commit anything uncommitted.
    status = _run(git + ["status", "--porcelain"])
    if status.stdout.strip():
        _run(git + ["add", "-A"])
        _run(git + ["commit", "-m", "docs: changes recovered from agent run"])

    # 2. Determine whether HEAD is ahead of the clone's BASE (i.e. real changes
    #    exist to PR). Compare against the base SHA the clone started from, NOT
    #    origin/main: an agent that ran `git push origin main` (claude does this)
    #    advances origin/main to include its own commit, so origin/main..HEAD is
    #    0 and the work looks like a no-op. base_sha..HEAD still sees it.
    compare_ref = f"origin/{base_branch}"
    if base_sha:
        base_exists = _run(git + ["cat-file", "-e", f"{base_sha}^{{commit}}"])
        if base_exists.returncode == 0:
            compare_ref = base_sha
    ahead = _run(git + ["rev-list", "--count", f"{compare_ref}..HEAD"])
    try:
        n_ahead = int((ahead.stdout or "0").strip() or "0")
    except ValueError:
        n_ahead = 0
    if n_ahead == 0:
        return None

    # 2.5 If the agent pushed straight to main (origin/main moved past base),
    #     reset origin/main back to base BEFORE opening the PR — otherwise the
    #     PR's head branch and base both contain the commit and the diff is
    #     empty. Also keeps the mirror clean for other agents/items.
    if base_sha:
        cur_main = _run(git + ["rev-parse", f"origin/{base_branch}"]).stdout.strip()
        if cur_main and cur_main != base_sha:
            reset = _run(git + ["push", "-f", "origin", f"{base_sha}:{base_branch}"])
            if log_dir:
                (log_dir / "fallback_pr.log").write_text(
                    f"agent pushed to main ({cur_main[:12]}); reset to base "
                    f"{base_sha[:12]} rc={reset.returncode}\n{reset.stderr[:300]}",
                    encoding="utf-8")

    # 3. Push HEAD to the fallback branch and open the PR.
    push = _run(git + ["push", "-f", "origin", f"HEAD:{branch}"])
    if push.returncode != 0:
        if log_dir:
            (log_dir / "fallback_pr.log").write_text(
                f"push failed:\n{push.stderr}", encoding="utf-8")
        return None
    pr = _run([
        "gh", "pr", "create", "--repo", mirror_repo,
        "--base", base_branch, "--head", branch,
        "--title", "docs: update documentation",
        "--body", "Documentation changes.",
    ])
    if pr.returncode != 0:
        # A PR for this branch may already exist (re-run) — fetch its URL.
        existing = _run([
            "gh", "pr", "list", "--repo", mirror_repo, "--head", branch,
            "--state", "open", "--json", "url", "--limit", "1",
        ])
        if existing.returncode == 0 and existing.stdout.strip():
            try:
                arr = json.loads(existing.stdout)
                if arr:
                    return arr[0]["url"]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        if log_dir:
            (log_dir / "fallback_pr.log").write_text(
                f"pr create failed:\n{pr.stderr}", encoding="utf-8")
        return None
    return (pr.stdout or "").strip() or None


# ─────────────────────────────────────────────────────────────────────────────
# Claude Code adapter
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup_agent_container(env: dict[str, str]) -> None:
    """Force-remove the one sandbox container recorded for this invocation."""
    cidfile_text = env.get("DOCBENCH_AGENT_CONTAINER_CIDFILE", "")
    if not cidfile_text:
        return
    cidfile = Path(cidfile_text)
    try:
        cid = cidfile.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    if re.fullmatch(r"[0-9a-fA-F]{12,64}", cid):
        runtime = env.get("DOCBENCH_AGENT_CONTAINER_RUNTIME") or "docker"
        sudo = env.get("DOCBENCH_AGENT_CONTAINER_SUDO") or "sudo"
        try:
            subprocess.run(
                [sudo, "-n", runtime, "rm", "-f", cid],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pass
    try:
        cidfile.unlink()
    except FileNotFoundError:
        pass


def _run_claude_with_inactivity_guard(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    prompt: str,
    timeout_s: int,
    inactivity_timeout_s: float,
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    """Stream Claude JSONL and stop a silent CLI/tool hang early."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    proc.stdin.write(prompt)
    proc.stdin.close()
    proc.stdin = None
    output: list[str] = []
    errors: list[str] = []
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, output)
    selector.register(proc.stderr, selectors.EVENT_READ, errors)
    started = time.monotonic()
    last_activity = started

    def finish_after_termination() -> tuple[str, str]:
        _terminate_agent_process_group(proc)
        _cleanup_agent_container(env)
        stdout, stderr = proc.communicate()
        output.append(stdout or "")
        errors.append(stderr or "")
        return "".join(output), "".join(errors)

    while True:
        now = time.monotonic()
        if now - started > timeout_s:
            stdout, stderr = finish_after_termination()
            raise subprocess.TimeoutExpired(
                cmd, timeout_s, output=stdout, stderr=stderr,
            )
        if now - last_activity > inactivity_timeout_s:
            stdout, stderr = finish_after_termination()
            reason = (
                "claude inactivity guard: no stream/tool event for "
                f"{inactivity_timeout_s:g}s"
            )
            return subprocess.CompletedProcess(
                cmd, proc.returncode if proc.returncode is not None else -15,
                stdout, stderr,
            ), reason

        for key, _ in selector.select(timeout=0.5):
            line = key.fileobj.readline()
            if line:
                key.data.append(line)
                last_activity = time.monotonic()
            else:
                try:
                    selector.unregister(key.fileobj)
                except KeyError:
                    pass

        if proc.poll() is not None and not selector.get_map():
            return subprocess.CompletedProcess(
                cmd, proc.returncode, "".join(output), "".join(errors),
            ), None


def _timeout_text(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired captures without losing partial output."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _persist_timeout_trace(
    log_dir: Path | None,
    exc: subprocess.TimeoutExpired,
    *,
    stdout_names: tuple[str, ...],
    stderr_name: str,
) -> tuple[str, str]:
    """Persist incomplete primary traces before returning a timeout result."""
    stdout = _timeout_text(exc.stdout or exc.output)
    stderr = _timeout_text(exc.stderr)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        for name in stdout_names:
            (log_dir / name).write_text(stdout, encoding="utf-8")
        if stderr:
            (log_dir / stderr_name).write_text(stderr, encoding="utf-8")
    return stdout, stderr


def run_claude(
    *,
    clone_dir: Path,
    prompt: str,
    mirror_repo: str,
    token: str,
    model: str = "claude-sonnet-4-6",
    budget_usd: float = 5.0,
    timeout_s: int = 1800,
    log_dir: Path | None = None,
    baseline_pr_numbers: set[int] | None = None,
    base_branch: str = "main",
    base_sha: str | None = None,
) -> AgentResult:
    """Invoke `claude --print` headless against `clone_dir`."""
    bin_path = shutil.which("claude")
    if not bin_path:
        return AgentResult(
            agent="claude", model=model, ok=False,
            verdict="UNKNOWN: claude not on PATH",
            docs_pr_url=None, elapsed_s=0.0,
            error="claude binary not found",
        )

    claude_tools = os.environ.get(
        "DOCBENCH_CLAUDE_CODE_TOOLS",
        # Edit requires files to be read through Read first. The operational
        # prompt below still excludes persisted tool-result files.
        "Bash,Read,Edit,Write,Glob,Grep",
    ).strip()
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("DOCBENCH_"):
            env.pop(key, None)
    # Candidate runs should use the logged-in Claude Code account (for example
    # a Team plan) rather than silently switching to metered Anthropic API
    # billing because load_env populated ANTHROPIC_API_KEY for judge/rubric code.
    for key in tuple(env):
        if key.startswith("ANTHROPIC_"):
            env.pop(key, None)
    use_bedrock = os.environ.get("DOCBENCH_CLAUDE_CODE_USE_BEDROCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env.pop("CLAUDE_CODE_USE_VERTEX", None)
    else:
        env.pop("CLAUDE_CODE_USE_BEDROCK", None)
        env.pop("CLAUDE_CODE_USE_VERTEX", None)
    brokered = os.environ.get("DOCBENCH_PATCH_BROKER_MODE") == "1"
    if not brokered:
        env["GH_TOKEN"] = token

    # stream-json emits every tool use, tool result, and message chunk as a
    # separate JSON line on stdout. We keep this for the session log so
    # post-run audits can inspect EXACTLY which Bash/WebFetch/gh commands
    # the agent ran. The final line has type:"result" — parsed for the
    # AgentResult.
    #
    # Do not pass --bare here: Claude Code documents that --bare disables OAuth
    # and keychain reads, which forces API-key billing. --safe-mode keeps normal
    # account auth while avoiding repo-local customizations.
    if os.environ.get("DOCBENCH_REQUIRE_AGENT_FS_ISOLATION") == "1":
        # The Linux isolation wrapper maps the process to UID 0 inside a user
        # namespace. Claude Code rejects bypassPermissions for any apparent
        # root process, so grant only the built-in tools needed by the agent.
        permission_args = [
            "--permission-mode", "dontAsk",
            "--allowedTools", claude_tools,
            "--tools", claude_tools,
        ]
    else:
        permission_args = ["--dangerously-skip-permissions"]

    cmd = [
        bin_path, "--print",
        "--model", model,
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--max-budget-usd", str(budget_usd),
        *permission_args,
        "--safe-mode",
    ]
    try:
        cmd, env, isolated_home = isolate_local_agent_process(cmd, env, clone_dir)
    except RuntimeError as exc:
        return AgentResult(
            agent="claude", model=model, ok=False,
            verdict=f"UNKNOWN: {exc}", docs_pr_url=None, elapsed_s=0.0,
            error=str(exc),
        )

    bounded_output_instruction = (
        "\n\nOperational constraint: keep shell output bounded. Do not run an "
        "unscoped `git show` or dump a large diff/file. Use `git show --stat`, "
        "path-scoped diffs, and bounded `sed` ranges instead. Do not open Claude "
        "Code persisted tool-result files under ~/.claude/projects.\n\n"
    )
    prompt = bounded_output_instruction + prompt
    t0 = time.time()
    inactivity_reason: str | None = None
    try:
        if os.environ.get("DOCBENCH_REQUIRE_AGENT_FS_ISOLATION") == "1":
            inactivity_timeout_s = max(
                30.0,
                float(os.environ.get("DOCBENCH_CLAUDE_INACTIVITY_TIMEOUT_S", "300")),
            )
            proc, inactivity_reason = _run_claude_with_inactivity_guard(
                cmd,
                env=env,
                cwd=clone_dir,
                prompt=prompt,
                timeout_s=timeout_s,
                inactivity_timeout_s=inactivity_timeout_s,
            )
        else:
            proc = subprocess.run(
                cmd, cwd=str(clone_dir), input=prompt, capture_output=True, text=True,
                env=env, timeout=timeout_s, check=False,
            )
    except subprocess.TimeoutExpired as exc:
        _cleanup_agent_container(env)
        elapsed = time.time() - t0
        partial_output, _ = _persist_timeout_trace(
            log_dir,
            exc,
            stdout_names=("claude.stream.jsonl",),
            stderr_name="claude.stderr.txt",
        )
        native_traces = snapshot_native_agent_traces(
            "claude", log_dir, started_at=t0, home_dir=isolated_home,
        )
        partial_text, partial_cost = _parse_claude_stream_result(partial_output)
        return AgentResult(
            agent="claude", model=model, ok=False,
            verdict="UNKNOWN: timed out",
            docs_pr_url=None, elapsed_s=round(elapsed, 1),
            raw_output=partial_text[:20_000],
            cost_usd=partial_cost,
            error=f"timeout after {timeout_s}s",
            extras={
                "native_traces": native_traces,
                "primary_trace_partial": bool(partial_output),
            },
        )
    _cleanup_agent_container(env)
    elapsed = time.time() - t0

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Persist the FULL stream-json transcript for post-run leak audits —
        # every Bash, WebFetch, Read, gh command the agent ran is in here.
        (log_dir / "claude.stream.jsonl").write_text(proc.stdout or "", encoding="utf-8")
        if proc.stderr:
            (log_dir / "claude.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    native_traces = snapshot_native_agent_traces(
        "claude", log_dir, started_at=t0, home_dir=isolated_home,
    )

    if inactivity_reason:
        result_text, cost_usd = _parse_claude_stream_result(proc.stdout or "")
        return AgentResult(
            agent="claude", model=model, ok=False,
            verdict="UNKNOWN: claude tool/CLI inactivity guard",
            docs_pr_url=None, elapsed_s=round(elapsed, 1),
            raw_output=result_text[:20_000], cost_usd=cost_usd,
            error=inactivity_reason,
            extras={
                "native_traces": native_traces,
                "inactivity_guard": True,
            },
        )

    # Claude Code stream-json can contain intermediate result events from task
    # notifications; use the final result/assistant text, not the first one.
    result_text, cost_usd = _parse_claude_stream_result(proc.stdout or "")
    reported_model = _parse_claude_reported_model(proc.stdout or "")
    rate_limit_error = _parse_claude_rate_limit(proc.stdout or "")
    if rate_limit_error:
        return AgentResult(
            agent="claude", model=model, ok=False,
            verdict="UNKNOWN: claude rate limited",
            docs_pr_url=None, elapsed_s=round(elapsed, 1),
            raw_output=result_text[:20_000],
            cost_usd=cost_usd,
            error=rate_limit_error,
            extras={
                "native_traces": native_traces,
                "rate_limited": True,
                "reported_model": reported_model,
            },
        )

    if brokered:
        verdict, pr_url = extract_brokered_verdict(result_text)
    else:
        verdict, pr_url = extract_verdict_from_repo(
            mirror_repo, token,
            baseline_pr_numbers=baseline_pr_numbers,
            agent_text=result_text,
        )
    if not brokered and _should_recover_pr(verdict, pr_url):
        # Claude commonly edits + commits the docs but skips the push/PR step.
        # Complete it on its behalf so the harness has a diff to extract.
        fb = ensure_pr_from_clone(clone_dir, mirror_repo, token,
                                  branch_suffix="claude", base_branch=base_branch,
                                  base_sha=base_sha, log_dir=log_dir)
        if fb:
            verdict, pr_url = f"DOCS_PR_URL: {fb}", fb
    return AgentResult(
        agent="claude", model=model, ok=(proc.returncode == 0),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        raw_output=result_text[:20_000],
        cost_usd=cost_usd,
        error=(proc.stderr[-500:] if proc.returncode != 0 else None),
        extras={"native_traces": native_traces, "reported_model": reported_model},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Codex CLI adapter
# ─────────────────────────────────────────────────────────────────────────────


@serialize_codex_auth
def run_codex(
    *,
    clone_dir: Path,
    prompt: str,
    mirror_repo: str,
    token: str,
    model: str | None = None,
    timeout_s: int = 1800,
    log_dir: Path | None = None,
    baseline_pr_numbers: set[int] | None = None,
    base_branch: str = "main",
    base_sha: str | None = None,
) -> AgentResult:
    """Invoke OpenAI's `codex exec` headless against `clone_dir`.

    Auth: uses the explicit `DOCBENCH_CODEX_AUTH_FILE` credential path. An explicit model may be used with either ChatGPT or
    API-key auth when that account exposes the requested model. For pinned
    runs, verify the native session's `turn_context.model` after launch.

    Uses `--dangerously-bypass-approvals-and-sandbox` so the agent can run gh
    commands (push, pr create) without interactive prompts. Captures the
    final agent message via `--output-last-message`.
    """
    bin_path = shutil.which("codex")
    if not bin_path:
        return AgentResult(
            agent="codex", model=model or "(default)", ok=False,
            verdict="UNKNOWN: codex not on PATH",
            docs_pr_url=None, elapsed_s=0.0,
            error="codex binary not found",
        )

    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("DOCBENCH_") or key.startswith("ANTHROPIC_"):
            env.pop(key, None)
    brokered = os.environ.get("DOCBENCH_PATCH_BROKER_MODE") == "1"
    if not brokered:
        env["GH_TOKEN"] = token

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    # The benchmark checkout/log directory is hidden inside the agent mount
    # namespace. Keep Codex's transient final-message file in the visible clone.
    last_msg_file = clone_dir / ".docbench-codex-last-message.txt"

    cmd = [
        bin_path, "exec",
        "--json",
        "--cd", ".",
        # These tools execute provider-side and therefore bypass the VM's
        # network namespace/firewall.  Disable them explicitly for benchmark
        # candidates instead of relying on host egress controls alone.
        "--config", 'web_search="disabled"',
        "--disable", "browser_use",
        "--disable", "browser_use_external",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--output-last-message", str(last_msg_file),
        "--color", "never",
    ]
    if model:
        cmd += ["--model", model]
    cmd.append(prompt)
    try:
        cmd, env, isolated_home = isolate_local_agent_process(cmd, env, clone_dir)
    except RuntimeError as exc:
        return AgentResult(
            agent="codex", model=model or "(default)", ok=False,
            verdict=f"UNKNOWN: {exc}", docs_pr_url=None, elapsed_s=0.0,
            error=str(exc),
        )

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(clone_dir), capture_output=True, text=True,
            env=env, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _cleanup_agent_container(env)
        elapsed = time.time() - t0
        partial_output, _ = _persist_timeout_trace(
            log_dir,
            exc,
            stdout_names=("codex.events.jsonl", "codex.stdout.txt"),
            stderr_name="codex.stderr.txt",
        )
        native_traces = snapshot_native_agent_traces(
            "codex", log_dir, started_at=t0, home_dir=isolated_home,
        )
        auth_persisted = _persist_refreshed_codex_auth(isolated_home)
        partial_text = _text_from_jsonl_events(partial_output) or partial_output
        return AgentResult(
            agent="codex", model=model, ok=False,
            verdict="UNKNOWN: timed out",
            docs_pr_url=None, elapsed_s=round(elapsed, 1),
            raw_output=partial_text[:20_000],
            error=f"timeout after {timeout_s}s",
            extras={
                "native_traces": native_traces,
                "primary_trace_partial": bool(partial_output),
                "codex_auth_persisted": auth_persisted,
            },
        )
    _cleanup_agent_container(env)
    elapsed = time.time() - t0

    if log_dir:
        (log_dir / "codex.events.jsonl").write_text(proc.stdout or "", encoding="utf-8")
        # Kept for backwards compatibility with older candidate audits.
        (log_dir / "codex.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        if proc.stderr:
            (log_dir / "codex.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    native_traces = snapshot_native_agent_traces(
        "codex", log_dir, started_at=t0, home_dir=isolated_home,
    )
    auth_persisted = _persist_refreshed_codex_auth(isolated_home)

    final_msg = ""
    if last_msg_file.exists():
        final_msg = last_msg_file.read_text(encoding="utf-8", errors="replace")
        last_msg_file.unlink()
    if not final_msg.strip():
        final_msg = _text_from_jsonl_events(proc.stdout or "") or proc.stdout or ""

    if brokered:
        verdict, pr_url = extract_brokered_verdict(final_msg)
    else:
        verdict, pr_url = extract_verdict_from_repo(
            mirror_repo, token,
            baseline_pr_numbers=baseline_pr_numbers,
            agent_text=final_msg,
        )
    if not brokered and _should_recover_pr(verdict, pr_url):
        # Codex commonly edits + commits the docs but skips the push/PR step.
        # Complete it on its behalf so the harness has a diff to extract.
        fb = ensure_pr_from_clone(clone_dir, mirror_repo, token,
                                  branch_suffix="codex", base_branch=base_branch,
                                  base_sha=base_sha, log_dir=log_dir)
        if fb:
            verdict, pr_url = f"DOCS_PR_URL: {fb}", fb
    return AgentResult(
        agent="codex", model=model or "(default)", ok=(proc.returncode == 0),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        raw_output=final_msg[:20_000],
        error=(proc.stderr[-500:] if proc.returncode != 0 else None),
        extras={
            "native_traces": native_traces,
            "codex_auth_persisted": auth_persisted,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpenCode CLI adapter
# ─────────────────────────────────────────────────────────────────────────────


def run_opencode(
    *,
    clone_dir: Path,
    prompt: str,
    mirror_repo: str,
    token: str,
    model: str | None = None,
    timeout_s: int = 1800,
    log_dir: Path | None = None,
    baseline_pr_numbers: set[int] | None = None,
    base_branch: str = "main",
    base_sha: str | None = None,
) -> AgentResult:
    """Invoke `opencode run` headless against `clone_dir`.

    OpenCode reads provider credentials from its auth store, environment, or the
    project's .env. For model-matrix runs, pass an explicit provider/model string
    such as `openrouter/qwen/qwen3.8-max`.
    """
    if model and model.rsplit("/", 1)[-1].lower() in {
        "qwen_3_coder", "qwen3-coder-plus", "qwen3-coder-next",
    }:
        return AgentResult(
            agent="opencode", model=model, ok=False,
            verdict="UNKNOWN: deprecated Qwen model",
            docs_pr_url=None, elapsed_s=0.0,
            error="This Qwen model is deprecated; select qwen3.8-max explicitly.",
        )
    bin_path = shutil.which("opencode")
    if not bin_path:
        return AgentResult(
            agent="opencode", model=model or "(default)", ok=False,
            verdict="UNKNOWN: opencode not on PATH",
            docs_pr_url=None, elapsed_s=0.0,
            error="opencode binary not found",
        )

    env = os.environ.copy()
    brokered = os.environ.get("DOCBENCH_PATCH_BROKER_MODE") == "1"
    if not brokered:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "true")

    cmd = [
        bin_path, "run",
        "--pure",
        "--dir", ".",
        "--title", "docbench-candidate",
        "--format", "json",
        "--print-logs",
        "--dangerously-skip-permissions",
    ]
    if model:
        cmd += ["--model", model]
    cmd.append(prompt)
    try:
        cmd, env, isolated_home = isolate_local_agent_process(cmd, env, clone_dir)
    except RuntimeError as exc:
        return AgentResult(
            agent="opencode", model=model or "(default)", ok=False,
            verdict=f"UNKNOWN: {exc}", docs_pr_url=None, elapsed_s=0.0,
            error=str(exc),
        )

    t0 = time.time()
    guard_reason: str | None = None
    try:
        proc, guard_reason = _run_opencode_with_loop_guard(
            cmd,
            env=env,
            timeout_s=timeout_s,
            cwd=clone_dir,
        )
    except subprocess.TimeoutExpired as exc:
        _cleanup_agent_container(env)
        elapsed = time.time() - t0
        partial_output, _ = _persist_timeout_trace(
            log_dir,
            exc,
            stdout_names=("opencode.events.jsonl", "opencode.stdout.txt"),
            stderr_name="opencode.stderr.txt",
        )
        native_traces = snapshot_native_agent_traces(
            "opencode", log_dir, started_at=t0, home_dir=isolated_home,
        )
        partial_text = _text_from_jsonl_events(partial_output) or partial_output
        return AgentResult(
            agent="opencode", model=model or "(default)", ok=False,
            verdict="UNKNOWN: timed out",
            docs_pr_url=None, elapsed_s=round(elapsed, 1),
            raw_output=partial_text[:20_000],
            error=f"timeout after {timeout_s}s",
            extras={
                "native_traces": native_traces,
                "primary_trace_partial": bool(partial_output),
            },
        )
    _cleanup_agent_container(env)
    elapsed = time.time() - t0

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "opencode.events.jsonl").write_text(proc.stdout or "", encoding="utf-8")
        # Kept for backwards compatibility with older candidate audits.
        (log_dir / "opencode.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        if proc.stderr:
            (log_dir / "opencode.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if guard_reason:
            (log_dir / "opencode.loop_guard.txt").write_text(guard_reason + "\n", encoding="utf-8")
    session_ids = set(re.findall(r"\bses_[A-Za-z0-9]+\b", proc.stdout or ""))
    native_traces = snapshot_native_agent_traces(
        "opencode",
        log_dir,
        started_at=t0,
        session_ids=session_ids,
        home_dir=isolated_home,
    )

    if guard_reason:
        return AgentResult(
            agent="opencode", model=model or "(default)", ok=False,
            verdict="UNKNOWN: opencode loop guard stopped run",
            docs_pr_url=None,
            elapsed_s=round(elapsed, 1),
            raw_output=(_text_from_jsonl_events(proc.stdout or "") or proc.stdout or "")[:20_000],
            error=guard_reason,
            extras={"native_traces": native_traces, "loop_guard": guard_reason},
        )

    terminal_error = _opencode_terminal_error(proc.stdout or "")
    if terminal_error:
        return AgentResult(
            agent="opencode", model=model or "(default)", ok=False,
            verdict="UNKNOWN: opencode provider/process failure",
            docs_pr_url=None,
            elapsed_s=round(elapsed, 1),
            raw_output=(_text_from_jsonl_events(proc.stdout or "") or proc.stdout or "")[:20_000],
            error=terminal_error[:2_000],
            extras={
                "native_traces": native_traces,
                "terminal_error": terminal_error[:2_000],
            },
        )

    result_text = _text_from_jsonl_events(proc.stdout or "") or proc.stdout or ""
    if brokered:
        verdict, pr_url = extract_brokered_verdict(result_text)
    else:
        verdict, pr_url = extract_verdict_from_repo(
            mirror_repo, token, baseline_pr_numbers=baseline_pr_numbers,
            agent_text=result_text,
        )
    if not brokered and _should_recover_pr(verdict, pr_url):
        fb = ensure_pr_from_clone(
            clone_dir, mirror_repo, token,
            branch_suffix="opencode", base_branch=base_branch,
            base_sha=base_sha, log_dir=log_dir,
        )
        if fb:
            verdict, pr_url = f"DOCS_PR_URL: {fb}", fb
    return AgentResult(
        agent="opencode", model=model or "(default)", ok=(proc.returncode == 0),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        raw_output=result_text[:20_000],
        error=(proc.stderr[-500:] if proc.returncode != 0 else None),
        extras={"native_traces": native_traces},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Devin API adapter
# ─────────────────────────────────────────────────────────────────────────────

DEVIN_API_BASE = os.environ.get("DEVIN_API_BASE", "https://api.devin.ai/v1")


def _restricted_third_party_url(base: str, path: str, *, allowed_host: str) -> str:
    """Build an HTTPS API URL while refusing arbitrary adapter egress."""
    parsed = urlparse(base)
    if (
        parsed.scheme != "https"
        or parsed.hostname != allowed_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"third-party adapter egress denied: expected https://{allowed_host}"
        )
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _devin_request(
    method: str, path: str, *, token: str, payload: dict | None = None, timeout: int = 90,
) -> dict:
    url = _restricted_third_party_url(
        DEVIN_API_BASE, path, allowed_host="api.devin.ai",
    )
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Devin {method} {path} → {e.code}: {raw[:400]}") from None
    if not raw.strip():
        return {}
    return json.loads(raw)


def run_devin(
    *,
    prompt: str,
    mirror_repo: str,
    token: str,                          # GitHub token (for verdict extraction)
    devin_api_key: str | None = None,
    timeout_s: int = 5400,
    poll_interval_s: int = 30,
    log_dir: Path | None = None,
    model: str = "devin",                # informational; Devin picks its own model
    baseline_pr_numbers: set[int] | None = None,
    base_branch: str = "main",
    resume_session_id: str | None = None,
    **_extras,
) -> AgentResult:
    """Create a Devin session pointed at the mirror repo and poll until done.

    Devin sessions clone the repo themselves — there's no local clone to hand
    them. The prompt must include the mirror repo URL so Devin knows where to
    work. We poll the session endpoint until status is terminal, then resolve
    the verdict from the mirror's PR list.

    Field names below (`status_enum`, `messages`, etc.) follow Devin's public
    API. If their API changes, the two places to patch are the polling loop's
    status check and the `_extract_devin_pr_url` helper.
    """
    api_key = devin_api_key or os.environ.get("DEVIN_API_KEY")
    if not api_key:
        return AgentResult(
            agent="devin", model=model, ok=False,
            verdict="UNKNOWN: DEVIN_API_KEY not set",
            docs_pr_url=None, elapsed_s=0.0,
            error="DEVIN_API_KEY missing",
        )

    # Always include the mirror URL in the prompt so Devin can find the repo.
    mirror_url = f"https://github.com/{mirror_repo}"
    full_prompt = (
        f"Repository: {mirror_url}\n\n{prompt}\n\n"
        f"Open your documentation PR against the `{base_branch}` branch of {mirror_url}."
    )

    t0 = time.time()
    # RESUME path: an existing session for this item is still running (e.g. a
    # slow task that out-ran a previous poll window). Re-poll it instead of
    # POSTing a new session — creating a new one abandons the in-flight work and
    # leaves the old session running orphaned. Only POST /sessions for a genuine
    # fresh start, never to recover from a timeout.
    if resume_session_id:
        session_id = resume_session_id
        print(f"[devin] resuming existing session {session_id} (no new session created)")
    else:
        # Devin's POST /sessions can be slow under load — retry once on timeout.
        create: dict | None = None
        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                create = _devin_request(
                    "POST", "/sessions",
                    token=api_key,
                    payload={"prompt": full_prompt},
                    timeout=120,
                )
                break
            except Exception as e:
                last_err = e
                print(f"[devin] session create attempt {attempt} failed: {e}")
                time.sleep(5)
        if create is None:
            return AgentResult(
                agent="devin", model=model, ok=False,
                verdict=f"UNKNOWN: devin session create failed: {last_err}",
                docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
                error=str(last_err),
            )

        session_id = create.get("session_id") or create.get("id")
        if not session_id:
            return AgentResult(
                agent="devin", model=model, ok=False,
                verdict="UNKNOWN: devin returned no session_id",
                docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
                error=f"raw response: {json.dumps(create)[:400]}",
                extras={"create_response": create},
            )

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "devin_session.json").write_text(json.dumps(create, indent=2), encoding="utf-8")

    deadline = t0 + timeout_s
    last_status: dict = {}
    terminal = {"finished", "blocked", "stopped", "expired"}
    while time.time() < deadline:
        try:
            last_status = _devin_request(
                "GET", f"/session/{session_id}", token=api_key,
            )
        except Exception as e:
            time.sleep(poll_interval_s)
            continue
        status = (
            last_status.get("status_enum")
            or last_status.get("status")
            or ""
        ).lower()
        if status in terminal:
            break
        time.sleep(poll_interval_s)

    elapsed = time.time() - t0
    if log_dir and last_status:
        (log_dir / "devin_final_status.json").write_text(
            json.dumps(last_status, indent=2), encoding="utf-8",
        )

    # Prefer Devin's own `pull_request` field — structured and unambiguous.
    # Fall back to the mirror PR list. As a last resort, scan ONLY Devin's
    # own messages (devin_message / devin_output) — never the
    # initial_user_message, because our prompt contains the literal marker
    # strings (DOCS_PR_URL, NO_DOC_CHANGES_NEEDED) as instructions and
    # scanning them would yield false-positive verdicts.
    pull_request = last_status.get("pull_request") or {}
    devin_pr_url = None
    if isinstance(pull_request, dict):
        devin_pr_url = pull_request.get("url") or pull_request.get("html_url")
    if devin_pr_url:
        verdict, pr_url = f"DOCS_PR_URL: {devin_pr_url}", devin_pr_url
        raw_text = json.dumps(last_status)[:20_000]
    else:
        devin_messages = [
            (m.get("message") or "") for m in (last_status.get("messages") or [])
            if m.get("type", "").startswith("devin_")
        ]
        agent_text = "\n".join(devin_messages)[:20_000]
        verdict, pr_url = extract_verdict_from_repo(
            mirror_repo, token,
            baseline_pr_numbers=baseline_pr_numbers, agent_text=agent_text,
        )
        raw_text = agent_text

    # A Devin run is "ok" if it produced a structured PR URL OR ended in
    # `finished`. `blocked` is a terminal state Devin uses for "task done,
    # awaiting confirmation" — it still counts as ok when a PR was opened.
    final_status_str = (last_status.get("status_enum") or last_status.get("status") or "").lower()
    no_doc = verdict == "NO_DOC_CHANGES_NEEDED"
    return AgentResult(
        agent="devin", model=model,
        ok=(devin_pr_url is not None)
        or ("finished" in final_status_str)
        or (no_doc and final_status_str in {"finished", "blocked"}),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        raw_output=raw_text,
        extras={"session_id": session_id, "final_status": last_status},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mintlify adapter
# ─────────────────────────────────────────────────────────────────────────────

MINTLIFY_API_BASE = os.environ.get("MINTLIFY_API", "https://api.mintlify.com")


def _mintlify_request(
    method: str, path: str, *, token: str, payload: dict | None = None, timeout: int = 60,
) -> dict:
    """Call Mintlify with curl's client fingerprint.

    Mintlify sits behind Cloudflare. Python urllib requests, even with normal
    headers, have been rejected with Cloudflare 1010, while the same request
    succeeds with curl. Use curl here so automated runs match the accepted API
    client shape. The token is passed through curl config on stdin instead of
    argv so it is not exposed in `ps`.
    """
    url = _restricted_third_party_url(
        MINTLIFY_API_BASE, path, allowed_host="api.mintlify.com",
    )
    config_lines = [
        f'url = "{url}"',
        f'request = "{method}"',
        f'header = "Authorization: Bearer {token}"',
        'header = "Accept: application/json"',
        "silent",
        "show-error",
        f"max-time = {int(timeout)}",
        'write-out = "\\n__MINTLIFY_HTTP_STATUS__:%{http_code}"',
    ]
    payload_path: str | None = None
    try:
        if payload is not None:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
                json.dump(payload, tmp)
                payload_path = tmp.name
            config_lines.extend([
                'header = "Content-Type: application/json"',
                f'data-binary = "@{payload_path}"',
            ])
        res = subprocess.run(
            ["curl", "--config", "-"],
            input="\n".join(config_lines) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if payload_path:
            try:
                Path(payload_path).unlink()
            except FileNotFoundError:
                pass

    stdout = res.stdout or ""
    marker = "\n__MINTLIFY_HTTP_STATUS__:"
    raw, _, status_text = stdout.rpartition(marker)
    status = int(status_text.strip() or "0") if status_text.strip().isdigit() else 0
    if res.returncode != 0 or status == 0 or status >= 400:
        detail = (raw or res.stderr or "").strip()
        raise RuntimeError(
            f"Mintlify {method} {path} → {status or 'curl failed'}: {detail[:400]}"
        )
    if not raw.strip():
        return {}
    return json.loads(raw)


def run_mintlify(
    *,
    prompt: str,
    mirror_repo: str,
    token: str,                          # GitHub token (for verdict extraction)
    mintlify_token: str | None = None,
    project_id: str | None = None,
    timeout_s: int = 2400,
    poll_interval_s: int = 15,
    sync_settle_s: int = int(os.environ.get("MINTLIFY_SYNC_SETTLE_S", "600")),
    job_attempts: int = 120,
    log_dir: Path | None = None,
    model: str = "mintlify",
    baseline_pr_numbers: set[int] | None = None,
    **_extras,
) -> AgentResult:
    """Trigger a Mintlify deployment sync against the project's active repo,
    then submit a documentation job with `prompt` and poll until done.

    Mintlify projects are tied to one mirror repo at a time. We assume the
    user has manually switched the Mintlify project's active directory to
    point at `mirror_repo` BEFORE invoking this adapter. The adapter does
    not (and cannot, today) change which repo a Mintlify project points at.

    Reads `prLink` from the job's final payload as the resulting PR URL.
    """
    api_key = mintlify_token or os.environ.get("MINTLIFY_TOKEN")
    if not api_key:
        return AgentResult(
            agent="mintlify", model=model, ok=False,
            verdict="UNKNOWN: MINTLIFY_TOKEN not set",
            docs_pr_url=None, elapsed_s=0.0,
            error="MINTLIFY_TOKEN missing",
        )

    project_id = project_id or os.environ.get("MINTLIFY_PROJECT_ID", "").strip()
    if not project_id:
        return AgentResult(
            agent="mintlify", model=model, ok=False,
            verdict="UNKNOWN: MINTLIFY_PROJECT_ID not set",
            docs_pr_url=None, elapsed_s=0.0,
            error="MINTLIFY_PROJECT_ID missing; configure your own project",
        )

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Wait for the AUTO-triggered sync to settle — do NOT trigger one.
    # The driver already pushed the mirror (set_to_sha + code overlay), and
    # Mintlify auto-syncs on every push to the connected repo. Firing our own
    # POST /v1/project/update here stacks a SECOND sync on the in-flight
    # push-sync, which jams Mintlify's queue into 'queued' indefinitely
    # (observed on the helm overlay batch — every item stuck in 'queued').
    # There's no status endpoint for the auto-sync, so we wait a fixed interval
    # for the push-triggered sync to finish, THEN fire the agent job. Tune via
    # MINTLIFY_SYNC_SETTLE_S.
    print(f"[mintlify] no manual sync; waiting {sync_settle_s}s for the push-triggered "
          f"sync to settle before firing the agent job…")
    time.sleep(sync_settle_s)

    # 3. Submit the agent job.
    job_payload = {"prompt": prompt}
    if log_dir:
        # Persist the exact prompt + payload we sent so each Mintlify run is
        # reproducible from the candidate's logs alone. The same prompt is
        # also written by the driver to log_dir/prompt.txt, but saving
        # alongside the job payload here keeps the Mintlify-specific view
        # self-contained.
        (log_dir / "mintlify_prompt.txt").write_text(prompt, encoding="utf-8")
        (log_dir / "mintlify_job_payload.json").write_text(
            json.dumps({
                "endpoint": f"POST {MINTLIFY_API_BASE}/v2/agent/{project_id}/job",
                "project_id": project_id,
                "payload": job_payload,
            }, indent=2),
            encoding="utf-8",
        )
    try:
        job = _mintlify_request(
            "POST", f"/v2/agent/{project_id}/job",
            token=api_key, payload=job_payload,
        )
    except Exception as e:
        return AgentResult(
            agent="mintlify", model=model, ok=False,
            verdict=f"UNKNOWN: mintlify job submit failed: {e}",
            docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
            error=str(e),
        )
    job_id = job.get("id") or job.get("jobId")
    if not job_id:
        return AgentResult(
            agent="mintlify", model=model, ok=False,
            verdict="UNKNOWN: mintlify job submit returned no id",
            docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
            error=f"raw: {json.dumps(job)[:300]}",
        )

    # 4. Poll job until terminal.
    deadline = t0 + timeout_s
    job_json: dict = {}
    job_status = ""
    while time.time() < deadline:
        try:
            job_json = _mintlify_request(
                "GET", f"/v2/agent/{project_id}/job/{job_id}", token=api_key,
            )
        except Exception:
            time.sleep(poll_interval_s)
            continue
        job_status = (job_json.get("status") or "").lower()
        if job_status in ("completed", "failed", "error"):
            break
        time.sleep(poll_interval_s)

    elapsed = time.time() - t0
    if log_dir:
        (log_dir / "mintlify_final_job.json").write_text(
            json.dumps(job_json, indent=2), encoding="utf-8",
        )

    pr_link = job_json.get("prLink") or job_json.get("pr_link")
    pr_url = pr_link if pr_link and pr_link != "null" else None
    no_doc = False
    if pr_url:
        verdict = f"DOCS_PR_URL: {pr_url}"
    else:
        # Fall back to scanning the mirror for any new non-candidate PR.
        verdict, pr_url = extract_verdict_from_repo(
            mirror_repo, token,
            baseline_pr_numbers=baseline_pr_numbers,
            agent_text=json.dumps(job_json)[:5000],
        )
        # A COMPLETED mintlify job that opened no PR (and none on the mirror) is a
        # deliberate no-op: the agent assessed the change and chose to write no
        # docs. A genuine failure ends 'failed'/'error', not 'completed'. Record
        # it as NO_DOC, not UNKNOWN (which would mislabel correct no-op resistance).
        if pr_url is None and job_status == "completed":
            verdict, no_doc = "NO_DOC_CHANGES_NEEDED", True

    return AgentResult(
        agent="mintlify", model=model,
        ok=(job_status == "completed" and (pr_url is not None or no_doc)),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        raw_output=json.dumps(job_json)[:20_000],
        extras={"project_id": project_id, "job_id": job_id, "final_job": job_json},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Promptless adapter
# ─────────────────────────────────────────────────────────────────────────────


# The Promptless API trigger does NOT fetch the code change itself — it only
# sees the `instructions` string and the freeform `context` JSON we send. So we
# hand it the code-only PR diff inline (same content the code-review agents read
# from the mirror PR) and tell it to open a docs PR against the mirror's main.
_PROMPTLESS_CODE_PR_RE = re.compile(r"PR opened at (https?://\S+)")

_PROMPTLESS_INSTRUCTIONS = (
    "A code change was just merged into this project. Review it and, if the "
    "documentation needs updating to reflect it, open a pull request against the "
    "main branch with the documentation changes. The code change is provided in "
    "the structured context below as `code_diff` (with its PR at `code_pr_url`)."
)


def _post_api_trigger(
    base_url: str, api_key: str, instructions: str, context: dict[str, Any],
    *, doc_collection_id: str | None = None, model: str | None = None,
    timeout: int = 90,
) -> dict:
    """POST one Promptless API trigger. Returns the parsed JSON response.

    When `doc_collection_id` is set, the trigger targets exactly that collection
    rather than every api pipeline in the org.
    """
    url = f"{base_url.rstrip('/')}/triggers"
    payload: dict[str, Any] = {"instructions": instructions}
    if context:
        payload["context"] = context
    if doc_collection_id:
        payload["doc_collection_id"] = doc_collection_id
    if model:
        payload["model"] = model
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Promptless POST /triggers → {e.code}: {raw[:400]}") from None
    return json.loads(raw) if raw.strip() else {}


def _gh_pr_diff_raw(pr_url: str, token: str) -> str:
    """Return the full unified diff for a PR via `gh pr diff` (unfiltered)."""
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "diff", pr_url],
        capture_output=True, text=True, env=env, check=False,
    )
    return res.stdout if res.returncode == 0 else ""


def list_mirror_branches(mirror_repo: str, token: str) -> list[str]:
    """Return all branch names on the mirror. Used to baseline before a run so
    we can spot the branch an agent pushes even when it never opens a PR."""
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{mirror_repo}/branches",
         "--jq", ".[].name"],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def _open_pr_from_branch(
    mirror_repo: str, token: str, *, branch: str, base: str, title: str, body: str,
) -> str | None:
    """Open a PR on the mirror from an already-pushed branch. Returns its URL.

    Promptless (and any agent run with auto-publish off) pushes its docs branch
    but does not open a PR. We open it here so the resulting diff surfaces in the
    same form as every other candidate; the PR content is entirely the agent's.
    """
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "pr", "create", "--repo", mirror_repo,
         "--head", branch, "--base", base, "--title", title, "--body", body],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode == 0:
        return res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else None
    # If a PR for this head already exists, recover its URL instead of failing.
    view = subprocess.run(
        ["gh", "pr", "view", branch, "--repo", mirror_repo, "--json", "url", "--jq", ".url"],
        capture_output=True, text=True, env=env, check=False,
    )
    if view.returncode == 0 and view.stdout.strip():
        return view.stdout.strip()
    print(f"[promptless] failed to open PR from branch {branch}: {res.stderr.strip()[:200]}")
    return None


def _restore_promptless_base_branch(
    mirror_path: Path | None,
    *,
    base_branch: str,
    expected_base_oid: str | None,
) -> bool:
    """Restore a benchmark base ref that disappeared during a remote run."""
    if mirror_path is None or not expected_base_oid:
        return False
    check = subprocess.run(
        ["git", "-C", str(mirror_path), "cat-file", "-e", f"{expected_base_oid}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )
    if check.returncode != 0:
        return False
    push = subprocess.run(
        ["git", "-C", str(mirror_path), "push", "origin", f"{expected_base_oid}:refs/heads/{base_branch}", "--force"],
        capture_output=True, text=True, check=False,
    )
    if push.returncode == 0:
        print(f"[promptless] restored pinned docs base {base_branch} -> {expected_base_oid[:12]}")
        return True
    print(f"[promptless] failed to restore pinned docs base {base_branch}: {push.stderr.strip()[:200]}")
    return False


def _find_new_branch(
    mirror_repo: str, token: str, baseline_branches: set[str],
) -> str | None:
    """Return one branch that appeared since `baseline_branches`, excluding our
    own code branch and known bot branches. Prefers promptless/* branches."""
    current = [b for b in list_mirror_branches(mirror_repo, token) if b not in baseline_branches]
    fresh = [
        b for b in current
        if b not in ("main", "master")
        and not any(b.startswith(p) for p in EXCLUDED_BRANCH_PREFIXES)
    ]
    if not fresh:
        return None
    promptless_branches = [b for b in fresh if b.startswith("promptless/")]
    return (promptless_branches or fresh)[0]


def _find_exact_docs_pr(
    mirror_repo: str,
    token: str,
    *,
    base_branch: str,
    head_branch: str,
    baseline_pr_numbers: set[int] | None = None,
) -> str | None:
    """Recover a docs PR created outside Promptless's suggestion table.

    Some Promptless workflows publish directly to the benchmark-provided docs
    branch. In that case the trigger is complete and the PR is real, but there
    is no suggestion row for ``dispatch_status`` to return. Matching both the
    trusted base and the per-run head keeps this recovery trigger-specific even
    on a mirror that contains outputs from other runs.
    """
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    res = subprocess.run(
        [
            "gh", "pr", "list", "--repo", mirror_repo, "--state", "all",
            "--json", "number,headRefName,baseRefName,url,createdAt",
            "--limit", "100",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        prs = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    baseline = baseline_pr_numbers or set()
    candidates = [
        pr for pr in prs
        if int(pr.get("number", -1)) not in baseline
        and pr.get("headRefName") == head_branch
        and pr.get("baseRefName") == base_branch
    ]
    candidates.sort(key=lambda pr: pr.get("createdAt") or "", reverse=True)
    return candidates[0].get("url") if candidates else None


def run_promptless(
    *,
    prompt: str,
    mirror_repo: str,
    token: str,                          # GitHub token (for PR diff + verdict extraction)
    api_trigger_key: str | None = None,
    runtime_base_url: str | None = None,
    code_pr_url: str | None = None,
    timeout_s: int = 2400,
    poll_interval_s: int = 20,
    log_dir: Path | None = None,
    model: str = "promptless",
    baseline_pr_numbers: set[int] | None = None,
    base_branch: str = "main",
    expected_docs_branch: str | None = None,
    expected_base_oid: str | None = None,
    docs_dir: str | None = None,
    analysis_version: str | None = None,
    mirror_path: Path | None = None,
    backend: PromptlessBackend | None = None,
    **_extras,
) -> AgentResult:
    """Fire the Promptless API trigger for the org whose doc collection IS the
    mirror repo, then poll the mirror until Promptless opens a docs PR.

    Promptless runs server-side (it clones its configured collection repo — the
    mirror — and opens a PR there), so there's no local clone to hand it. We
    deliver the code change inline via the trigger `context`, then resolve the
    resulting docs PR from the mirror's PR list, just like the Mintlify adapter.

    Requires the org's API trigger key (`PROMPTLESS_API_TRIGGER_KEY`) and the
    explicit runtime base URL (`PROMPTLESS_RUNTIME_BASE_URL`), plus an optional
    private backend configured through `DOGBENCH_PROMPTLESS_BACKEND`. The public
    package contains no private provisioning or database implementation.
    """
    t0 = time.time()
    # The public timeout is a hard end-to-end candidate cap. Collection
    # analysis must not silently receive a second, longer budget.
    configured_analysis_timeout_s = int(
        os.environ.get(
            "PROMPTLESS_COLLECTION_ANALYSIS_TIMEOUT_S",
            str(timeout_s),
        )
    )
    analysis_timeout_s = min(timeout_s, configured_analysis_timeout_s)
    api_key = api_trigger_key or os.environ.get("PROMPTLESS_API_TRIGGER_KEY")
    base_url = runtime_base_url or os.environ.get("PROMPTLESS_RUNTIME_BASE_URL")
    if not api_key:
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict="UNKNOWN: PROMPTLESS_API_TRIGGER_KEY not set",
            docs_pr_url=None, elapsed_s=0.0,
            error="PROMPTLESS_API_TRIGGER_KEY missing",
        )

    if not base_url:
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict="UNKNOWN: PROMPTLESS_RUNTIME_BASE_URL not set",
            docs_pr_url=None, elapsed_s=0.0,
            error="PROMPTLESS_RUNTIME_BASE_URL missing; configure your own endpoint",
        )
    try:
        backend = load_promptless_backend(backend)
    except RuntimeError as exc:
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict=f"UNKNOWN: {exc}", docs_pr_url=None, elapsed_s=0.0,
            error=str(exc),
        )

    # The code PR lives on the mirror; prefer the explicit arg, else parse it
    # from the code-review prompt ("...PR opened at <url>...").
    if not code_pr_url:
        m = _PROMPTLESS_CODE_PR_RE.search(prompt or "")
        code_pr_url = m.group(1) if m else None

    code_diff = _gh_pr_diff_raw(code_pr_url, token) if code_pr_url else ""
    context: dict[str, Any] = {}
    if code_pr_url:
        context["code_pr_url"] = code_pr_url
    if code_diff:
        context["code_diff"] = code_diff

    # Forward the SAME canonical task prompt every other agent receives, so the
    # input is identical across agents and only agent quality varies. The other
    # adapters consume `prompt` directly (CLI agents as their task text, mintlify
    # as its job prompt); promptless used to discard it and send a hardcoded
    # "a code change was merged" instruction with only the code diff — which left
    # docs_only tasks (no code diff) with no context at all. Now the shared
    # prompt is the trigger instruction; code_pr_url/code_diff remain as
    # supplementary structured context (the same content CLI agents read from
    # the mirror PR). Fall back to the generic instruction only if no prompt was
    # supplied.
    instructions = (prompt or "").strip() or _PROMPTLESS_INSTRUCTIONS.replace(
        "main branch", f"{base_branch} branch"
    )

    # Ensure this mirror repo has a Promptless doc collection in the configured
    # account, and target that collection in the trigger so it does not fan out to
    # other repos' collections. This must be a hard precondition: an untargeted
    # trigger can dispatch against the wrong collection in the configured account.
    repo_url = f"https://github.com/{mirror_repo}"
    doc_collection_id: str | None = None
    try:
        doc_collection_id = backend.ensure_collection(
            repo_url,
            docs_dir=docs_dir,
            analysis_version=analysis_version,
            timeout_s=max(1, analysis_timeout_s),
        )
        if not isinstance(doc_collection_id, str) or not doc_collection_id.strip():
            raise RuntimeError("Promptless backend returned no collection ID")
        if analysis_version:
            if mirror_path is None:
                raise RuntimeError("Promptless trigger overlay requires the local benchmark mirror")
            overlay = backend.analysis_overlay(
                repo_path=mirror_path,
                repo_url=repo_url,
                collection_id=doc_collection_id,
                current_docs_tree=analysis_version,
            )
            if overlay:
                context["dogbench_collection_analysis_overlay"] = overlay
                instructions += (
                    "\n\nDogBench analysis overlay: the structured context contains "
                    "`dogbench_collection_analysis_overlay`. Validate that its "
                    "`doc_collection_id` matches this run's mounted collection. "
                    "Use the mounted collection analysis as the baseline, then apply "
                    "the original-baseline `documentation_change_summary` and changed-path "
                    "manifest as an ephemeral update for "
                    "research and scoping in this run. The overlay supersedes stale "
                    "baseline conclusions where they conflict. Do not publish, persist, "
                    "or write this overlay to the Knowledge Base or collection."
                )
    except Exception as e:  # noqa: BLE001 - surface provisioning failures as candidate failures
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict=f"UNKNOWN: promptless collection provisioning failed: {e}",
            docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
            error=str(e),
            extras={"mirror_repo": mirror_repo, "repo_url": repo_url},
        )

    # A configured Promptless account may contain multiple collections. The API's
    # doc_collection_id targets dispatch bookkeeping, but the worker still needs
    # the repository identity in its natural-language instructions to know which
    # collection to mount. Without this, a worker can search/mount unrelated
    # collections while trying to infer what "this repository" means.
    instructions = (
        f"{instructions}\n\n"
        f"The only documentation repository for this run is `{mirror_repo}` "
        f"(Promptless doc collection `{doc_collection_id}`). "
        "Mount, inspect, and modify exactly that repository. "
        "Do not mount or inspect any other repository or documentation collection."
    )

    # Baseline the mirror's branches just before triggering. Promptless runs
    # with the pipeline's auto_publish off, so it pushes its docs branch but
    # opens no PR — we detect that new branch and open the PR ourselves.
    baseline_branches = set(list_mirror_branches(mirror_repo, token))

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "promptless_trigger_request.json").write_text(
            json.dumps({
                "endpoint": f"POST {base_url.rstrip('/')}/triggers",
                "instructions": instructions,
                "context_keys": sorted(context.keys()),
                "analysis_overlay": (
                    {
                        key: value
                        for key, value in context.get(
                            "dogbench_collection_analysis_overlay", {}
                        ).items()
                        if key != "documentation_change_summary"
                    }
                    or None
                ),
                "code_pr_url": code_pr_url,
                "doc_collection_id": doc_collection_id,
                "base_branch": base_branch,
                "analysis_version": analysis_version,
                "model": model,
            }, indent=2),
            encoding="utf-8",
        )
    trigger_t0 = time.time()
    try:
        resp = _post_api_trigger(
            base_url, api_key, instructions, context,
            doc_collection_id=doc_collection_id,
            model=None if model == "promptless" else model,
        )
    except RuntimeError as e:
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict=f"UNKNOWN: promptless trigger submit failed: {e}",
            docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
            error=str(e),
            extras={"doc_collection_id": doc_collection_id, "mirror_repo": mirror_repo},
        )
    trigger_event_id = resp.get("trigger_event_id")
    if not isinstance(trigger_event_id, str) or not trigger_event_id.strip():
        return AgentResult(
            agent="promptless", model=model, ok=False,
            verdict="UNKNOWN: promptless trigger returned no trigger_event_id",
            docs_pr_url=None, elapsed_s=round(time.time() - t0, 1),
            error=f"raw: {json.dumps(resp)[:300]}",
            extras={"trigger_response": resp},
        )
    trigger_event_id = trigger_event_id.strip()
    print(f"[promptless] trigger_event_id={trigger_event_id}; polling mirror for docs PR/branch…")
    if log_dir:
        (log_dir / "promptless_trigger_response.json").write_text(
            json.dumps(resp, indent=2), encoding="utf-8",
        )

    # Poll the mirror for promptless's output (a docs PR or pushed branch). We
    # ALSO poll the trigger's resolution in the runtime DB every ~2 min so we can
    # (a) stop as soon as the trigger resolves instead of waiting out the whole
    # timeout, and (b) tell apart four outcomes a bare branch-poll cannot:
    # produced-docs (recover the suggestion PR), genuine no-change, crashed
    # dispatch, and still-running. A docs PR may never land on the mirror as a
    # NEW branch — e.g. a fast dedup (resolved against a prior identical trigger,
    # no new branch) or a slow run that finishes after our timeout.
    def _check_status() -> "dict | None":
        try:
            return backend.dispatch_status(trigger_event_id)
        except Exception as e:  # noqa: BLE001 - best-effort
            print(f"[promptless] dispatch_status unavailable ({e})")
            return None

    deadline = t0 + timeout_s
    pr_url: str | None = None
    new_branch: str | None = None
    status: dict | None = None
    status_unavailable = False
    last_status_check = 0.0
    _STATUS_EVERY_S = 60.0
    completed_without_suggestion_at: float | None = None
    terminal_resolution_observed = False
    completed_grace_s = min(120.0, max(float(poll_interval_s), timeout_s * 0.25))
    # PRIMARY signal is the trigger's OWN resolution, keyed by trigger_event_id
    # (dispatch_status → suggestion_pr). On a persistent continual-learning
    # mirror the PR/branch list accumulates across items, so baseline-diffing it
    # cross-contaminates — it can grab a PR/branch produced for a DIFFERENT item
    # (the bug that gave two items the same diff and others a wrong-topic branch).
    # The trigger-keyed suggestion is unambiguous, so we trust it first and only
    # fall back to baseline polling when the runtime DB is unreachable.
    while time.time() < deadline:
        if not status_unavailable and (time.time() - last_status_check) >= _STATUS_EVERY_S:
            last_status_check = time.time()
            status = _check_status()
            if status is None:
                status_unavailable = True
            elif status.get("suggestion_pr"):
                pr_url = status["suggestion_pr"]          # authoritative for THIS trigger
                terminal_resolution_observed = True
                break
            elif status.get("suggestion_branch"):
                new_branch = status["suggestion_branch"]  # authoritative branch for THIS trigger
                terminal_resolution_observed = True
                break
            elif status.get("failed"):
                terminal_resolution_observed = True
                break                                     # dispatch failed → handled below
            elif str(status.get("effective_status") or status.get("trigger_status", "")).startswith("completed"):
                if expected_docs_branch:
                    pr_url = _find_exact_docs_pr(
                        mirror_repo,
                        token,
                        base_branch=base_branch,
                        head_branch=expected_docs_branch,
                        baseline_pr_numbers=baseline_pr_numbers,
                    )
                    if pr_url:
                        print(
                            "[promptless] recovered exact benchmark docs PR "
                            f"{pr_url} (base={base_branch}, head={expected_docs_branch})"
                        )
                        break
                # The trigger can be marked completed just before the
                # suggestion row/branch becomes visible. Give branch-only
                # suggestions a short window to surface before calling no-doc.
                now = time.time()
                if completed_without_suggestion_at is None:
                    completed_without_suggestion_at = now
                elif now - completed_without_suggestion_at >= completed_grace_s:
                    terminal_resolution_observed = True
                    break                                 # resolved, no suggestion → no-change
                last_status_check = 0.0
        if status_unavailable:
            # Degraded mode only: the runtime DB is unreachable, so we can't key
            # to the trigger. Baseline-diff the mirror (may be contaminated on a
            # shared collection mirror — last resort).
            _verdict, pr_url = extract_verdict_from_repo(
                mirror_repo, token, baseline_pr_numbers=baseline_pr_numbers, agent_text="",
            )
            if pr_url:
                break
            new_branch = _find_new_branch(mirror_repo, token, baseline_branches)
            if new_branch:
                break
        time.sleep(poll_interval_s)

    # A status read after the deadline may discover that the runtime row has
    # since become completed. That must not retroactively turn a capped run
    # into an accepted no-op: no-change is valid only when terminal completion
    # was observed (including its publication grace) before the cap expired.
    timeout_reached = bool(
        timeout_s > 0
        and time.time() >= deadline
        and not terminal_resolution_observed
    )

    # Ensure we have a final resolution read when nothing surfaced yet. This
    # matters for branch-only suggestions that arrive just after the trigger is
    # marked completed.
    if not pr_url and not new_branch and not status_unavailable:
        final_status = _check_status()
        if final_status is not None:
            status = final_status
            if status.get("suggestion_pr"):
                pr_url = status["suggestion_pr"]
            elif status.get("suggestion_branch"):
                new_branch = status["suggestion_branch"]
            elif str(status.get("effective_status") or status.get("trigger_status", "")).startswith("completed") and expected_docs_branch:
                pr_url = _find_exact_docs_pr(
                    mirror_repo,
                    token,
                    base_branch=base_branch,
                    head_branch=expected_docs_branch,
                    baseline_pr_numbers=baseline_pr_numbers,
                )
                if pr_url:
                    print(
                        "[promptless] recovered exact benchmark docs PR "
                        f"{pr_url} (base={base_branch}, head={expected_docs_branch})"
                    )

    opened_pr = False
    if not pr_url and new_branch:
        # Promptless pushed a docs branch but opened no PR; open it so the diff
        # surfaces like every other candidate. Content is entirely promptless's.
        _restore_promptless_base_branch(
            mirror_path,
            base_branch=base_branch,
            expected_base_oid=expected_base_oid,
        )
        pr_url = _open_pr_from_branch(
            mirror_repo, token,
            branch=new_branch, base=base_branch,
            title="docs: update for code change",
            body="Documentation update produced by Promptless for the code change under review.",
        )
        opened_pr = pr_url is not None

    elapsed = time.time() - t0
    verdict_verified: bool | None = None
    dispatch_detail = ""
    resolution_reason = None
    trigger_status = None
    recovered_pr = None
    incomplete = False
    if not pr_url and not new_branch:
        if timeout_reached:
            incomplete = True
            trigger_status = status.get("trigger_status") if status else None
        elif status is None:
            verdict_verified = None  # couldn't check the runtime DB
        else:
            verdict_verified = True
            resolution_reason = status.get("resolution_reason")
            trigger_status = status.get("trigger_status")
            if status.get("failed"):
                dispatch_detail = status.get("detail") or "dispatch failed"
            elif status.get("suggestion_pr"):
                # Docs WERE produced — either by this trigger (doc_agent_resolved)
                # or by the trigger this one duplicates — but the branch poll
                # missed it (slow run, or dedup produced no new branch). Capture
                # that PR so it's a real candidate, not a false no-change.
                recovered_pr = status["suggestion_pr"]
                pr_url = recovered_pr
                print(f"[promptless] recovered docs PR via {resolution_reason or 'suggestion'}: {recovered_pr}")
            elif status.get("suggestion_branch"):
                # Docs WERE produced but the Promptless org's publish policy did
                # not auto-create a PR. Open one from the suggestion branch so
                # the candidate is represented like every other agent output.
                new_branch = status["suggestion_branch"]
            elif not str(status.get("effective_status") or trigger_status or "").startswith("completed"):
                # Timed out before the trigger resolved — NOT a no-change.
                incomplete = True

    if pr_url:
        verdict = f"DOCS_PR_URL: {pr_url}"
    elif new_branch:
        verdict = f"UNKNOWN: promptless pushed branch {new_branch} but PR open failed"
    elif dispatch_detail:
        # A crashed dispatch is NOT a docs judgement — surface it as an error so
        # it isn't mistaken for a real "no change needed".
        verdict = f"UNKNOWN: promptless dispatch failed: {dispatch_detail}"
    elif incomplete:
        verdict = f"UNKNOWN: promptless trigger did not resolve within {timeout_s}s (status={trigger_status})"
    else:
        verdict = "NO_DOC_CHANGES_NEEDED"

    error_msg = dispatch_detail or (
        f"trigger did not resolve within {timeout_s}s (status={trigger_status})" if incomplete else None
    )
    promptless_trace: dict | None = None
    try:
        promptless_trace = backend.export_trace(trigger_event_id)
    except Exception as e:  # noqa: BLE001 - best-effort
        print(f"[promptless] trace export unavailable ({e})")
    if log_dir and promptless_trace is not None:
        (log_dir / "promptless_trace.json").write_text(
            json.dumps(promptless_trace, indent=2, default=str),
            encoding="utf-8",
        )
    return AgentResult(
        agent="promptless", model=model,
        ok=(pr_url is not None) or (verdict == "NO_DOC_CHANGES_NEEDED" and not error_msg),
        verdict=verdict, docs_pr_url=pr_url,
        elapsed_s=round(elapsed, 1),
        error=error_msg,
        raw_output=json.dumps({"trigger_event_id": trigger_event_id, "trigger_response": resp})[:20_000],
        extras={
            "trigger_event_id": trigger_event_id,
            "runtime_base_url": base_url,
            "doc_collection_id": doc_collection_id,
            "analysis_version": analysis_version,
            "promptless_branch": new_branch,
            "pr_opened_by_harness": opened_pr,
            "verdict_verified": verdict_verified,
            "resolution_reason": resolution_reason,
            "effective_status": (
                status.get("effective_status") if status else None
            ),
            "recovered_suggestion_pr": recovered_pr,
            "promptless_trace": "promptless_trace.json" if promptless_trace is not None else None,
            "timeout_reached": timeout_reached,
        },
    )
