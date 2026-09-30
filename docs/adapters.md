# Running the retained agent adapters

The package retains the Claude Code, Codex, OpenCode, Devin, Mintlify, and
Promptless adapter entry points. Local execution initially supports Linux with
Docker or Podman and passwordless `sudo` for the selected runtime. Provider jobs
are billable. No adapter runs during installation, validation, or the offline tests.

The local adapters preserve the original command construction, verdict parsing,
trace capture, timeout handling, OpenCode loop guards, and controller patch
recovery. Cloud adapters preserve their provider request and polling contracts.
Cloud services have provider-controlled environments; the local container egress
policy does not apply to them. Use an explicit model and record the returned
provider metadata when reporting results.

## Linux container setup

Install the package, Git, your chosen container runtime, and the corresponding
agent CLI on the host. The host executable must be on `PATH`; the actual candidate
runs the executable from the container image. The shipped image pins Claude Code
2.1.195, Codex 0.144.6, and OpenCode 1.17.11, matching the extracted source.
The original container runs as UID/GID `1000:1000`. Its mounted checkout and
disposable home must be writable by that identity, and prepared code must be
readable. The setup script does not change ownership of operator files.

Run the setup script only after choosing your own local resource names:

```sh
export DOCBENCH_AGENT_SANDBOX_BACKEND=docker
export DOCBENCH_AGENT_SANDBOX_IMAGE=dogbench-agent:local
export DOCBENCH_AGENT_SANDBOX_NETWORK=dogbench-candidates-internal
export DOCBENCH_EGRESS_NETWORK=dogbench-proxy-outbound
export DOCBENCH_PROXY_NAME=dogbench-provider-proxy
export DOCBENCH_PROXY_IMAGE=docker.io/ubuntu/squid:6.10-24.10_edge
export DOCBENCH_AGENT_EGRESS_PROXY=http://dogbench-provider-proxy:3128
export DOCBENCH_REQUIRE_AGENT_FS_ISOLATION=1

# From a source checkout:
bash dogbench/assets/agent-sandbox/setup_linux.sh
```

For an installed package, find the asset directory with:

```sh
python -c 'from importlib.resources import files; print(files("dogbench") / "assets/agent-sandbox")'
```

The script builds the pinned image and creates the two networks and Squid proxy.
It checks that the candidate network is internal and refuses to replace an
existing proxy container. It does not install a runtime or change system services.
The proxy permits only the provider/authentication hosts listed in `squid.conf`;
it has no published host port. Keep provider additions explicit in that file.

Qwen3.8 Max uses the retained OpenCode sibling-filter patch and a separate image:

```sh
export DOCBENCH_QWEN_SANDBOX_IMAGE=dogbench-agent-qwen:local
sudo -n docker build \
  --build-arg BASE_AGENT_IMAGE="$DOCBENCH_AGENT_SANDBOX_IMAGE" \
  -t "$DOCBENCH_QWEN_SANDBOX_IMAGE" \
  -f dogbench/assets/agent-sandbox/Containerfile.qwen \
  dogbench/assets/agent-sandbox
```

The adapter requires that explicit image setting when the requested command
contains `qwen/qwen3.8-max`. It does not silently substitute the unpatched image.
The legacy mount-namespace test backend and execution without isolation are not
supported by this public runner.

## Local credentials and invocation

Select the credential files you intend to use; the runner does not discover or
copy accounts from the host home directory:

| Adapter | Explicit configuration |
| --- | --- |
| Claude Code account | `DOCBENCH_CLAUDE_AUTH_FILE=/absolute/path/to/.credentials.json` |
| Claude through Bedrock | `DOCBENCH_CLAUDE_CODE_USE_BEDROCK=1` plus your AWS credential configuration |
| Codex account | `DOCBENCH_CODEX_AUTH_FILE=/absolute/path/to/auth.json` |
| OpenCode account/provider | `DOCBENCH_OPENCODE_AUTH_FILE=/absolute/path/to/auth.json`; optional `DOCBENCH_OPENCODE_CONFIG_FILE=/absolute/path/to/opencode.jsonc` |
| OpenCode through OpenRouter | `OPENROUTER_API_KEY` in the controller environment |

Only configured files are copied into each disposable container home. Local
candidates do not receive the controller GitHub credential. Claude account mode
retains the original suppression of unrelated `ANTHROPIC_*` credentials; an
inherited API key does not silently change that lane's billing route. Bedrock
uses explicit AWS credentials or the AWS CLI credential export mechanism, without
exposing instance metadata to the container.

Codex refresh-token updates are persisted only to the explicitly configured
`DOCBENCH_CODEX_AUTH_FILE`, after checking account identity. Invocations sharing
that file are serialized. `DOCBENCH_CODEX_AUTH_LOCK` can specify the lock path;
otherwise it sits beside the configured auth file. With no account auth file,
the lock uses an account-specific temporary filename.

After preparing the frozen items with the main README workflow:

```sh
export DOCBENCH_CODEX_AUTH_FILE=/absolute/path/to/your/auth.json
dogbench run items.jsonl --prepared prepared-inputs \
  --agent codex --model YOUR_MODEL_ID --timeout 1800 \
  --output runs/codex-first-run
```

Use `--agent claude` or `--agent opencode` with their exact model identifiers for
the other local adapters. Each run needs a new output directory. The run output
contains the prediction transport, captured patches, native traces where the CLI
exposes them, and audit results. The runner processes each requested item once;
it does not silently resume a prior provider session.

## Preparing a cloud mirror with the original helpers

Install the cloud extra and GitHub CLI first:

```sh
pip install '.[cloud]'
```

This extra supplies `PyYAML>=6.0.3`, used by the retained MkDocs-to-Mintlify
navigation converter. `dogbench.cloud_mirrors.Mirror` retains private repository
creation, neutral snapshots, managed identity scrubbing, code PR construction,
external code overlays, and cloning. `configure_mirrors()` requires explicit
controller storage; `Mirror.ensure()` requires your GitHub owner, work directory,
and token. There is no built-in organization, account, project, or cache path.

The following setup example **creates a private repository and code PR in your
specified account, and pushes frozen snapshots**. It uses one item from
`items.one.jsonl`; prepare that same one-item file before invoking the runner.
Choose a separate mirror for each independent candidate run.

```python
import json
import os
import subprocess
import uuid
from pathlib import Path
from dogbench.cloud_mirrors import Mirror, configure_mirrors

item = json.loads(Path("items.one.jsonl").read_text())
root = Path("cloud-state").resolve()
mirror_overlays = root / "mirror-overlays"
source_overlays = root / "source-overlays"
mirror_overlays.mkdir(parents=True, exist_ok=True)
source_overlays.mkdir(parents=True, exist_ok=True)
configure_mirrors(
    index_path=root / "mirror-index.json",
    mirror_overlays_root=mirror_overlays,
    source_overlays_root=source_overlays,
)
def slug(url):
    return url.removeprefix("https://github.com/").removesuffix(".git")

mirror = Mirror.ensure(
    slug(item["docs"]["repo_url"]), os.environ["DOGBENCH_GITHUB_TOKEN"],
    org=os.environ["DOGBENCH_MIRROR_OWNER"], work_dir=root / "clones",
    mirror_name=uuid.uuid4().hex,
)
mirror.ensure_sha(item["docs"]["base_sha"])
base_sha = mirror.set_to_sha(item["docs"]["base_sha"])
code_pr_url = None
code_prefix = ""
code = item.get("code")
if code:
    if slug(code["repo_url"]) == mirror.source_repo:
        mirror.ensure_sha(code["base_sha"])
        mirror.ensure_sha(code["head_sha"])
        code_pr_url = mirror.open_code_only_pr(
            code["base_sha"], code["head_sha"],
            pr_file_allowlist=set(code["paths"]),
        )
    else:
        code_prefix = "_code_repo/"
        base_sha = mirror.overlay_external_tree(slug(code["repo_url"]), code["base_sha"])
        code_pr_url = mirror.open_overlay_code_pr(
            slug(code["repo_url"]), code["base_sha"], code["head_sha"],
            pr_file_allowlist=set(code["paths"]),
        )
    if code_pr_url is None:
        raise RuntimeError("No code PR was produced; inspect setup output before running")
# open_code_only_pr can leave the local checkout on its input branch.
subprocess.run(["git", "-C", str(mirror.mirror_dir), "checkout", "main"], check=True)

config = {
    "github_token_env": "DOGBENCH_GITHUB_TOKEN",
    "transport_mode": "managed",
    "mirror_overlays_root": str(mirror_overlays),
    "source_overlays_root": str(source_overlays),
    "items": {
        item["instance_id"]: {
            "mirror_repo": mirror.mirror_repo,
            "clone_dir": str(mirror.mirror_dir),
            "base_branch": "main",
            "base_sha": base_sha,
            "docs_work_branch": "candidate/docs",
            "code_pr_url": code_pr_url,
            "code_pr_path_prefix": code_prefix,
        }
    },
}
Path("cloud-run.json").write_text(json.dumps(config, indent=2) + "\n")
```

The original managed transform removes `.github`, generates or preserves the
Mintlify navigation configuration, applies explicitly configured transport
overlays, and scrubs upstream identifiers using the original rules. The runner
recreates those transformations locally from frozen inputs and compares the
entire expected Git tree with the configured mirror. It also binds the overlay
contents and validates the transformed code PR paths and blobs. A changed file
is not accepted merely because its name resembles a transport file.

`transport_mode: "exact"` supports a separately prepared mirror with the
unmodified frozen tree; it does not accept managed transformations. Creating
that alternative mirror requires handling any upstream automation separately.
The standard example above uses the original managed transport.

`Mirror.recreate()`, `reset_to_base()`, and cleanup methods retain their original
repository/branch mutation semantics. They are explicit operator tools; the
public run command does not invoke them to recycle an old run automatically.

## Cloud provider configuration

Add the following top-level fields to `cloud-run.json`. Keep secret values in
the named environment variables, outside the JSON file:

| Adapter | Additional fields |
| --- | --- |
| Devin | `"devin_api_key_env": "DEVIN_API_KEY"` |
| Mintlify | `"mintlify_token_env": "MINTLIFY_TOKEN"`, `"project_id": "YOUR_PROJECT_ID"`, `"project_mirror_confirmed": true` |
| Promptless | `"promptless_api_trigger_key_env": "PROMPTLESS_API_TRIGGER_KEY"`, `"runtime_base_url": "YOUR_EXPLICIT_RUNTIME_URL"` |

Mintlify must already be configured by the operator to use the specified mirror;
the adapter does not select or reconfigure an account's project. It preserves the
original wait for the mirror-push sync (`MINTLIFY_SYNC_SETTLE_S`, default 600 seconds)
before submitting its documentation job. The former fixed project ID is removed.

For example, after adding the Devin field and setting the two credentials:

```sh
dogbench run items.one.jsonl --prepared prepared-one \
  --agent devin --model devin --timeout 5400 \
  --config cloud-run.json --output runs/devin-first-run
```

The controller verifies the mirror is private, belongs to the configured
repository URL, differs from the upstream source, and matches the frozen input
and branch. Cloud runs may create or recover documentation PRs in that mirror.

## Optional Promptless backend

The public package contains the API-trigger submission, mirror polling, terminal
status handling, patch/PR recovery, and trace-writing behavior. It excludes the
private provisioning, analysis, database, and trace-access implementations.
Select an operator-owned plugin using:

```sh
export DOGBENCH_PROMPTLESS_BACKEND=your_package.dogbench_backend:create_backend
```

The no-argument factory must return an implementation of
`dogbench.promptless_backend.PromptlessBackend`:

- `ensure_collection(repo_url, *, docs_dir, analysis_version, timeout_s) -> str`
  returns an attested collection ID in the configured account.
- `analysis_overlay(*, repo_path, repo_url, collection_id, current_docs_tree)`
  returns an attested transient analysis overlay or `None`.
- `dispatch_status(trigger_event_id)` returns the trigger-keyed status dictionary
  used by the original adapter: `suggestion_pr`, `suggestion_branch`, `failed`,
  `effective_status`, `trigger_status`, `detail`, and `resolution_reason` as
  applicable. Missing status enables the original degraded mirror polling; it
  does not establish completion.
- `export_trace(trigger_event_id)` returns a sanitized trace dictionary or `None`.

The plugin supplies its own explicitly configured account credentials and private
connections. A Python caller may instead pass `backend=` directly to
`run_promptless()`. Without a configured backend, endpoint, or trigger key, the
adapter returns an explicit unavailable/configuration error. It never treats
missing infrastructure as a completed no-op or invokes a private fallback.
