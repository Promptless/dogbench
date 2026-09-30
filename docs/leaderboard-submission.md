# Leaderboard submission guide

Submit your agent for review on the [DogBench leaderboard](https://dogbench.ai).
This guide covers the evidence to provide, the commands available in this repo,
and the review process. Submissions are reviewed manually through GitHub.

## Choose a submission path

| Path | What to provide | What it establishes |
| --- | --- | --- |
| Public development results | A complete run on a pinned development release, with predictions, scores, and traces | Reproducible development performance; the labels and rubrics are public |
| Held-out leaderboard evaluation | A runnable agent configuration or access instructions, plus a development run demonstrating integration | A request for maintainers to evaluate the agent on the private split |

The public release contains 175 development items. The leaderboard's held-out
split contains 117 items: 82 requiring a documentation update and 35 requiring
no update. Development scores and five-item sample scores must be labeled with
their split and coverage; they are not interchangeable with held-out scores.

For a held-out evaluation, open an issue before investing in a full run.
Maintainers must agree on integration, access, execution limits, and evaluation
availability. The private tasks, rubrics, and reference changes stay under
maintainer control. A submitted development score does not automatically create
a leaderboard entry.

## Requirements

- **Use one fixed configuration.** Record the exact model ID, harness version,
  prompts, tools, budgets, and agent image. Disclose routing, fine-tuning, custom
  tools, and benchmark-specific changes.
- **Include every task in the declared release.** Preserve failed runs, invalid
  patches, and abstentions. Do not omit difficult items or select only successes.
- **Run each task once.** Disclose retries, repeated runs, best-of selection,
  and manual intervention. Report a changed attempt policy separately. Internal
  model-call retries must follow the documented adapter behavior.
- **Keep evaluation material outside the agent.** Do not expose rubrics, labels,
  reference patches, historical trajectories, or another task's context. Disclose
  any prior use of DoGBench material for training or agent development.
- **Restrict source lookup.** Local agents should use the runner's isolation and
  allow only necessary model-provider endpoints. For cloud agents, document
  which restrictions can be enforced and supply the available access logs and
  native traces. An instruction not to browse is not proof that browsing was blocked.
- **Retain evidence.** Keep task-level predictions, execution outcomes, traces,
  audits, and judge decisions. Explain missing evidence and redactions.

See the [README](../README.md), [execution guide](execution.md), and
[known issues](known-issues.md) for execution, scoring, and reproduction limits.
Modified protocols need explicit review before comparison with the leaderboard.

## 1. Install and check your integration

Run these commands from the root of this source checkout. Install this checkout
so the `dogbench` command uses its implementation; do not assume an unrelated
PyPI installation has the same commands or scoring behavior.

You need Python 3.11+, Git, and curl. Preparation and canonical validation
contact GitHub; set `GITHUB_TOKEN` or `GH_TOKEN` on the trusted controller if
needed for API access or rate limits. Keep that credential outside the agent.
The package does not automatically read the GitHub CLI's stored login. If you
already authenticated with `gh`, expose that token to the controller explicitly:

```sh
export GH_TOKEN="$(gh auth token)"
```

Do not include its value in logs or submission files. If preparation stops on a
GitHub API rate limit, configure authentication or wait for the limit to reset,
then prepare again in a fresh output directory.

Local agent execution requires **Linux**, Docker or Podman, passwordless sudo
for the runtime, and the selected agent CLI on the host. Before `dogbench run`,
complete [Linux container setup and explicit credential configuration](adapters.md).
That setup builds the agent image and configures the internal network and
provider proxy. Installing the Python package alone does not prepare a runnable
agent. On macOS or Windows, use a Linux execution host for local adapters.

Scoring invokes the **host Codex CLI** with `gpt-5.6-terra` by default. Configure
its authentication separately from the candidate's container credentials.
Agent and judge calls may incur charges; scoring defaults to up to 20 attempts
per scored patch. `score --model` and `--max-attempts` change that configuration
and must be recorded.

Install and inspect the CLI first:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,cloud]'
dogbench --version
dogbench --help
```

Then prepare the five-item integration sample:

```sh
dogbench prepare data/sample-5/items.jsonl --output workspaces/sample-r2
dogbench verify workspaces/sample-r2 --json
dogbench run data/sample-5/items.jsonl --prepared workspaces/sample-r2 \
  --agent codex --model MODEL_ID --output runs/sample-r2
dogbench validate runs/sample-r2/predictions.jsonl \
  --items data/sample-5/items.jsonl --traces runs/sample-r2/traces --json
dogbench score runs/sample-r2/predictions.jsonl \
  --outcomes data/sample-5/outcomes.jsonl --output scores/sample-r2
dogbench report scores/sample-r2/scores.json \
  --outcomes data/sample-5/outcomes.jsonl --json
```

Replace `MODEL_ID` with an accessible model ID and configure the provider before
running. Run each command separately and inspect its exit status before moving
on. Preparation or verification failures must be resolved before dispatch. If
`run` exits nonzero, inspect `run.json` for completion, failed items, and its stop
reason. Preserve those artifacts; do not silently retry or drop failed items.
This CLI uses `--agent claude`, `codex`, `opencode`, `devin`, `mintlify`, or `promptless`.
Use fresh workspace, run, and score directories for each new run. Preparation
can reject approved dataset context under the retained checks; see
[known issues](known-issues.md). Report that blocker instead of editing the task
or bypassing attestation to manufacture a valid run. All five samples
require patches, so they test integration but yield no headline composite.

## 2. Run the public development release

Download both files from one immutable Hugging Face revision. The revision in
[data/release-reference.json](../data/release-reference.json) is pinned below.
Authentication is optional for this public dataset. Record the runner revision
with `git rev-parse HEAD` and preserve any local changes as a diff. Keep the
runner code and installed assets fixed from preparation through verification,
execution, and scoring. After changing them, prepare fresh workspaces before
starting a new run.

```sh
mkdir -p data/development
curl -fL https://huggingface.co/datasets/promptless-research/dogbench-dev/resolve/f8e5dd2040414810cc79e8f77167c84a12d76932/items.jsonl -o data/development/items.jsonl
curl -fL https://huggingface.co/datasets/promptless-research/dogbench-dev/resolve/f8e5dd2040414810cc79e8f77167c84a12d76932/outcomes.jsonl -o data/development/outcomes.jsonl
```

Verify both downloads against the pinned hashes before preparation:

```sh
python - <<'PYCODE'
import hashlib
import json
from pathlib import Path
reference = json.loads(Path("data/release-reference.json").read_text())
for name, metadata in reference["files"].items():
    actual = hashlib.sha256((Path("data/development") / name).read_bytes()).hexdigest()
    if actual != metadata["sha256"]:
        raise SystemExit(f"Checksum mismatch: {name}")
print("Release checksums verified")
PYCODE

dogbench prepare data/development/items.jsonl --output workspaces/development
dogbench verify workspaces/development --json
dogbench run data/development/items.jsonl --prepared workspaces/development \
  --agent codex --model MODEL_ID --output runs/development
```

Use all 175 items for a full development submission. For a custom harness,
produce records matching the [prediction schema](../dogbench/schemas/prediction-v1.schema.json)
and preserve the traces and configuration. Consult [execution configuration](execution.md)
for provider limits and runtime artifacts. A completed prediction has
`instance_id`, `decision` (`patch` or `abstain`), and a nonempty unified diff for
`patch`. An abstention has no patch or an empty patch. An execution failure has
`status` (`error`, `timeout`, or `invalid`) and `error`, with no decision or patch.
Supply primary agent traces under `<instance_id>/` in the trace root. Custom
harnesses must preserve the same input and access boundaries; schema compliance
alone does not establish comparable execution.

## 3. Validate, score, and report

```sh
dogbench validate runs/development/predictions.jsonl \
  --items data/development/items.jsonl --traces runs/development/traces \
  --json > runs/development/validation.json
dogbench score runs/development/predictions.jsonl \
  --outcomes data/development/outcomes.jsonl --output scores/development
dogbench report scores/development/scores.json \
  --outcomes data/development/outcomes.jsonl \
  --output scores/development/report.json --json
```

Validation checks prediction shape, coverage, source attestations, and
contamination evidence. Require a zero exit status and `valid: true` and
`canonical: true` in the JSON result. A nonzero validation exit requires review;
a generated score alone does not make that submission valid.
Do not weaken validation with `--format-only`, `--offline`, or
`--allow-missing-traces` for a canonical submission. Scoring does not substitute
for validation. Saved judgments can be replayed with `score --judgments`;
disclose their provenance and retain patch/rubric hash bindings.

The report JSON stores metrics under `summary`:

| Report field | Meaning |
| --- | --- |
| `delivered_patch_quality` | D: mean quality across all patch-needed items, including failed deliveries |
| `no_op_recall` | N: percentage of abstention items correctly left unchanged |
| `composite_score` | Harmonic mean of D and N for complete coverage of both classes |
| `items`, `expected_items`, `complete`, `missing_ids` | Coverage of the declared outcome population |
| `execution_failures` | Number of predictions whose status is not completed |
| `p0_clean_delivery_percent` | Percentage of patch-needed items with a correct decision and a P0-clean patch |

Contamination findings are in `runs/development/validation.json`, not the report
summary. Include both files. `score` also writes `scores.json`, `report.json`,
and per-item judge artifacts in its output directory.
A partial run or a sample containing only one task class has no composite.
Scores use equal criterion weights and a ceiling of 60 for an active P0 failure.
Disclose changes to the judge, rubric, or scoring rules.

## 4. Prepare the evidence bundle

Keep the run directory, its `predictions.jsonl`, `run.json`, task artifacts,
primary `traces/`, validation output, and scoring output together. Also retain
prepared-input attestations and judge decisions on the trusted controller.
These records must remain outside the agent sandbox.

Add a `README.md` describing:

| Detail | What to include |
| --- | --- |
| Submitter | Organization and a GitHub contact |
| Dataset | Release, split, revision, source-file SHA-256 hashes, and task count |
| Agent | Full model/provider ID, harness version, image digest, prompts, tools, and limits |
| Scorer | Runner commit, judge model/settings, prompt and rubric identities |
| Attempts | Attempts per task, retries, selection policy, and human intervention |
| Access | Allowed network destinations, cloud restrictions, and prior benchmark exposure |
| Results | D, N, composite, coverage, failures, and contamination findings |
| Exceptions | Missing traces, redactions, custom behavior, and protocol deviations |
| Reproduction | Exact commands, configuration, and setup instructions |

Host the bundle in a stable downloadable archive or repository and record its
SHA-256 checksum. Keep large traces out of your documentation PR. Check for
credentials before sharing; document redactions without removing behavior
needed for review. Preserve the original bundle privately. Redacting hashed
files changes their hashes: include a redaction record and checksums for the
shared copy, and explain resulting verification differences.

Never publish held-out task inputs or evaluator material. Arrange access to
private evidence with maintainers before uploading it.

## 5. Submit for review

Open a GitHub issue in this repository with the title
`Submission: <agent/model> — <development results or held-out evaluation>`.
Use the following outline:

```markdown
## Agent
- Agent/model and organization:
- Model, code, or product link:
- GitHub contact:

## Evaluation
- Requested path: development results / held-out evaluation
- Dataset release, split, revision, and task count:
- Runner commit and judge configuration:
- Delivered patch quality / correct abstention / composite:
- Failures and contamination findings:

## Evidence
- Bundle download link and SHA-256:
- Reproduction commands and agent configuration:
- Trace availability and redactions:
- Protocol changes, retries, and prior benchmark exposure:

## Held-out integration, if requested
- How maintainers can run or access the agent:
- Required credentials and provider endpoints (names only; no secrets):
- Execution limits and any cloud restrictions:
```

For a held-out request without results, mark the result fields as pending.
Do not put API keys or private contact information in the issue. If an adapter
change is needed, submit it in a separate pull request and link it from the issue.

Maintainers review completeness, reproducibility, contamination evidence,
scoring compatibility, and integration requirements. They may request changes
or a rerun. Accepted held-out results are added to the website manually after
verification. An issue or merged adapter PR does not automatically publish scores.

## Current tooling

This package provides `prepare`, `verify`, `run`, `validate`, `score`, and
`report`. It has no automatic submission or publication command. GitHub issues
and evidence bundles are the submission mechanism; maintainers review the
results before adding them to the leaderboard.
