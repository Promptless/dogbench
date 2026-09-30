# DogBench

> **Work in progress:** This repository is actively being developed. Feedback and contributions are welcome.

DogBench evaluates whether agents can maintain expert-quality user-facing
software documentation. Each task starts from a frozen repository snapshot and
a code change or reported documentation gap. The agent writes a documentation
patch or explicitly returns `NO_DOC_CHANGES_NEEDED`.

The benchmark has 292 tasks. The public development release contains 175 tasks
(123 documentation changes and 52 abstentions); the other 117 tasks remain
held out. This repository contains the execution and scoring implementation,
five development samples, and a historical archive of 735 sanitized agent
trajectories. Full development inputs and frozen rubrics are distributed on
[Hugging Face](https://huggingface.co/datasets/promptless-research/dogbench-dev).
See [data/README.md](data/README.md) for immutable revisions, hashes, and artifact
coverage.

See the [submission guide](docs/leaderboard-submission.md) to share development
results or request a held-out evaluation. The five bundled examples all require
patches and do not produce a headline composite score.

## Install

Python 3.11 or newer and Git are required. Agent execution initially supports
Linux. The preparation, validation, saved-judgment scoring, and reporting
commands also work without an agent provider.

```sh
python -m pip install -e '.[test,cloud]'
dogbench --help
```

Local agents require configured CLI authentication and the isolated Linux
container/proxy setup. Cloud agents require explicit accounts and sanitized
mirror configuration. See [adapter setup](docs/adapters.md) and
[execution configuration](docs/execution.md). Promptless additionally requires
an optional private backend; a missing backend is an explicit configuration
error.

## Prepare and verify

```sh
dogbench prepare data/sample-5/items.jsonl --output workspaces
dogbench verify workspaces --json
```

Preparation verifies upstream coordinates, filters task context, creates
anonymous writable documentation and read-only code repositories, removes
source history, and seals the resulting inputs. Controller attestations and
expected outcomes must stay outside the agent's workspace. `prepare --offline`
is for synthetic fixtures or previously attested sources; its output records
the weakened check.

The 175 registered development items use frozen research prompts and input
bindings. Preparation and verification check exact prompt bytes, item identity,
documentation/code base trees, and code-change paths and blob hashes. Local
execution uses that verified prompt directly. Prepare fresh workspaces after
upgrading; older development workspaces without this binding are rejected.
Custom items use the existing prompt renderer. Cloud adapters retain their
provider-specific prompt construction.

## Run and validate

After configuring an agent and Linux isolation:

```sh
dogbench run data/sample-5/items.jsonl --prepared workspaces \
  --agent codex --model MODEL_ID --output runs/codex
dogbench validate runs/codex/predictions.jsonl \
  --items data/sample-5/items.jsonl --traces runs/codex/traces --json
```

`run` invokes the selected provider and may incur charges. It accepts `claude`,
`codex`, `opencode`, `devin`, `mintlify`, and `promptless`, along with `--model`,
`--timeout`, `--budget-usd`, and an optional `--config` JSON file. The budget
setting is passed to adapters that support it; it is not a universal spending
cap. See the execution guide for per-provider configuration and trace layout.

Validation checks prediction shape and task coverage, re-attests source inputs,
audits traces, and checks for copied human or cross-agent patches. Missing
traces fail closed. `--format-only`, `--offline`, and
`--allow-missing-traces` explicitly weaken validation and are recorded in the
result. A valid JSON format alone does not establish a valid benchmark run.

## Score and report

Score saved predictions using the frozen outcomes for the same release:

```sh
dogbench score runs/codex/predictions.jsonl \
  --outcomes data/sample-5/outcomes.jsonl --output scores/codex
dogbench report scores/codex/scores.json --json
```

The first command invokes the original Terra judge through Codex. Add
`--judgments saved-judgments.json` to recompute from existing judgments without
model calls. A saved bundle may map each instance ID to
`{"judgment": {...}, "patch_sha256": "...", "rubric_sha256": "..."}`;
provided hashes are checked against the actual patch and frozen rubric.
Raw historical criterion judgments can also be replayed, but do not by
themselves prove which inputs produced them. Scoring does not substitute for
the preceding contamination validation.

The judge receives the candidate patch and frozen rubric. Requirements and
triggered conditional criteria each carry weight 1; violated deduction-only
criteria each subtract 1. Scores are floored at zero, and any active P0 failure
caps quality at 60. Delivered quality averages across all patch-needed tasks,
including zeroes for wrong decisions and operational failures. Abstention
recall measures correct abstention on no-update tasks. The headline composite
is their harmonic mean, `2QA / (Q + A)`.

Reports require explicit complete task membership and both task classes for a
headline composite. Partial or single-class inputs retain diagnostics and
report the composite as undefined. Historical saved judgments have limited
coverage and must not be presented as a complete rerun of the current release.

## Prediction format

Each prediction is one JSONL record keyed by `instance_id`:

```json
{"instance_id":"example-task","status":"completed","decision":"patch","patch":"diff --git a/docs/guide.md b/docs/guide.md\n..."}
{"instance_id":"another-task","status":"completed","decision":"abstain"}
{"instance_id":"failed-task","status":"timeout","error":"Agent exceeded the item timeout."}
```

Operational failures have no `decision` or `patch`. Schemas for items, outcomes,
and predictions are in [dogbench/schemas](dogbench/schemas). Empty historical
documentation paths do not imply abstention; labels are provided separately
in the outcomes file.

## Scope and limitations

This extraction retains original benchmark behavior, including known defects.
Read [known issues](docs/known-issues.md) before interpreting failures or
replaying historical artifacts. Dataset construction, rubric generation and
calibration, reviewer records, deployment state, and the private Promptless
backend are outside this package. The implementation does not regenerate task
context or rubrics.

The [dataset card](DATASET_CARD.md) describes the public data and limitations.
Repository layout:

- `dogbench/`: preparation, execution, adapters, validation, judging, reporting.
- `data/`: five samples, release bindings, and verified saved development results.
- `historical-trajectories/`: 735 sanitized historical traces and their verifier.
- `tests/`: deterministic tests using synthetic tasks and mocked providers.
- `docs/`: runtime configuration and known issues.

```sh
python -m pytest
python historical-trajectories/verify.py
```

The Python distributions contain software and configuration assets. Dataset
samples and historical archives remain in the Git tree and are not bundled
into the wheel. No live provider jobs are run by the test suite.

## License and citation

Software is licensed under [Apache 2.0](LICENSE). Original dataset material is
subject to [separate dataset terms](DATASET_LICENSE.md); upstream repository
content retains its original terms and notices. See [NOTICE](NOTICE) and
[CITATION.cff](CITATION.cff).
