# Run frozen prepared inputs

The runner consumes the public item contract and the `prepare` output. It does
not create task context, rubrics, outcomes, or reference answers. Use a fresh
output directory for each run and keep it outside the prepared-input directory.

```sh
dogbench run items.jsonl --prepared prepared/ --output runs/codex/ \
  --agent codex --model gpt-5.5 --timeout 1800 --json

dogbench validate runs/codex/predictions.jsonl --items items.jsonl \
  --traces runs/codex/traces --json
```

The real local run requires Linux and configured Docker or Podman isolation.
See [adapter configuration](adapters.md) for the explicit container image,
network, proxy and credential files. A local run has no GitHub credential and
makes no pull request. Its documentation checkout has no upstream remote;
prepared code is mounted as read-only `/task/code_repo` when present. The
controller keeps labels, reference candidates and run logs outside that checkout.

Use `--agent claude`, `codex` or `opencode` for local lanes and an explicit
`--model` value. `--budget-usd` applies to the Claude adapter. The default is the
original adapter's $5 limit. `--timeout` sets the per-item wall-clock cap.
Adapter-specific API-call retries and provider settings remain in the extracted
adapters. The runner dispatches one attempt per item and does not select among
repeated completed attempts.

Claude now enables `Bash,Read,Edit,Write,Glob,Grep` by default. `Read` lets
Claude satisfy `Edit`'s requirement to read an existing file before editing it.
This deliberately changes the historical tool configuration, which excluded
`Read`. To replay that historical configuration, explicitly set
`DOCBENCH_CLAUDE_CODE_TOOLS=Bash,Edit,Write,Glob,Grep`. The operational prompt
still prohibits opening persisted Claude tool-result files.

A usable local patch can survive process termination. The exact original order
is preserved: validate a configured Claude model contract, export the sealed
workspace patch, attempt recovery, derive the verdict from that patch, then
apply the trajectory and contamination gates. A failed model contract cannot
be rescued by a patch. A later contamination failure overrides patch recovery.
An empty successful local diff becomes `NO_DOC_CHANGES_NEEDED`, regardless of
contradictory final prose. A Claude rate-limit result stops without writing a
candidate for that item.

## Frozen development inputs

For registered development items, local execution reads the exact prompt bytes
sealed in `TASK.md`; it does not reconstruct framing from the item context.
The shipped `dogbench/assets/research_inputs.json` registry binds all 175 items
to their frozen prompts, source snapshots, and code-change blobs. Preparation,
verification, and local dispatch reject changed registered items or input drift.
Each registered run records `logs/research_input_binding.json` with registry,
item, prompt, and research logical-input hashes. This record describes the
frozen local input; cloud adapters still construct their transport-specific prompts.

Existing development workspaces must be prepared again in a fresh directory.
Unregistered custom items retain the existing renderer and do not receive a
frozen research binding. These input checks do not change model settings,
provider behavior, judge retries, or the contamination policy.

## Controller configuration

`--config config.json` accepts a JSON object. Credential fields contain
environment-variable **names**, never secret values.

| Key | Meaning |
| --- | --- |
| `candidate_key` | Original research lane key; defaults to the agent name. Use the exact original key when comparing a run with saved lane artifacts. |
| `candidate_reference_root` | Optional controller-only directory containing `<instance_id>/human.json` or `human.diff`, and original canonical lane JSON files for the deterministic post-run overlap gate. |
| `protected_host_paths` | Additional controller path strings the original trace scanner must flag; the research deployment's private host paths are not public defaults. |
| `items` | Per-item cloud configuration, keyed by `instance_id`; its values override shared cloud fields. |

The runtime overlap gate evaluates only references actually supplied under
`candidate_reference_root`. An absent human reference is recorded as
`human_similarity.evaluated=false`, preserving the original gate behavior. It
is not evidence of clean overlap. The separate `validate` command reconstructs
public human references and audits exported predictions; run it before
interpreting a result as validated.

`run.json` records counts, completion, any stop reason and the original outcome
exit-code precedence: contamination **3**, rate limiting or pending broker
output **2**, other failures **1**, success **0**. An incomplete run is not a
complete benchmark score. `predictions.jsonl` follows the existing public
schema; failure records contain an error rather than a patch or decision. Full
candidate patches and failure details remain in `items/<id>/candidate.json`.

`traces/<id>/` contains only the original primary agent trace artifacts for the
public validation command. `items/<id>/logs/` also contains controller audits and
must not be passed as a trace root: controller metadata legitimately contains
protected source identifiers.

## Cloud lanes

Devin, Mintlify and Promptless require an explicitly configured, pre-provisioned
private mirror per item. The run command does not create an account, select an
organization, choose a hosted project, or provision a mirror automatically.
The retained `dogbench.cloud_mirrors.Mirror` helpers perform original managed
transport setup when invoked explicitly with caller-owned repositories and
credentials. Follow the adapter configuration for each service.

A cloud run can write documentation and pull requests to the configured mirror.
The trusted controller retains the original constrained output repair: retarget
a PR whose first commit is based on the exact frozen base but aimed at the wrong
branch; for Mintlify, recover the original safe protected-base-write topology by
copying the unchanged agent commit, restoring the base, closing the malformed
PR and opening a replacement. Other invalid source histories remain failures.

Example cloud config (replace every placeholder with your own setup):

```json
{
  "github_token_env": "BENCHMARK_GITHUB_TOKEN",
  "devin_api_key_env": "BENCHMARK_DEVIN_KEY",
  "items": {
    "example-project-pr123": {
      "mirror_repo": "YOUR-ORG/YOUR-PRIVATE-MIRROR",
      "clone_dir": "/absolute/path/to/your/mirror-clone",
      "base_branch": "frozen-base",
      "base_sha": "0000000000000000000000000000000000000000",
      "docs_work_branch": "candidate-work",
      "transport_mode": "exact",
      "code_pr_url": "https://github.com/YOUR-ORG/YOUR-PRIVATE-MIRROR/pull/1",
      "code_pr_path_prefix": ""
    }
  }
}
```

```sh
dogbench run items.jsonl --prepared prepared/ --output runs/devin/ \
  --agent devin --model devin --config config.json --json
```

Each cloud item requires `mirror_repo`, `clone_dir`, `base_branch`, `base_sha`,
`docs_work_branch` and `github_token_env`. The mirror must be private and differ
from every upstream repository. Its configured local clone must be clean and
pinned, and its remote base branch must match. For code tasks, `code_pr_url`
must identify an open PR in that mirror, and its changed paths and blob contents
must correspond to the prepared code change. Docs-only items omit it.

`code_pr_path_prefix` is either empty (same-repository code) or `_code_repo/`
(separate code repository). `transport_mode=exact` requires the frozen tree
plus that optional exact code-base overlay. To use the original managed
transport, set `transport_mode=managed` and provide explicit
`mirror_overlays_root` and `source_overlays_root` directory paths. Empty
directories are valid when there are no overlays. The verifier derives and
checks the original deterministic managed tree, including its sanitization and
configuration transformations; arbitrary unverified mirror changes are not
accepted.

Additional required service configuration:

| Agent | Fields |
| --- | --- |
| Devin | `devin_api_key_env` |
| Mintlify | `mintlify_token_env`, `project_id`, `project_mirror_confirmed: true` after configuring that project to use the named mirror |
| Promptless | `promptless_api_trigger_key_env`, `runtime_base_url`; optional `analysis_version`; explicit backend configuration described in the adapter guide |

The same dataset/model-independent prompts, `NO_DOC_CHANGES_NEEDED` token,
original provider polling and output extraction remain in use. Managed services
have their original access boundary and trace limitations; configuring a cloud
lane does not turn it into a local sealed-container evaluation.


### Public input privacy

The public input registry replaces personal home-directory prefixes in quoted
upstream diagnostics with `~`. Affected records declare `public_redactions`,
and their `prompt_sha256` binds the published, redacted prompt. Other frozen
source and item bindings remain unchanged. These records are not byte-identical
to the original research prompts.
