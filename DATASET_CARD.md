---
pretty_name: DogBench Development
language:
- en
license: cc-by-nc-nd-4.0
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- benchmark
- documentation
- software-engineering
- agents
configs:
- config_name: items
  default: true
  data_files:
  - split: development
    path: items.jsonl
- config_name: outcomes
  data_files:
  - split: development
    path: outcomes.jsonl
---

# DogBench development set

DogBench evaluates whether AI agents can maintain user-facing software documentation. Each task provides a code change or documentation request and a repository snapshot from before the change. The agent must write a documentation patch or abstain when no update is needed.

This release contains **175 development items: 123 documentation tasks and 52 abstention tasks**. The 117 evaluation items are held out.

## Load the dataset

```python
from datasets import load_dataset

repo = "promptless-research/dogbench-dev"
revision = "f8e5dd2040414810cc79e8f77167c84a12d76932"
items = load_dataset(repo, "items", split="development", revision=revision)
outcomes = load_dataset(repo, "outcomes", split="development", revision=revision)
```

The Hugging Face dataset contains two files with 175 rows each, joined by
`instance_id`. This software repository bundles only five samples; see
[the release reference](data/release-reference.json) for immutable file hashes.

| File | Contents |
| --- | --- |
| [items.jsonl](https://huggingface.co/datasets/promptless-research/dogbench-dev/resolve/f8e5dd2040414810cc79e8f77167c84a12d76932/items.jsonl) | Task context and repository coordinates |
| [outcomes.jsonl](https://huggingface.co/datasets/promptless-research/dogbench-dev/resolve/f8e5dd2040414810cc79e8f77167c84a12d76932/outcomes.jsonl) | Expected decisions and full rubric text |

## Fields

`items.jsonl` contains:

- `instance_id`: item identifier.
- `source_url`: original pull request or issue.
- `context`: sanitized task description and supporting context.
- `docs`: documentation repository URL, base/head commits, and historical reference paths.
- `code`: code-change URL, repository, commits, and allowed source paths; null for documentation-request tasks.

`outcomes.jsonl` contains `expected_outcome` (`patch` or `abstain`), `rubric_markdown`, its `rubric_sha256`, and rubric status and notes. Abstention items have no rubric. Read rubrics directly from `rubric_markdown`; no separate rubric files are needed.

Empty `docs.paths` and a null `docs.head_sha` mean no historical documentation diff is supplied. Use `expected_outcome` to determine whether an item expects abstention.

## Running agents

With DogBench installed, prepare the downloaded inputs:

```sh
dogbench prepare items.jsonl --output workspaces
```

Keep source identities, historical documentation, rubrics, and outcomes outside the agent sandbox. Use identity-masked workspaces, mount code read-only, block network access except to the model provider, and retain tool traces for contamination checks. Operational failures do not count as abstentions.

## Scoring

Patches are judged against task-specific binary criteria, rather than textual similarity to a historical patch. Quality is scored from 0–100; a P0 failure caps it at 60. The composite score is `2DN / (D + N)`, where `D` is mean delivered quality across documentation tasks and `N` is abstention recall, both on a 0–100 scale. Missed, empty, or invalid patches receive zero quality. Report failures and P0-failure rates alongside the composite.

## Limitations

- Public source material may have appeared in model training data.
- Automated rubrics and LLM judging can disagree with expert review. Some rubric cross-references name removed criteria; score only explicitly defined criterion blocks.
- Some abstention labels rely on the absence of a documentation update within an audit window.
- Reported runs use one sample per system and item; not every documented workflow is executed. Scores evaluate the model and harness together.

## License

Original DogBench dataset material is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), to the extent contributors hold the necessary rights. Upstream content retains its original licenses and notices.
