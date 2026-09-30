# Five development samples

Revision `development-v18-sample-5-r2` contains five items from the 175-item
`dogbench-development-v18` release (seed `2026092804`). These are hand-selected
workflow examples, not a representative score or a held-out evaluation.
Selection did not use agent scores. All five require patches; this sample does not
demonstrate abstention behavior and includes three Celery tasks.

| Item | Expected decision | Task |
| --- | --- | --- |
| celery-celery-pr10174 | patch | Document broker-pool acquisition timeouts |
| celery-celery-pr10206 | patch | Document individual request time-limit attributes |
| bentoml-bentoml-pr5518 | patch | Explain early inclusion of dependency-install files |
| celery-celery-pr9879 | patch | Document Redis result-backend credential providers |
| tiangolo-fastapi-pr14681 | patch | Correct cancellation in the streaming example |

Source identities and hashes are recorded in `provenance.jsonl` and
`selection.json`. Public records cannot establish absence of private prior
exposure. Some PRs include automated reviews; human-only origins are not asserted.

`items.jsonl` preserves exact selected source lines; `rubrics.jsonl` preserves the
released rubric text unchanged. `selection.json` records source hashes.
`labels.jsonl` and `evaluations.jsonl` retain the public package’s adapter format.
This CLI uses `outcomes.jsonl` for scoring. Historical reference patches are not bundled. The controller keeps labels, rubrics, and provenance
outside agent workspaces.

From the repository root:

```sh
python -m pip install -e '.[test,cloud]'
dogbench prepare data/sample-5/items.jsonl --output workspaces/sample-r2
dogbench verify workspaces/sample-r2 --json
```

After configuring an agent and Linux isolation, run and score the samples:

```sh
dogbench run data/sample-5/items.jsonl --prepared workspaces/sample-r2 \
  --agent codex --model MODEL_ID --output runs/sample-r2
dogbench validate runs/sample-r2/predictions.jsonl \
  --items data/sample-5/items.jsonl --traces runs/sample-r2/traces --json
dogbench score runs/sample-r2/predictions.jsonl \
  --outcomes data/sample-5/outcomes.jsonl --output scores/sample-r2
dogbench report scores/sample-r2/scores.json --json
```

See the root README and [adapter setup](../../docs/adapters.md) for credentials
and supported agents. Source preparation downloads pinned Git snapshots and
requires network access. Use a new workspace directory when changing samples.
All five tasks require patches, so the report has no headline composite score.

`outcomes.jsonl` preserves the matching records from the pinned Hugging Face
release, including exact rubric text and hashes, for this CLI's `score` command.
The sample manifest binds all bundled files, including this adapted README.

The full development dataset is on
[Hugging Face](https://huggingface.co/datasets/promptless-research/dogbench-dev).
Dataset terms are in `DATASET_LICENSE.md`. Pinned upstream license links and
file-level qualifications are in `source_licenses.json`.
Upstream terms remain separate from the dataset license.

Revision r2 replaces the original five-item sample; use fresh prepared workspaces.
