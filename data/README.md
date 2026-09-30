# Public development artifacts

The full [development-v18 dataset](https://huggingface.co/datasets/promptless-research/dogbench-dev/tree/f8e5dd2040414810cc79e8f77167c84a12d76932) contains 175 tasks: 123 documentation updates and 52 abstentions. Full task inputs and rubrics stay on Hugging Face. `release-reference.json` pins that revision, both file hashes, and its public development membership.

This repository includes:

- `sample-5/`: five unchanged development task inputs, their canonical outcomes and rubrics, and upstream license references. All five require patches; the sample does not demonstrate abstention or produce a headline composite score.
- `../historical-trajectories/`: 735 saved runs for 105 development items across seven agent categories. These are historical examples, not reruns of the current public software.

The 117 held-out evaluation tasks and their references, rubrics, and item-level scores are not distributed here. The bundled examples do not establish a full development score or a held-out score. No new model evaluations were run to construct these artifacts.

Verify the bundled records without network or model access from the repository root:

```sh
python historical-trajectories/verify.py
python -m unittest discover -s tests -p test_release_artifacts.py
```

Use the root README for the supported task preparation and scoring commands. Dataset terms are in [DATASET_LICENSE.md](../DATASET_LICENSE.md), with third-party rights described in [DATA_LICENSE.md](../DATA_LICENSE.md).
