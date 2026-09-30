# Historical development trajectories

This archive contains **735 saved trajectories: seven non-cloud agent
categories for 105 development items** from the 175-item development split
(`split4-seed2026092804-v1`). Every record has a nonempty saved event stream.

| Directory | Records |
| --- | ---: |
| `claude-opus-4.8/` | 105 |
| `claude-sonnet-4.6/` | 105 |
| `glm-5.2/` | 105 |
| `gpt-5.5/` | 105 |
| `gpt-5.6-sol/` | 105 |
| `kimi-k2.7-code/` | 105 |
| `qwen3.8-max/` | 105 |

Each `<model>/<item>.json.gz` file contains the saved agent event stream,
task prompt, supporting context, final response, candidate documentation patch,
and basic run metadata. `index.json` lists the item IDs and file hashes;
its `development_items_sha256` hashes this 105-item subset.

The public copies redact credential patterns and personal home paths. Private
source-file provenance is omitted. These are historical runs; their presence
does not imply that their patches passed evaluation or that replay will produce
the same output. They have not been rerun with this public software, and their saved prompts may differ from the current development task framing. See the [data license](../DATA_LICENSE.md) for reuse terms.

The item IDs are a subset of the pinned public development membership in `data/release-reference.json`.

Verify completeness and file hashes:

```sh
python historical-trajectories/verify.py
```
