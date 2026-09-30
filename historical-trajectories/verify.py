"""Verify the published historical development trajectory archive."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    index = json.loads((ROOT / 'index.json').read_text(encoding='utf-8'))
    assert index['schema_version'] == 'dogbench-historical-trajectories-index-v2'
    assert set(index) == {
        'schema_version', 'split', 'split_id', 'development_items_sha256',
        'item_ids', 'models', 'records',
    }
    assert index['split'] == 'development'
    items = index['item_ids']
    models = index['models']
    assert len(items) == len(set(items)) == 105
    assert len(models) == len(set(models)) == 7
    assert sha256(('\n'.join(items) + '\n').encode()) == index['development_items_sha256']

    records = index['records']
    assert len(records) == 735
    assert {(row['instance_id'], row['model_key']) for row in records} == {
        (item, model) for item in items for model in models
    }
    assert {p.relative_to(ROOT).as_posix() for p in ROOT.glob('*/*.json.gz')} == {
        row['path'] for row in records
    }

    for row in records:
        assert set(row) == {'instance_id', 'model_key', 'path', 'sha256', 'trace_available'}
        expected = f"{row['model_key']}/{row['instance_id']}.json.gz"
        assert row['path'] == expected
        data = (ROOT / expected).read_bytes()
        assert sha256(data) == row['sha256'], expected
        run = json.loads(gzip.decompress(data))
        assert run['schema_version'] == 'dogbench-historical-trajectory-v2'
        assert set(run) == {
            'schema_version', 'instance_id', 'model_key', 'agent', 'model',
            'generated_at', 'ok', 'verdict', 'elapsed_s', 'trace_format',
            'prompt', 'supporting_context', 'trace', 'final_response', 'patch',
        }
        assert (run['instance_id'], run['model_key']) == (
            row['instance_id'], row['model_key']
        )
        assert bool(run['trace']) == row['trace_available']
        assert isinstance(run['prompt'], str)
        assert isinstance(run['patch'], str)
        assert row['trace_available'], expected

    print(f"Verified {len(records)} records: {len(items)} items × {len(models)} models")
    print(f"Saved event streams: {sum(row['trace_available'] for row in records)}")


if __name__ == '__main__':
    main()
