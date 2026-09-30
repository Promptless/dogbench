"""Exercise the publication boundary with nested and compressed private-content fixtures."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from release_guard import ReleaseError, check_bytes, scan
from export_public import export

PRIVATE = '/' + 'Users' + '/privacy-fixture/task.txt'

@pytest.mark.parametrize('value', [
    {'inputs': {'versions': {'local': PRIVATE}}},
    {'agents': [{'verdicts': {'C1': {'reason': PRIVATE}}}]},
    {'nested': json.dumps({'message': PRIVATE})},
    {'nested': json.dumps({'message': PRIVATE}) + '\n' + json.dumps({'message': 'safe'})},
])
def test_private_paths_in_nested_values(value):
    with pytest.raises(ReleaseError, match='personal path'):
        check_bytes(json.dumps(value).encode(), 'item.json')

def test_gzip_and_zip_are_scanned_recursively():
    payload = gzip.compress(json.dumps({'prompt': PRIVATE}).encode())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as archive:
        archive.writestr('item.json.gz', payload)
    with pytest.raises(ReleaseError, match='personal path'):
        check_bytes(buf.getvalue(), 'release.zip')

def test_tar_is_scanned():
    buf = io.BytesIO()
    payload = json.dumps({'text': PRIVATE}).encode()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        entry = tarfile.TarInfo('package/item.json'); entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    with pytest.raises(ReleaseError, match='personal path'):
        check_bytes(buf.getvalue(), 'release.tar.gz')

def test_archive_symlinks_are_rejected():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as archive:
        entry = tarfile.TarInfo('item'); entry.type = tarfile.SYMTYPE; entry.linkname = '/private'
        archive.addfile(entry)
    with pytest.raises(ReleaseError, match='Nonregular'):
        check_bytes(buf.getvalue(), 'release.tar')

def test_credential_error_does_not_print_secret():
    secret = 'gh' + 'p_' + 'a' * 36
    with pytest.raises(ReleaseError) as exc:
        check_bytes(json.dumps({'explanation': secret}).encode(), 'item.json')
    assert secret not in str(exc.value)

def test_clean_export_drops_history_browser_logs_caches(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    keep = ['README.md', 'dogbench/example.py', 'historical-trajectories/model/item.json.gz']
    for name in keep:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(b'{"trace":"public example"}') if p.suffix=='.gz' else b'public\n')
    for name in ['.git/config', '.playwright-mcp/browser.yml', 'outputs/audit.json', 'dogbench/__pycache__/x.pyc', 'build/pkg.py', 'logs/run.txt', 'dogbench/debug.log', '.env']:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(PRIVATE)
    destination = tmp_path/'release'
    assert export(source, destination) == 3
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(keep)

def test_unknown_top_level_files_not_exported(tmp_path):
    source=tmp_path/'source'; source.mkdir(); (source/'private-notes.txt').write_text(PRIVATE)
    assert export(source,tmp_path/'release') == 0

def test_public_prompt_redaction_hashes():
    from dogbench.research_inputs import input_registry
    import hashlib
    for record in input_registry()['items'].values():
        assert hashlib.sha256(record['prompt'].encode()).hexdigest() == record['prompt_sha256']
        if record.get('public_redactions'):
            assert '~/' in record['prompt']
