"""Fail closed when a release contains private content or local development artifacts."""
from __future__ import annotations
import argparse
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile
import zlib

MAX_BYTES = 128 * 1024 * 1024
MAX_DEPTH = 8
PATTERNS = {
    'personal path': re.compile(r'/Users/[A-Za-z0-9_.-]+/|/home/(?:ec2-user|ubuntu|frances[^/\s]*)/|[A-Z]:\\Users\\[A-Za-z0-9_.-]+\\'),
    'internal host or artifact': re.compile(r'barely[-]awake[-]licorice|172\.31\.41\.240|trace\.promptless\.ai|promptless[_-]clean[_-]rerun|data/(?:cloud-runs|analysis)/'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|xox[baprs]-[A-Za-z0-9-]{25,})'),
}
BLOCKED_PARTS = {'.git', '.venv', '.playwright-mcp', '__pycache__', '.pytest_cache', 'node_modules', 'outputs', 'logs', 'runs', 'workspaces', 'build', 'dist', '.mypy_cache', '.ruff_cache', '.cache'}
BLOCKED_SUFFIXES = {'.pyc', '.pyo', '.log', '.pem', '.key', '.bak', '.orig'}

class ReleaseError(ValueError):
    pass


def check_name(name: str) -> None:
    path = PurePosixPath(name)
    for category, pattern in PATTERNS.items():
        if pattern.search(name):
            raise ReleaseError(f"{category} in release member name")
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ReleaseError(f'Unsafe release member: {name}')
    if any(p in BLOCKED_PARTS or p.endswith('.egg-info') for p in path.parts):
        # Distribution metadata is expected in source archives, never a checkout export.
        raise ReleaseError(f'Development artifact: {name}')
    if path.suffix in BLOCKED_SUFFIXES or path.name == '.DS_Store' or path.name.startswith('.env'):
        raise ReleaseError(f'Private or temporary file: {name}')


def check_text(text: str, name: str, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise ReleaseError(f'Nested content exceeds review limit: {name}')
    for category, pattern in PATTERNS.items():
        if pattern.search(text):
            # Never print the matching value: it might be a credential.
            raise ReleaseError(f'{category} in {name}')
    stripped = text.strip()
    if stripped.startswith(('{', '[', '"')):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            if '\n' in stripped:
                for line in stripped.splitlines():
                    if line.lstrip().startswith(('{', '[', '\"')):
                        check_text(line, name, depth + 1)
            return
        check_value(value, name, depth + 1)


def check_value(value, name: str, depth: int = 0) -> None:
    if isinstance(value, str):
        check_text(value, name, depth)
    elif isinstance(value, dict):
        for key, child in value.items():
            check_text(str(key), name, depth)
            check_value(child, name, depth)
    elif isinstance(value, list):
        for child in value:
            check_value(child, name, depth)


def bounded_read(stream) -> bytes:
    data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ReleaseError('Expanded release member exceeds review limit')
    return data


def check_bytes(data: bytes, name: str, depth: int = 0) -> None:
    if len(data) > MAX_BYTES or depth > MAX_DEPTH:
        raise ReleaseError(f'Release member exceeds review limit: {name}')
    if name.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
            for member in archive:
                # egg-info is legitimate distribution metadata; inspect its contents.
                check_name('/'.join(p for p in PurePosixPath(member.name).parts if not p.endswith('.egg-info')))
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ReleaseError(f'Nonregular archive member: {name}')
                if member.isfile():
                    check_bytes(bounded_read(archive.extractfile(member)), member.name, depth + 1)
    elif name.endswith(('.zip', '.whl')):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                check_name(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ReleaseError(f'Symlink in archive: {name}')
                if not member.is_dir():
                    with archive.open(member) as stream:
                        check_bytes(bounded_read(stream), member.filename, depth + 1)
    elif name.endswith('.gz'):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            check_bytes(bounded_read(stream), name[:-3], depth + 1)
    elif name.lower().endswith('.png'):
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            raise ReleaseError(f'Invalid PNG: {name}')
        position = 8
        while position < len(data):
            size = int.from_bytes(data[position:position + 4], 'big')
            kind = data[position + 4:position + 8]
            chunk = data[position + 8:position + 8 + size]
            if len(chunk) != size:
                raise ReleaseError(f'Truncated PNG: {name}')
            if kind in (b'tEXt', b'eXIf'):
                check_text(chunk.decode('utf-8', errors='replace'), name)
            elif kind == b'zTXt':
                keyword, payload = chunk.split(b'\0', 1)
                if payload[0] != 0:
                    raise ReleaseError(f'Unknown PNG compression: {name}')
                inflater = zlib.decompressobj()
                decoded = inflater.decompress(payload[1:], MAX_BYTES + 1)
                if len(decoded) > MAX_BYTES or not inflater.eof:
                    raise ReleaseError(f'PNG metadata exceeds review limit: {name}')
                check_text(decoded.decode('utf-8', errors='replace'), name)
            elif kind == b'iTXt':
                keyword, rest = chunk.split(b'\0', 1)
                compressed, method = rest[:2]
                language, translated, payload = rest[2:].split(b'\0', 2)
                check_text(translated.decode('utf-8', errors='replace'), name)
                if compressed:
                    if method != 0:
                        raise ReleaseError(f'Unknown PNG compression: {name}')
                    inflater = zlib.decompressobj()
                    payload = inflater.decompress(payload, MAX_BYTES + 1)
                    if len(payload) > MAX_BYTES or not inflater.eof:
                        raise ReleaseError(f'PNG metadata exceeds review limit: {name}')
                check_text(payload.decode('utf-8', errors='replace'), name)
            position += size + 12
    elif name.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.attachments:
            raise ReleaseError(f'Unreviewed PDF attachments: {name}')
        check_text(str(reader.metadata or {}), name)
        if reader.xmp_metadata:
            check_text(reader.xmp_metadata.stream.get_data().decode('utf-8'), name)
        for page in reader.pages:
            check_text(page.extract_text() or '', name)
            for annotation in page.get('/Annots', []):
                check_text(str(annotation.get_object()), name)
    else:
        if name.lower().endswith(('.jpg', '.jpeg')):
            check_text(data.decode('utf-8', errors='replace'), name)
            check_text(data.decode('utf-16le', errors='replace'), name)
            return
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ReleaseError(f'Unsupported binary release member: {name}') from exc
        check_text(text, name)
        if name.endswith('.jsonl'):
            for line in text.splitlines():
                if line.strip():
                    check_value(json.loads(line), name)


def scan(root: Path) -> int:
    if root.is_symlink():
        raise ReleaseError('Release root is a symlink')
    if root.is_file():
        check_bytes(root.read_bytes(), root.name)
        return 1
    count = 0
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        check_name(name)
        if path.is_symlink():
            raise ReleaseError(f'Symlink in release: {name}')
        if path.is_file():
            check_bytes(path.read_bytes(), name)
            count += 1
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f'Checked {scan(path)} files: {path.name}')
