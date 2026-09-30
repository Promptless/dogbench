"""Export a reviewed working tree without copying any Git history or local artifacts."""
from pathlib import Path
import argparse
import shutil
from release_guard import BLOCKED_PARTS, BLOCKED_SUFFIXES, ReleaseError, scan

FILES = {'README.md','LICENSE','NOTICE','DATASET_CARD.md','DATASET_LICENSE.md','DATA_LICENSE.md','CITATION.cff','MANIFEST.in','pyproject.toml','.gitignore'}
DIRECTORIES = {'.github','docs','dogbench','tests','tools','data','historical-trajectories'}

def export(source: Path, destination: Path) -> int:
    if destination.exists():
        raise ReleaseError('Export destination must be new')
    destination.mkdir(parents=True)
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if rel.parts[0] not in FILES | DIRECTORIES:
            continue
        if any(part in BLOCKED_PARTS or part.endswith('.egg-info') for part in rel.parts):
            continue
        if path.suffix in BLOCKED_SUFFIXES or path.name == '.DS_Store' or path.name.startswith('.env'):
            continue
        if path.is_symlink():
            raise ReleaseError(f'Symlink in export input: {rel}')
        if path.is_file():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return scan(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(f'Exported and checked {export(args.source.resolve(), args.destination.resolve())} files')
