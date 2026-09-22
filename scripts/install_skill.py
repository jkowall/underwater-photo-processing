"""Copy the skill payload into a personal agent skills directory.

Defaults to Codex's skills path when CODEX_HOME / ~/.codex exists; otherwise
use --destination (e.g. ~/.cursor/skills/underwater-photo-processing).
The skill itself is plain SKILL.md + helpers — not Codex-specific.
"""
from pathlib import Path
import argparse
import os
import shutil

PAYLOAD = (
    'SKILL.md',
    'agents',
    'references',
    'scripts/process_batch.py',
    'scripts/underwater_pipeline.py',
)


def default_destination() -> Path:
    codex_home = os.environ.get('CODEX_HOME')
    if codex_home:
        return Path(codex_home) / 'skills' / 'underwater-photo-processing'
    codex_skills = Path.home() / '.codex' / 'skills'
    if codex_skills.is_dir() or (Path.home() / '.codex').is_dir():
        return codex_skills / 'underwater-photo-processing'
    cursor_skills = Path.home() / '.cursor' / 'skills'
    return cursor_skills / 'underwater-photo-processing'


def install(source, target, update=False):
    existing = target / 'SKILL.md'
    if target.exists() and not update:
        raise FileExistsError(
            'Skill already exists; use --update after reviewing your local changes'
        )
    if target.exists() and not existing.is_file():
        raise ValueError(
            'Destination exists without a skill entrypoint; refusing to overwrite'
        )
    target.mkdir(parents=True, exist_ok=True)
    for relative in PAYLOAD:
        src = source / relative
        dst = target / relative
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description='Install underwater-photo-processing skill for any SKILL.md host'
    )
    parser.add_argument('--destination', type=Path, default=default_destination())
    parser.add_argument(
        '--update',
        action='store_true',
        help='Replace known skill files while preserving unrelated files',
    )
    args = parser.parse_args()
    try:
        install(Path(__file__).resolve().parent.parent, args.destination, args.update)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Installed:', args.destination)


if __name__ == '__main__':
    main()
