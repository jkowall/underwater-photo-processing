"""Copy only the skill payload into a personal Codex skill directory."""
from pathlib import Path
import argparse
import os
import shutil

PAYLOAD=('SKILL.md','agents','references','scripts/process_batch.py','scripts/underwater_pipeline.py')

def install(source,target,update=False):
    existing=target/'SKILL.md'
    if target.exists() and not update:
        raise FileExistsError('Skill already exists; use --update after reviewing your local changes')
    if target.exists() and not existing.is_file():
        raise ValueError('Destination exists without a skill entrypoint; refusing to overwrite')
    target.mkdir(parents=True,exist_ok=True)
    for relative in PAYLOAD:
        src=source/relative;dst=target/relative
        if src.is_dir():
            shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def main():
    parser=argparse.ArgumentParser()
    default=Path(os.environ.get('CODEX_HOME',Path.home()/'.codex'))/'skills'/'underwater-photo-processing'
    parser.add_argument('--destination',type=Path,default=default)
    parser.add_argument('--update',action='store_true',help='Replace known skill files while preserving unrelated files')
    args=parser.parse_args()
    try:install(Path(__file__).resolve().parent.parent,args.destination,args.update)
    except (OSError,ValueError) as error:parser.error(str(error))
    print('Installed:',args.destination)

if __name__=='__main__':main()
