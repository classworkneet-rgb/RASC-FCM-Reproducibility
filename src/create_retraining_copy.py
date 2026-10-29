"""Create a fresh execution tree while retaining the frozen reference checkpoints."""
import argparse, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',required=True)
    a=p.parse_args();target=Path(a.destination).resolve()
    if target==ROOT or target.is_relative_to(ROOT):
        raise SystemExit('The destination must be outside the source package.')
    if target.exists():raise SystemExit('The destination already exists; no files were replaced.')
    shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('results','selector_checkpoints','__pycache__'))
    (target/'results').mkdir(exist_ok=True)
    print('Fresh execution tree:',target)
    print('Reference checkpoints are retained for independent comparison; training does not load them.')

if __name__=='__main__':main()
