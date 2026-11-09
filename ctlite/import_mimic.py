"""Convert an authorized copy of the exact MIMIC benchmark caches outside this repository."""
import argparse
from pathlib import Path
import torch
from .data import save_records, sha256
from .train import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Authorized directory containing train.pt, val.pt, test.pt')
    p.add_argument('--destination',type=Path,required=True,help='Private output directory outside the repository')
    p.add_argument('--trust-source',action='store_true',help='Acknowledge that original torch caches may contain pickle objects')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.destination.resolve().is_relative_to(root):
        p.error('Keep restricted MIMIC data outside this public repository')
    if not a.trust_source:
        p.error('Only load a trusted authorized cache; pass --trust-source after checking its provenance')
    manifest={}
    for split in ('train','val','test'):
        source=a.source/f'{split}.pt'
        target=a.destination/f'{split}.npz'
        if target.exists():
            p.error(f'Refusing to overwrite {target}')
        rows=torch.load(source,map_location='cpu',weights_only=False)
        save_records(rows,target)
        manifest[split]=dict(source_sha256=sha256(source),npz_sha256=sha256(target),records=len(rows))
    dump(a.destination/'manifest.json',manifest)
    print('Converted authorized caches. Keep this directory private.')


if __name__=='__main__':
    main()
