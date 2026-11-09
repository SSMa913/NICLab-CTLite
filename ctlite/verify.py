"""Verify release hashes, original run evidence, and optional checkpoint metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_split, evaluate, sha256
from .train import make_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate',action='store_true',help='Re-evaluate the 15 included public-data checkpoints')
    p.add_argument('--device',default='cuda')
    p.add_argument('--results',type=Path,help='Compare 20 newly trained runs with the paper reference')
    a=p.parse_args()
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    reference=json.loads((ROOT/'evidence/reference_results.json').read_text())
    for ds,splits in manifest.items():
        for split,entry in splits.items():
            if entry['released_file']:
                assert sha256(ROOT/entry['released_file'])==entry['npz_sha256'],(ds,split)
    for entry in json.loads((ROOT/'evidence/original_file_manifest.json').read_text()):
        ds,seed=entry['dataset'],entry['seed']
        original=ROOT/'evidence/original_runs'/ds/f'seed_{seed}.json'
        assert sha256(original)==entry['result_sha256'],original
        run=json.loads(original.read_text())
        assert run['seed']==seed
        assert run['best_epoch']==min(run['history'],key=lambda h:h['mse'])['epoch']
        if entry['released_checkpoint']:
            assert sha256(ROOT/entry['released_checkpoint'])==entry['checkpoint_sha256']
    for ds,r in reference.items():
        assert [x['seed'] for x in r['runs']]==list(range(2024,2029))
        for metric in ('mse','mae'):
            values=[run[metric] for run in r['runs']]
            assert abs(np.mean(values)-r[metric]['mean'])<1e-12
            assert abs(np.std(values,ddof=1)-r[metric]['std'])<1e-12
    print('PASS: public split hashes, 20 original run files, 15 checkpoints, and five-seed aggregation')
    if a.results:
        gaps=[]
        for ds,r in reference.items():
            for archived in r['runs']:
                fresh=json.loads((a.results/ds/f"seed_{archived['seed']}"/'result.json').read_text())
                for metric in ('mse','mae'):
                    gaps.append(abs(fresh['test'][metric]-archived[metric]))
        print(f'Fresh-run maximum absolute test metric difference: {max(gaps):.12g}')
        if max(gaps)>2e-6:
            raise SystemExit('Differences exceed 2e-6. Inspect environment, epoch selection, and input hashes.')
    if a.evaluate:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device(a.device)
        for ds in ('HumanActivity','USHCN','PhysioNet'):
            cfg=json.loads((ROOT/'configs'/f'{ds}.json').read_text())
            test=load_split(ROOT/'data'/ds/'test.npz',device)
            for archived in reference[ds]['runs']:
                ck=torch.load(ROOT/'evidence/checkpoints'/ds/f"seed_{archived['seed']}.pt",map_location='cpu',weights_only=True)
                model=make_model(cfg,device)
                model.load_state_dict(ck['state_dict'])
                result=evaluate(model,test,128)
                for metric in ('mse','mae'):
                    assert abs(result[metric]-archived[metric])<2e-6,(ds,archived['seed'],metric,result)
            print(f'PASS: {ds} five-checkpoint test re-evaluation (absolute tolerance 2e-6)')
            del test,model


if __name__=='__main__':
    main()
