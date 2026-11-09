"""Combine measured CTLite results with explicitly attributed Table I baselines."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('HumanActivity', 'USHCN', 'PhysioNet', 'MIMIC')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, help='Use new runs instead of archived paper results')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/table1.csv')
    a = p.parse_args()
    with (ROOT/'evidence/table1_baselines.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    reference = json.loads((ROOT/'evidence/reference_results.json').read_text())
    for ds in DATASETS:
        if a.results:
            runs = [json.loads((a.results/ds/f'seed_{s}'/'result.json').read_text()) for s in range(2024,2029)]
            for seed, run in zip(range(2024,2029), runs):
                assert run['seed'] == seed and run['dataset'] == ds
            metrics = {m: dict(mean=float(np.mean([r['test'][m] for r in runs])),
                               std=float(np.std([r['test'][m] for r in runs], ddof=1))) for m in ('mse','mae')}
        else:
            metrics = reference[ds]
        rows.append(dict(model='CTLite', dataset=ds,
                         **{m+'_'+s: f'{metrics[m][s]:.8f}' for m in ('mse','mae') for s in ('mean','std')},
                         source='CTLite_new_five_seed_runs' if a.results else 'CTLite_archived_five_seed_runs'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| Model | '+' | '.join(d+' MSE / MAE' for d in DATASETS)+' |', '|---|'+'---|'*4]
    for model in dict.fromkeys(r['model'] for r in rows):
        cells=[]
        for ds in DATASETS:
            r=next(r for r in rows if r['model']==model and r['dataset']==ds)
            cells.append(' / '.join(r[m+'_mean']+' ± '+r[m+'_std'] for m in ('mse','mae')))
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    a.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
