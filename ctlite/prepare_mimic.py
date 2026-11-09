"""Build private MIMIC splits from an authorized De Brouwer complete_tensor.csv."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .data import save_records, sha256
from .train import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.destination.resolve().is_relative_to(root):
        p.error('Keep restricted data outside the public repository')
    if any((a.destination/f'{s}.npz').exists() for s in ('train','val','test')):
        p.error('Destination already contains splits')
    import pandas as pd
    frame=pd.read_csv(a.csv,index_col=0)
    frame=frame.astype({'UNIQUE_ID':'int16','TIME_STAMP':'int16','LABEL_CODE':'int16','VALUENORM':'float32'})
    frame=frame.sort_values(['UNIQUE_ID','TIME_STAMP'])
    wide=frame.pivot(index=['UNIQUE_ID','TIME_STAMP'],columns='LABEL_CODE',values='VALUENORM').sort_index()
    wide=wide.dropna(axis=1,how='all')
    if wide.shape[1]!=96:
        p.error(f'Expected 96 variables, found {wide.shape[1]}. Check the upstream preprocessing version.')
    ids=wide.index.get_level_values(0).unique()
    tmax=wide.index.get_level_values(1).max()
    if tmax!=96:
        p.error(f'Expected maximum timestamp 96, found {tmax}')
    train_val=len(ids)-math.ceil(.1*len(ids))
    ntrain=train_val-math.ceil(.1*train_val)
    rows=[]
    for _,frame in wide.groupby(level=0,sort=False):
        times=torch.tensor(frame.index.get_level_values(1).to_numpy()/tmax,dtype=torch.float32)
        values=torch.tensor(frame.to_numpy(),dtype=torch.float32)
        cut=int((times<=71.5/96).sum())
        x,y=values[:cut],values[cut:cut+3]
        xm,ym=torch.isfinite(x),torch.isfinite(y)
        row=dict(x=torch.nan_to_num(x),y=torch.nan_to_num(y),x_mask=xm.float(),y_mask=ym.float(),
                 x_mark=times[:cut],y_mark=times[cut:cut+3])
        rows.append(row)
    manifests={}
    for split,records in [('train',rows[:ntrain]),('val',rows[ntrain:train_val]),('test',rows[train_val:])]:
        dest=a.destination/f'{split}.npz'
        save_records(records,dest)
        manifests[split]=dict(records=len(records),npz_sha256=sha256(dest))
    dump(a.destination/'manifest.json',dict(source_sha256=sha256(a.csv),splits=manifests,
         preprocessing='sorted patient split 81/9/10, cutoff 71.5/96, next three recorded timestamps'))
    print('Prepared private MIMIC splits. Compare their hashes against data/manifest.json before claiming exact replication.')


if __name__=='__main__':
    main()
