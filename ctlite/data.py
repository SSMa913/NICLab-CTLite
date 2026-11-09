"""Lossless, pickle-free benchmark split storage and fixed global padding."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

FIELDS = ('x', 'x_mask', 'x_mark', 'y', 'y_mask', 'y_mark')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_records(rows, path):
    """Export only the six numeric input/target fields, without identifiers."""
    arrays = {}
    for key in FIELDS:
        seq = [r[key].detach().cpu().float().numpy() for r in rows]
        if key.endswith('mark'):
            seq = [v.reshape(-1, 1) for v in seq]
        arrays[key] = np.concatenate(seq, axis=0)
        arrays[key + '_offsets'] = np.cumsum([0] + [len(v) for v in seq], dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_split(path, device):
    """Match the original experiment's float32 padding over the whole split."""
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for key in FIELDS:
            values, offsets = data[key], data[key + '_offsets']
            if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
                raise ValueError(f'Invalid offsets in {path}: {key}')
            seq = [torch.from_numpy(values[a:b].copy()).float() for a, b in zip(offsets[:-1], offsets[1:])]
            result[key] = pad_sequence(seq, batch_first=True).to(device)
    return result


def batches(data, batch_size, shuffle=False):
    n = len(data['y'])
    device = data['y'].device
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for ids in order.split(batch_size):
        yield {k: v[ids] for k, v in data.items()}


def encode_split(model, data, batch_size=256):
    """The fixed encoder has no learned parameters; its summaries can be reused."""
    with torch.no_grad():
        chunks = [model.encode(data['x'][i:i+batch_size], data['x_mark'][i:i+batch_size],
                               data['x_mask'][i:i+batch_size])
                  for i in range(0, len(data['x']), batch_size)]
    return {**{'z_' + k: torch.cat([c[k] for c in chunks]) for k in chunks[0]},
            **{k: data[k] for k in ('y', 'y_mask', 'y_mark')}}


def predict(model, batch):
    if 'z_last' in batch:
        return model.decode({k[2:]: v for k, v in batch.items() if k.startswith('z_')}, batch['y_mark'])
    return model(**batch)


def evaluate(model, data, batch_size=128):
    model.eval()
    totals = torch.zeros(3, device=data['y'].device)
    with torch.no_grad():
        for batch in batches(data, batch_size):
            error = (predict(model, batch) - batch['y']) * batch['y_mask']
            totals += torch.stack([error.square().sum(), error.abs().sum(), batch['y_mask'].sum()])
    squared, absolute, count = totals.tolist()
    if count == 0:
        raise ValueError('No observed targets in evaluation split')
    return dict(mse=squared/count, mae=absolute/count, points=int(count))
