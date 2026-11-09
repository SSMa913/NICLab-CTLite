"""Train the fixed Table I recipe with raw Adam weights and validation selection."""
import argparse
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from .model import CTLite
from .data import batches, encode_split, evaluate, load_split, predict, sha256

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def make_model(cfg, device):
    return CTLite(cfg['channels'], cfg['rank'], cfg['width'], cfg['rates']).to(device)


def fit(cfg, seed, train, val, test, device, folder, provenance):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    assert sum(p.numel() for p in model.parameters()) == cfg['parameters']
    best, stale, best_epoch = float('inf'), 0, 0
    state, history = None, []
    start = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        for batch in batches(train, cfg['batch_size'], shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            error = (predict(model, batch) - batch['y']) * batch['y_mask']
            loss = error.square().sum() / batch['y_mask'].sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        metrics = evaluate(model, val, cfg['validation_batch_size'])
        history.append(dict(epoch=epoch, lr=cfg['learning_rate'], **metrics))
        if metrics['mse'] < best:
            best, stale, best_epoch = metrics['mse'], 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(dict(dataset=cfg['dataset'], seed=seed, epoch=epoch, validation=metrics)), flush=True)
        if stale >= cfg['patience']:
            break
    model.load_state_dict(state)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'best.pt'
    torch.save(dict(state_dict=state, config=cfg, seed=seed), checkpoint)
    result = dict(dataset=cfg['dataset'], seed=seed, config=cfg, parameters=cfg['parameters'],
                  best_epoch=best_epoch, epochs=len(history), history=history,
                  validation=evaluate(model, val, cfg['validation_batch_size']),
                  test=evaluate(model, test, cfg['validation_batch_size']),
                  fit_seconds=time.perf_counter()-start, provenance=provenance,
                  checkpoint_sha256=sha256(checkpoint))
    dump(folder / 'result.json', result)
    print('RESULT ' + json.dumps({k: result[k] for k in ('dataset', 'seed', 'best_epoch', 'test')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, help='Directory with train.npz, val.npz and test.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: all five paper seeds')
    parser.add_argument('--resume', action='store_true', help='Reuse completed runs only if config and provenance match')
    parser.add_argument('--allow-data-mismatch', action='store_true', help='Allow other splits for new experiments, not paper replication')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    seeds = args.seeds or cfg['seeds']
    if len(seeds) != len(set(seeds)):
        parser.error('Seeds must be unique')
    data_dir = args.data_dir or ROOT/'data'/cfg['dataset']
    if cfg['dataset'] == 'MIMIC' and args.data_dir is None:
        parser.error('MIMIC requires --data-dir pointing to your authorized external data')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Use --device cpu for a portability check; published results used CUDA.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = {s: data_dir/f'{s}.npz' for s in ('train', 'val', 'test')}
    for p in paths.values():
        if not p.is_file():
            parser.error(f'Missing benchmark split: {p}')
    data_hashes = {s: sha256(p) for s, p in paths.items()}
    expected = json.loads((ROOT/'data/manifest.json').read_text())[cfg['dataset']]
    if not args.allow_data_mismatch and any(data_hashes[s] != expected[s]['npz_sha256'] for s in paths):
        parser.error('Data hashes differ from the paper splits. Check preprocessing, or explicitly use --allow-data-mismatch for a new experiment.')
    provenance = dict(data_sha256=data_hashes,
                      code_sha256={p.name: sha256(p) for p in (ROOT/'ctlite').glob('*.py')},
                      environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                       numpy=np.__version__, device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                                       cuda=torch.version.cuda, threads=4, tf32=False))
    # Construct summaries before the per-seed RNG reset, as in the archived runs.
    encoder = make_model(cfg, device)
    train = encode_split(encoder, load_split(paths['train'], device), cfg['encoding_batch_size'])
    val = encode_split(encoder, load_split(paths['val'], device), cfg['encoding_batch_size'])
    del encoder
    test = load_split(paths['test'], device)
    results = []
    for seed in seeds:
        folder = args.output/cfg['dataset']/f'seed_{seed}'
        if (folder/'result.json').exists():
            if not args.resume:
                parser.error(f'{folder} already exists. Choose another --output or use --resume.')
            result = json.loads((folder/'result.json').read_text())
            if result['config'] != cfg or result['provenance'] != provenance or sha256(folder/'best.pt') != result['checkpoint_sha256']:
                raise ValueError(f'Resume provenance mismatch: {folder}')
        else:
            result = fit(cfg, seed, train, val, test, device, folder, provenance)
        results.append(result)
    summary = dict(dataset=cfg['dataset'], seeds=seeds, runs=[dict(seed=r['seed'], **r['test']) for r in results],
                   **{m:dict(mean=float(np.mean([r['test'][m] for r in results])),
                             std=float(np.std([r['test'][m] for r in results], ddof=1)) if len(results)>1 else None)
                      for m in ('mse','mae')})
    dump(args.output/cfg['dataset']/'summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
