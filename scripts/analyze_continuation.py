"""Reduced-resolution replication metrics, separate from production profiles.

No PH backend is substituted: persistence is explicitly not computed here.
Output records checkpoint, code and option hashes so edits invalidate the cache.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.checkpoints import atomic_json, fingerprint
from src.training.train import build, evaluate
from src.data.torus import dataset, coordinates
from src.metrics.geometry import cka, rms_scale, effective_rank
from src.metrics.jacobian import tangent_jacobians, summarize
from src.metrics.ntk import empirical_ntk
from src.metrics.probes import evaluate_probes
from src.metrics.fibers import torus_fiber_pairs, paired_separation, noise_decodability


def feature_values(network, x):
    with torch.no_grad():
        chunks = [network.representations(v) for v in x.split(512)]
    return [torch.cat([c[i] for c in chunks]).detach().cpu().numpy() for i in range(len(chunks[0]))]


def code_digest():
    root = Path(__file__).resolve().parents[1]
    dependencies = [
        'src/data/torus.py', 'src/models/mlp.py', 'src/training/train.py',
        'src/metrics/geometry.py', 'src/metrics/jacobian.py', 'src/metrics/ntk.py',
        'src/metrics/probes.py', 'src/metrics/fibers.py',
    ]
    paths = [Path(__file__)] + [root/path for path in dependencies]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def analyze(run, output, options, version):
    config = json.loads((run/'config.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if summary['status'] != 'converged':
        return None
    checkpoint = run/'final.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    key = fingerprint(dict(checkpoint=checkpoint_hash, options=options, code=version))
    destination = output/run.name/(key+'.json')
    if destination.exists():
        return json.loads(destination.read_text())
    torch.set_num_threads(1)
    model = build(config, config['seed'])
    gx, _, z, q = dataset(options['grid_points'], grid=True)
    px, py, pz, _ = dataset(options['probe_train'], seed=4001)
    tx, ty, tz, _ = dataset(options['probe_test'], seed=4002)
    initial = feature_values(model.network, gx)
    ids = np.random.default_rng(777).choice(len(gx), options['jacobian_points'], replace=False)
    ntk_ids = np.random.default_rng(778).choice(len(gx), options['ntk_points'], replace=False)
    k0 = empirical_ntk(model.network, gx[ntk_ids], config['gamma'])
    state = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(state['model']); model.eval()
    h = feature_values(model.network, gx)
    ph = feature_values(model.network, px); th = feature_values(model.network, tx)
    a, b = torus_fiber_pairs(options['fiber_points'])
    ha = feature_values(model.network, coordinates(a)@q.T)
    hb = feature_values(model.network, coordinates(b)@q.T)
    k = empirical_ntk(model.network, gx[ntk_ids], config['gamma'])
    _, accuracy = evaluate(model, tx, ty)
    row = dict(run_id=run.name, gamma=config['gamma'], seed=config['seed'], step=state['step'],
               training_loss=summary['history'][-1]['training_loss'], test_accuracy=accuracy,
               weight_displacement=summary['weight_displacement'],
               ntk_drift=float(np.linalg.norm(k-k0)/np.linalg.norm(k0)),
               checkpoint_sha256=checkpoint_hash, code_sha256=version, options=options,
               persistence={'status': 'not_computed', 'reason': 'This reduced CPU profile does not execute PH.'},
               layers=[])
    for layer, current in enumerate(h):
        scale = rms_scale(current)
        jac, _ = summarize(tangent_jacobians(model.network, z[ids], q, layer), scale)
        probes = evaluate_probes(ph[layer], th[layer], py.numpy(), ty.numpy(),
                                 pz[:, 1].numpy(), tz[:, 1].numpy(),
                                 seed=config['seed'], max_iter=options['probe_iterations'])
        values = dict(layer=layer+1, scale=scale, cka_drift=1-cka(initial[layer], current),
                      effective_rank=effective_rank(current), jacobian=jac, probes=probes,
                      nuisance_fiber=paired_separation(ha[layer], hb[layer], scale))
        if layer == len(h)-1:
            values['noise_probe'] = noise_decodability(ph[layer], th[layer],
                pz[:, 1].numpy(), tz[:, 1].numpy(), seed=config['seed'])
        row['layers'].append(values)
    atomic_json(destination, row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='results/continuation')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1: p.error('--limit must be positive')
    root = Path(a.root)
    study = json.loads((root/'study_config.json').read_text())
    version = code_digest()
    summaries = sorted(root.glob('runs/*/summary.json'))
    records = []
    for summary in summaries[:a.limit]:
        row = analyze(summary.parent, root/'diagnostics', study['metrics'], version)
        if row is not None:
            records.append(row)
            atomic_json(root/'diagnostics.json', records)
            print(json.dumps(dict(analyzed=len(records), gamma=row['gamma'], seed=row['seed'])), flush=True)


if __name__ == '__main__': main()
