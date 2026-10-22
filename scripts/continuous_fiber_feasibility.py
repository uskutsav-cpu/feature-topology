"""Screen fixed primary checkpoints for a separately scoped circle certificate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest, load_json, publish
from src.metrics.continuous_fibers import fit_circular_readout, features_and_phi_derivative
from src.training.train import build


def run(repo, arrays, output):
    repo, arrays = Path(repo).resolve(), Path(arrays)
    protocol_path = repo/'configs/continuous_fiber_feasibility_v1.json'
    protocol_sha = digest(protocol_path)
    index = load_json((repo/'configs/completion/primary_inputs_v3.json').read_bytes())
    if digest(arrays) != index['analysis_data']['sha256']:
        raise ValueError('Feasibility grid differs from the frozen primary array archive')
    with np.load(arrays, allow_pickle=False) as frozen:
        gx, latent, q = [frozen[k].copy() for k in ['gx', 'z', 'q']]
    axis = (np.arange(64)+.5)*2*np.pi/64
    validation = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    torch.set_num_threads(2)
    results = []
    for record in sorted([r for r in index['runs'] if r['seed'] == 0], key=lambda r:r['gamma']):
        directory = repo/'results/main/runs'/record['run_id']
        if digest(directory/'config.json') != record['files']['config.json'] or digest(directory/'summary.json') != record['files']['summary.json']:
            raise ValueError('Feasibility training identity differs from frozen sources')
        checkpoint, selected = matched_checkpoint(directory, .1)
        if checkpoint is None or digest(checkpoint) != record['files'][checkpoint.name]:
            raise ValueError('Feasibility selected checkpoint is not the frozen source')
        model = build(record['config'], 0)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model.load_state_dict(state['model'])
        network = model.network.double().eval()
        with torch.no_grad():
            features = np.concatenate([network.representations(batch)[-1].numpy()
                                       for batch in torch.from_numpy(gx.astype(np.float64)).split(1024)])
            readout, intercept, fit = fit_circular_readout(features, latent[:, 1])
            derivative = np.concatenate([features_and_phi_derivative(network, batch, q)[1].numpy()
                                         for batch in torch.from_numpy(validation).split(512)])
        phi = validation[:, 1]
        target_derivative = np.column_stack([-np.sin(phi), np.cos(phi)])
        errors = np.linalg.norm(derivative @ readout.T-target_derivative, axis=1)
        result = dict(schema='feature-topology.continuous-fiber-feasibility-result.v1',
                      run_id=record['run_id'], gamma=record['gamma'], seed=0,
                      selected_step=selected['step'], checkpoint=checkpoint.name,
                      checkpoint_sha256=digest(checkpoint), analysis_data_sha256=digest(arrays),
                      protocol_sha256=protocol_sha, samples=len(validation),
                      empirical_max_derivative_error=float(errors.max()),
                      empirical_q95_derivative_error=float(np.quantile(errors, .95)),
                      empirical_screen_passed=bool(errors.max() < 2/np.pi),
                      continuous_certificate=False, **fit,
                      readout_hex=[[float(value).hex() for value in row] for row in readout],
                      intercept_hex=[float(value).hex() for value in intercept],
                      source_sha256={str(p.relative_to(repo)):digest(p) for p in [
                          repo/'scripts/continuous_fiber_feasibility.py',
                          repo/'src/metrics/continuous_fibers.py']},
                      scope='Approximate witness and sampled feasibility only; not a rigorous certificate')
        publish(output, {record['run_id']+'.json': (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()})
        results.append(result)
        print(json.dumps({k:result[k] for k in ['gamma','empirical_max_derivative_error','empirical_screen_passed']}),flush=True)
    publish(output, {'manifest.json':(json.dumps(dict(
        schema='feature-topology.continuous-fiber-feasibility-manifest.v1',
        protocol_sha256=protocol_sha, cells=len(results), runs=results),indent=2,allow_nan=False)+'\n').encode()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--arrays',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.repo,args.arrays,args.output)
