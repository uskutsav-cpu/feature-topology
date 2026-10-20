"""Replay source-bound circle artifacts using already recovered physical inputs.

This uses the same interval implementation on the current host. It is an
additional execution check, not an independently implemented verifier.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flint
from flint import ctx
import numpy as np
import torch

from scripts.collect_fixed_fiber_results import collect
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_rational_fiber_verifier import source_files
from src.metrics.joint_fiber_verifier import JointMaskFixedThetaNetwork
from src.metrics.rational_fixed_fiber_verifier import replay_circle


def execute(repo, collection, output, gamma=None):
    repo, collection, output = Path(repo).resolve(), Path(collection).resolve(), Path(output)
    checked = collect(repo, collection)
    if checked != load_json((collection/'collection.json').read_bytes()):
        raise ValueError('Stored circle collection differs from its source-bound artifacts')
    index_path = repo/'configs/completion/primary_inputs_v3.json'
    index = load_json(index_path.read_bytes())
    arrays = repo/'results/continuous_fiber_inputs/analysis_data.npz'
    if digest(arrays) != index['analysis_data']['sha256']:
        raise ValueError('Physical embedding asset differs from the frozen source')
    with np.load(arrays, allow_pickle=False) as archive:
        embedding = archive['q'].copy()
    protocol_path = repo/'configs/fixed_fiber_verifier_v4.json'
    protocol = load_json(protocol_path.read_bytes())
    models = {row['run_id']: row for row in index['runs']}
    witnesses = load_json((repo/'results/continuous_fiber_feasibility/manifest.json').read_bytes())['runs']
    witnesses = [row for row in witnesses if row['empirical_screen_passed'] and
                 (gamma is None or row['gamma'] == gamma)]
    if not witnesses:
        raise ValueError('No eligible fixed witness at the requested gamma')
    completed = []
    ctx.prec = 128
    for witness in witnesses:
        rid = witness['run_id']
        record = models[rid]
        checkpoint = repo/'results/main/runs'/rid/witness['checkpoint']
        if not checkpoint.resolve().is_relative_to(repo) or digest(checkpoint) != witness['checkpoint_sha256']:
            raise ValueError('Physical source checkpoint differs from the fixed witness')
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if (state['config'] != record['config'] or state['step'] != witness['selected_step']
                or any(not torch.isfinite(value).all() for value in state['model'].values())):
            raise ValueError('Source model identity or inference tensors changed')
        layers = [(state['model'][f'network.hidden.{i}.weight'].numpy(),
                   state['model'][f'network.hidden.{i}.bias'].numpy(), True)
                  for i in range(record['config']['depth'])]
        readout = np.array([[float.fromhex(value) for value in row] for row in witness['readout_hex']])
        for theta in [1, 3, 5, 7]:
            stem = rid+f'-theta-{theta}-8'
            result_path = collection/(stem+'.json')
            result = load_json(result_path.read_bytes())
            binding = dict(run_id=rid, gamma=witness['gamma'], seed=0,
                checkpoint_sha256=digest(checkpoint), input_archive_sha256=record['archive_sha256'],
                input_index_sha256=digest(index_path), analysis_data_sha256=digest(arrays),
                witness_sha256=digest(repo/'results/continuous_fiber_feasibility'/(rid+'.json')),
                protocol_sha256=digest(protocol_path), theta_fraction=[theta,8],
                source_sha256={str(p.relative_to(repo)):digest(p) for p in source_files(repo)})
            if result['binding'] != binding:
                raise ValueError('Local physical source differs from the hosted certificate binding')
            positive = result['continuous_certificate'] is True
            if positive:
                network = JointMaskFixedThetaNetwork(layers, embedding, readout,
                    theta_numerator=theta,
                    maximum_uncertain=protocol['maximum_uncertain_masks_to_enumerate'])
                replay_circle(network, result)
            receipt = dict(schema='feature-topology.fixed-fiber-local-replay.v4',
                run_id=rid, theta_fraction=[theta,8], result_sha256=digest(result_path),
                binding=binding, positive_certificate=positive,
                replay_passed=positive, status='passed' if positive else 'not_a_positive_certificate',
                verifier_sha256=digest(Path(__file__)),
                arithmetic=dict(python_flint=flint.__version__, precision_bits=ctx.prec),
                host=dict(system=platform.system(), machine=platform.machine(),
                          python=platform.python_version()),
                scope='Additional local execution of the same interval verifier; not an independent implementation or a Lean proof')
            publish(output, {stem+'.json':(json.dumps(receipt,indent=2)+'\n').encode()})
            completed.append(dict(run_id=rid, theta_fraction=[theta,8], status=receipt['status']))
            print(json.dumps(completed[-1]), flush=True)
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gamma', type=float)
    args = parser.parse_args()
    execute(args.repo, args.collection, args.output, args.gamma)
