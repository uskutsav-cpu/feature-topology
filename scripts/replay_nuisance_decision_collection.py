"""Additional local replay of the recorded complete-classifier decision claims."""
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

from scripts.collect_nuisance_decision_results import collect
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_decision_verifier import classifier_from_state
from src.metrics.nuisance_decision_verifier import replay_decision


def execute(repo, collection, output, gamma=None):
    repo, collection, output = Path(repo).resolve(), Path(collection).resolve(), Path(output)
    checked = collect(repo, collection)
    if checked != load_json((collection/'collection.json').read_bytes()):
        raise ValueError('Stored classifier collection differs from source-bound artifacts')
    index = load_json((repo/'configs/completion/primary_inputs_v3.json').read_bytes())
    arrays = repo/'results/continuous_fiber_inputs/analysis_data.npz'
    if digest(arrays) != index['analysis_data']['sha256']:
        raise ValueError('Physical embedding differs from the frozen classifier source')
    with np.load(arrays, allow_pickle=False) as frozen:
        embedding = frozen['q'].copy()
    records = {row['run_id']:row for row in index['runs']}
    selected = [row for row in checked['records'] if gamma is None or row['gamma']==gamma]
    if not selected:
        raise ValueError('No recorded classifier domain at the requested gamma')
    ctx.prec = 128
    completed = []
    for row in selected:
        rid, theta = row['run_id'], row['theta_fraction'][0]
        witness = load_json((repo/'results/continuous_fiber_feasibility'/(rid+'.json')).read_bytes())
        checkpoint = repo/'results/main/runs'/rid/witness['checkpoint']
        stem = rid+f'-theta-{theta}-8'
        result_path = collection/(stem+'.json')
        result = load_json(result_path.read_bytes())
        if (not checkpoint.resolve().is_relative_to(repo)
                or digest(checkpoint) != result['binding']['checkpoint_sha256']
                or digest(arrays) != result['binding']['analysis_data_sha256']):
            raise ValueError('Physical classifier input differs from the recorded source')
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if state.get('step') != result['binding']['selected_step']:
            raise ValueError('Physical classifier selected step differs from the recorded source')
        certified = row['decision_certificate'] or row['counterexample_certificate']
        if certified:
            network = classifier_from_state(state, records[rid], embedding, theta)
            replay_decision(network, result)
        receipt = dict(schema='feature-topology.nuisance-decision-local-replay.v1',
            run_id=rid, theta_fraction=[theta,8], target_class=theta//2,
            result_sha256=digest(result_path), binding=result['binding'],
            decision_certificate=row['decision_certificate'],
            counterexample_certificate=row['counterexample_certificate'],
            replay_passed=certified, status='passed' if certified else 'not_a_certificate',
            verifier_sha256=digest(Path(__file__)),
            arithmetic=dict(python_flint=flint.__version__, precision_bits=ctx.prec),
            host=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version()),
            scope='Additional local execution of the same Arb verifier; not an independent implementation or Lean proof')
        publish(output, {stem+'.json':(json.dumps(receipt, indent=2)+'\n').encode()})
        completed.append(dict(run_id=rid, theta_fraction=[theta,8], status=receipt['status'],
                              decision_certificate=row['decision_certificate'],
                              counterexample_certificate=row['counterexample_certificate']))
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
