"""Replay an explicitly weaker derived margin; preserve original hosted claims.

Registered after a Mac/Linux enclosure difference in the initial local replay.
The single deterministic rule halves the hosted positive margin (integer floor
at denominator 2**64). No search, refit, domain change or classifier change.
It validates the same strict-correct-class property, not the original numerical
margin. Negative point claims remain unchanged.
"""
import argparse
import copy
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
from src.metrics.rational_fixed_fiber_verifier import exact_bound


def derived_claim(result, original_sha256):
    claim = copy.deepcopy(result)
    claim['schema'] = 'feature-topology.nuisance-decision-derived-claim.v1'
    claim['derivation'] = dict(original_result_sha256=original_sha256,
        policy='floor(original_positive_margin_numerator/2), unchanged denominator 2^64; negative claim unchanged',
        scope='Same specified correct-class property at a weaker numerical margin; original hosted margin is not validated here')
    if result['decision_certificate']:
        original = result['scaled_margin_lower_rational']
        exact_bound(original, 64)
        weakened = dict(numerator=str(int(original['numerator'])//2), denominator=str(1<<64))
        if not 0 < exact_bound(weakened, 64) < exact_bound(original, 64):
            raise ValueError('Fixed half-margin derivation is not strictly positive and weaker')
        claim['scaled_margin_lower_rational'] = weakened
    return claim


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
        claim = derived_claim(result, digest(result_path))
        if certified:
            network = classifier_from_state(state, records[rid], embedding, theta)
            replay_decision(network, claim)
        claim_path = output/(stem+'.local-claim.json')
        publish(output, {claim_path.name:(json.dumps(claim, indent=2)+'\n').encode()})
        receipt = dict(schema='feature-topology.nuisance-decision-local-replay.v2',
            run_id=rid, theta_fraction=[theta,8], target_class=theta//2,
            result_sha256=digest(result_path), binding=result['binding'],
            local_claim_sha256=digest(claim_path), hosted_numerical_margin_replay_claimed=False,
            decision_certificate=row['decision_certificate'],
            counterexample_certificate=row['counterexample_certificate'],
            replay_passed=certified, status='passed' if certified else 'not_a_certificate',
            verifier_sha256=digest(Path(__file__)),
            arithmetic=dict(python_flint=flint.__version__, precision_bits=ctx.prec),
            host=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version()),
            scope='Additional local replay of a fixed weaker derived margin using the same Arb verifier; not an independent implementation, validation of the hosted numerical margin, or Lean proof')
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
