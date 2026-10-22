"""Verify and separately replay one prospectively specified replication circle."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import flint
from flint import ctx
import numpy as np
import torch

from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import verify_input
from scripts.remote_nuisance_decision_verifier import classifier_from_state
from scripts.remote_variability_baselines import download
from scripts.screen_fiber_replication import replication_records
from scripts.validate_fiber_replication_screens import validate
from src.metrics.joint_fiber_verifier import JointMaskFixedThetaNetwork
from src.metrics.rational_fixed_fiber_verifier import exact_bound, replay_circle, verify_circle
from src.metrics.nuisance_decision_verifier import replay_decision, verify_decision


def source_files(repo):
    return [repo/name for name in [
        'scripts/remote_fiber_replication.py','scripts/screen_fiber_replication.py',
        'scripts/validate_fiber_replication_screens.py','scripts/remote_nuisance_decision_verifier.py',
        'scripts/evaluate_nuisance_shift.py','src/metrics/continuous_fibers.py',
        'src/metrics/continuous_fiber_verifier.py','src/metrics/continuous_fiber_affine_verifier.py',
        'src/metrics/fixed_fiber_verifier.py','src/metrics/joint_fiber_verifier.py',
        'src/metrics/rational_fixed_fiber_verifier.py','src/metrics/nuisance_decision_verifier.py',
        'src/models/mlp.py','src/data/torus.py','src/training/train.py',
        'configs/fiber_replication_v1.json','docs/FIBER_REPLICATION_PROTOCOL.md',
        '.github/workflows/fiber_replication_v1.yml','requirements.certification.txt']]


def buffered_decision(result):
    result=copy.deepcopy(result)
    if result['decision_certificate']:
        original=result['scaled_margin_lower_rational']
        exact_bound(original,64)
        lowered=dict(numerator=str(int(original['numerator'])//2),denominator=str(1<<64))
        result['unbuffered_margin_lower_rational']=original
        result['margin_encoding_policy']='Prospectively fixed integer-floor half margin at denominator 2^64'
        if not exact_bound(lowered,64)>0:
            result.update(status='inconclusive',decision_certificate=False,counterexample_certificate=False,
                reason='Fixed half-margin buffer has no positive exact bound')
            result.pop('leaf_codes');result.pop('scaled_margin_lower_rational')
        else:
            result['scaled_margin_lower_rational']=lowered
    return result


def execute(repo, output, gamma, seed, theta, replay=False):
    repo,output=Path(repo).resolve(),Path(output)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/fiber_replication_v1.json'
    protocol=load_json(protocol_path.read_bytes())
    records=replication_records(index,protocol)
    if theta not in protocol['theta_numerators']:
        raise ValueError('Unexpected replication theta')
    choices=[row for row in records if row['gamma']==gamma and row['seed']==seed]
    if len(choices)!=1:
        raise ValueError('Unexpected replication model')
    record=choices[0];rid=record['run_id'];stem=rid+f'-theta-{theta}-8'
    screen_root=repo/'results/fiber_replication_screens_v1'
    screens=validate(repo,screen_root)
    witness=next(row for row in screens['records'] if row['run_id']==rid)
    with tempfile.TemporaryDirectory(prefix='feature-topology-replication-') as temporary:
        temporary=Path(temporary)
        archive,arrays=temporary/'input.tar.gz',temporary/'analysis_data.npz'
        tag='primary-inputs-v3-2026-09-20'
        download(tag,record['asset'],archive)
        directory=verify_input(archive,record,temporary/'inputs')
        download(tag,index['analysis_data']['asset'],arrays)
        if digest(arrays)!=index['analysis_data']['sha256']:
            raise ValueError('Replication embedding source changed')
        with np.load(arrays,allow_pickle=False) as frozen:embedding=frozen['q'].copy()
        checkpoint,selected=matched_checkpoint(directory,.1)
        if (checkpoint is None or digest(checkpoint)!=witness['checkpoint_sha256']
                or selected['step']!=witness['selected_step']
                or selected['training_loss']!=witness['actual_training_loss']):
            raise ValueError('Replication checkpoint selection differs from the recorded witness')
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        if state.get('step')!=selected['step']:
            raise ValueError('Replication source state step changed')
        ctx.prec=128
        classifier=classifier_from_state(state,record,embedding,theta)
        retention_network=None
        if witness['empirical_screen_passed']:
            layers=[(state['model'][f'network.hidden.{layer}.weight'].numpy(),
                     state['model'][f'network.hidden.{layer}.bias'].numpy(),True)
                    for layer in range(record['config'].get('depth',4))]
            readout=np.array([[float.fromhex(value) for value in row] for row in witness['readout_hex']])
            retention_network=JointMaskFixedThetaNetwork(layers,embedding,readout,
                theta_numerator=theta,maximum_uncertain=6)
        binding=dict(run_id=rid,gamma=gamma,seed=seed,selected_step=selected['step'],
            actual_training_loss=selected['training_loss'],theta_fraction=[theta,8],target_class=theta//2,
            checkpoint_sha256=digest(checkpoint),input_archive_sha256=record['archive_sha256'],
            input_index_sha256=digest(index_path),analysis_data_sha256=digest(arrays),
            screen_result_sha256=digest(screen_root/(rid+'.json')),
            screen_collection_sha256=digest(screen_root/'collection.json'),protocol_sha256=digest(protocol_path),
            source_sha256={path.relative_to(repo).as_posix():digest(path) for path in source_files(repo)})
        path=output/(stem+'.json')
        if replay:
            result=load_json(path.read_bytes())
            if result.get('binding')!=binding:
                raise ValueError('Replication result differs from the replayed sources')
            retention,decision=result['retention'],result['decision']
            retained=retention.get('continuous_certificate') is True
            positive,negative=decision.get('decision_certificate') is True,decision.get('counterexample_certificate') is True
            if retained:
                if retention_network is None:raise ValueError('Unscreened replication witness has a positive certificate')
                replay_circle(retention_network,retention)
            elif retention.get('status') not in {'fixed_witness_screen_failed','fixed_witness_rejected','inconclusive'}:
                raise ValueError('Unknown replication retention outcome')
            if positive or negative:
                replay_decision(classifier,decision)
            elif (decision.get('status')!='inconclusive' or decision.get('decision_certificate') is not False
                  or decision.get('counterexample_certificate') is not False):
                raise ValueError('Unknown replication decision outcome')
            receipt=dict(schema='feature-topology.fiber-replication-replay.v1',run_id=rid,
                gamma=gamma,seed=seed,theta_fraction=[theta,8],result_sha256=digest(path),
                retention_certificate=retained,decision_certificate=positive,counterexample_certificate=negative,
                retention_replay_passed=retained,decision_replay_passed=positive or negative,
                retention_status='passed' if retained else 'not_a_positive_certificate',
                decision_status='passed' if positive or negative else 'not_a_certificate',
                joint_retention_and_correct_decision=retained and positive,
                scope='Separate-process execution of the same Arb verifiers; no independent verifier or Lean proof')
            publish(output,{stem+'.replay.json':(json.dumps(receipt,indent=2)+'\n').encode()})
            return receipt
        budget=protocol['budget']
        if retention_network is None:
            retention=dict(status='fixed_witness_screen_failed',continuous_certificate=False,
                theta_fraction=[theta,8],reason='The predeclared fixed numerical readout failed its screen; not evidence of noninjectivity')
        else:
            retention=verify_circle(retention_network,maximum_boxes=budget['maximum_boxes'],
                maximum_depth=budget['maximum_depth'],seconds=budget['seconds_per_property_per_domain'])
        decision=buffered_decision(verify_decision(classifier,maximum_boxes=budget['maximum_boxes'],
            maximum_depth=budget['maximum_depth'],seconds=budget['seconds_per_property_per_domain']))
        result=dict(schema='feature-topology.fiber-replication-domain.v1',binding=binding,
            retention=retention,decision=decision,
            joint_retention_and_correct_decision=retention['continuous_certificate'] and decision['decision_certificate'],
            arithmetic=dict(python_flint=flint.__version__,precision_bits=ctx.prec),
            host=dict(python=platform.python_version(),torch=torch.__version__,
                source_commit=os.environ.get('GITHUB_SHA'),workflow_run=os.environ.get('GITHUB_RUN_ID')),
            scope=protocol['scope'])
        publish(output,{stem+'.json':(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()})
        return dict(run_id=rid,gamma=gamma,seed=seed,theta_fraction=[theta,8],
            retention_status=retention['status'],decision_status=decision['status'],
            joint_retention_and_correct_decision=result['joint_retention_and_correct_decision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--gamma',type=float,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--theta-numerator',type=int,required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    print(json.dumps(execute(args.repo,args.output,args.gamma,args.seed,args.theta_numerator,args.replay)))
