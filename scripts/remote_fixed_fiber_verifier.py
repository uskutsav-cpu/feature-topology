"""Verify one entire smooth nuisance circle at a specified exact task coordinate."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flint import ctx
import flint
import torch

from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.remote_nuisance_evaluation import verify_input
from scripts.remote_variability_baselines import download
from src.metrics.fixed_fiber_verifier import FixedThetaIntervalNetwork,replay_circle,verify_circle


def source_files(repo):
    return [repo/name for name in [
        'scripts/remote_fixed_fiber_verifier.py','src/metrics/fixed_fiber_verifier.py',
        'src/metrics/continuous_fiber_affine_verifier.py','src/metrics/continuous_fiber_verifier.py',
        'docs/CONTINUOUS_FIBER_CRITERION.md','docs/CONTINUOUS_FIBER_AFFINE_EXTENSION.md',
        'configs/fixed_fiber_verifier_v2.json']]


def execute(repo,output,gamma,theta_numerator,replay=False):
    repo,output=Path(repo).resolve(),Path(output)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/fixed_fiber_verifier_v2.json'
    protocol=load_json(protocol_path.read_bytes())
    witnesses=load_json((repo/'results/continuous_fiber_feasibility/manifest.json').read_bytes())['runs']
    candidates=[row for row in witnesses if row['gamma']==gamma and row['seed']==0]
    if len(candidates)!=1:
        raise ValueError('Expected one fixed feasibility witness')
    witness=candidates[0]
    if (witness.get('empirical_screen_passed') is not True
            or witness.get('continuous_certificate') is not False
            or witness.get('protocol_sha256')!=digest(repo/'configs/continuous_fiber_feasibility_v1.json')):
        raise ValueError('This witness was not eligible under the recorded screen')
    witness_path=repo/'results/continuous_fiber_feasibility'/(witness['run_id']+'.json')
    if load_json(witness_path.read_bytes())!=witness:
        raise ValueError('Witness manifest and coefficient record disagree')
    for name,expected in witness['source_sha256'].items():
        if digest(repo/name)!=expected:
            raise ValueError('Witness-construction source changed: '+name)
    records=[row for row in index['runs'] if row['run_id']==witness['run_id']]
    if len(records)!=1 or records[0]['gamma']!=gamma or records[0]['seed']!=0:
        raise ValueError('Witness does not name a frozen primary checkpoint')
    record=records[0]
    with tempfile.TemporaryDirectory(prefix='feature-topology-continuum-') as temporary:
        temporary=Path(temporary)
        archive,arrays=temporary/'input.tar.gz',temporary/'analysis_data.npz'
        tag='primary-inputs-v3-2026-09-20'
        download(tag,record['asset'],archive)
        directory=verify_input(archive,record,temporary/'inputs')
        download(tag,index['analysis_data']['asset'],arrays)
        if digest(arrays)!=index['analysis_data']['sha256']:
            raise ValueError('Embedding source array changed')
        with np.load(arrays,allow_pickle=False) as frozen:
            q=frozen['q'].copy()
        checkpoint,selected=matched_checkpoint(directory,.1)
        if (checkpoint is None or digest(checkpoint)!=witness['checkpoint_sha256']
                or selected['step']!=witness['selected_step']):
            raise ValueError('Continuous witness checkpoint selection changed')
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        if state.get('config')!=record['config'] or state.get('step')!=selected['step']:
            raise ValueError('Continuous source model identity mismatch')
        if any(not torch.isfinite(value).all() for value in state['model'].values()):
            raise ValueError('Nonfinite source inference state')
        layers=[]
        for layer in range(record['config']['depth']):
            prefix=f'network.hidden.{layer}.'
            layers.append((state['model'][prefix+'weight'].numpy(),
                           state['model'][prefix+'bias'].numpy(),True))
        readout=np.array([[float.fromhex(value) for value in row] for row in witness['readout_hex']])
        if readout.shape!=(2,record['config']['width']) or not np.isfinite(readout).all():
            raise ValueError('Malformed circular readout coefficients')
        ctx.prec=128
        network=FixedThetaIntervalNetwork(layers,q,readout,theta_numerator=theta_numerator)
        binding=dict(run_id=record['run_id'],gamma=gamma,seed=0,
                     checkpoint_sha256=digest(checkpoint),input_archive_sha256=record['archive_sha256'],
                     input_index_sha256=digest(index_path),analysis_data_sha256=digest(arrays),
                     witness_sha256=digest(witness_path),protocol_sha256=digest(protocol_path),
                     theta_fraction=[theta_numerator,8],
                     source_sha256={str(p.relative_to(repo)):digest(p) for p in source_files(repo)})
        stem=record['run_id']+f'-theta-{theta_numerator}-8'
        result_path=output/(stem+'.json')
        if replay:
            result=load_json(result_path.read_bytes())
            if result.get('binding')!=binding:
                raise ValueError('Continuum result is not bound to the replayed source/witness')
            if result['continuous_certificate']:
                replay_circle(network,result)
            receipt=dict(schema='feature-topology.fixed-fiber-replay.v2',
                         run_id=record['run_id'],result_sha256=digest(result_path),
                         theta_fraction=[theta_numerator,8],
                         positive_certificate=result['continuous_certificate'],
                         standalone_replay_passed=bool(result['continuous_certificate']),
                         status='passed' if result['continuous_certificate'] else 'not_a_positive_certificate',
                         scope='Same Arb implementation in a separate process; no independent implementation or Lean claim')
            publish(output,{stem+'.replay.json':(json.dumps(receipt,indent=2)+'\n').encode()})
            return receipt
        budget=protocol['budget']
        result=verify_circle(network,maximum_boxes=budget['maximum_boxes'],
                              maximum_depth=budget['maximum_depth'],seconds=budget['seconds_per_witness'])
        result.update(schema='feature-topology.fixed-fiber-verification.v2',binding=binding,
                      arithmetic=dict(python_flint=flint.__version__,precision_bits=ctx.prec),
                      host=dict(python=platform.python_version(),torch=torch.__version__,
                                source_commit=os.environ.get('GITHUB_SHA'),
                                workflow_run=os.environ.get('GITHUB_RUN_ID')))
        publish(output,{stem+'.json':(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()})
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--gamma',type=float,required=True)
    parser.add_argument('--theta-numerator',type=int,choices=[1,3,5,7],required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    result=execute(args.repo,args.output,args.gamma,args.theta_numerator,args.replay)
    print(json.dumps({k:v for k,v in result.items() if k in [
        'status','continuous_certificate','checked_boxes','certified_leaves','unresolved_boxes',
        'standalone_replay_passed','scope']}))
