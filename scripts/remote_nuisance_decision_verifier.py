"""Verify correct decisions on the sixteen previously specified nuisance circles."""
import argparse
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

from scripts.collect_fixed_fiber_results import collect
from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.remote_nuisance_evaluation import verify_input
from scripts.remote_variability_baselines import download
from src.metrics.nuisance_decision_verifier import CenteredCircleClassifier,replay_decision,verify_decision


def source_files(repo):
    return [repo/name for name in [
        'scripts/remote_nuisance_decision_verifier.py',
        'src/metrics/nuisance_decision_verifier.py',
        'src/metrics/continuous_fiber_verifier.py','src/metrics/fixed_fiber_verifier.py',
        'src/metrics/rational_fixed_fiber_verifier.py','src/models/mlp.py','src/data/torus.py',
        'src/training/train.py',
        'scripts/evaluate_nuisance_shift.py','configs/nuisance_decision_verifier_v1.json',
        'docs/NUISANCE_DECISION_CERTIFICATION.md',
        '.github/workflows/nuisance_decision_verifier_v1.yml']]


def classifier_from_state(state,record,embedding,theta_numerator):
    config=record['config']
    if (state.get('config')!=config or config.get('centered') is not True
            or config.get('manifold','torus')!='torus' or config.get('swap',False) is not False
            or config.get('relevance',0)!=0):
        raise ValueError('Decision certificate requires the unchanged centered primary torus classifier')
    # Match the original builder's defaults; historical configs omit dimension.
    depth,width,dimension=(config.get('depth',4),config.get('width',256),config.get('dimension',16))
    model=state['model']
    expected=set()
    for branch in ['network','initial']:
        for layer in range(depth):
            expected.update([f'{branch}.hidden.{layer}.weight',f'{branch}.hidden.{layer}.bias'])
        expected.update([f'{branch}.head.weight',f'{branch}.head.bias'])
    if set(model)!=expected:
        raise ValueError('Unsupported or incomplete source classifier architecture')
    if (embedding.shape!=(dimension,4) or embedding.dtype!=np.float32
            or not np.isfinite(embedding).all()):
        raise ValueError('Invalid source embedding for the complete classifier')
    branches=[]
    for branch in ['network','initial']:
        layers=[]
        for layer in range(depth):
            weight,bias=(model[f'{branch}.hidden.{layer}.{name}'] for name in ['weight','bias'])
            if (tuple(weight.shape)!=(width,dimension if layer==0 else width)
                    or tuple(bias.shape)!=(width,)):
                raise ValueError('Source hidden-layer dimensions changed')
            layers.append((weight.numpy(),bias.numpy(),True))
        weight,bias=(model[f'{branch}.head.{name}'] for name in ['weight','bias'])
        if tuple(weight.shape)!=(4,width) or tuple(bias.shape)!=(4,):
            raise ValueError('Source linear head is not the four-class head')
        layers.append((weight.numpy(),bias.numpy(),False))
        branches.append(layers)
    if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in model.values()):
        raise ValueError('Source classifier contains unsupported or nonfinite tensors')
    return CenteredCircleClassifier(*branches,embedding,config['gamma'],theta_numerator)


def execute(repo,output,gamma,theta_numerator,replay=False):
    repo,output=Path(repo).resolve(),Path(output)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/nuisance_decision_verifier_v1.json'
    protocol=load_json(protocol_path.read_bytes())
    retention_root=repo/'results/fixed_fiber_verification_v4'
    collection=collect(repo,retention_root)
    if collection!=load_json((retention_root/'collection.json').read_bytes()):
        raise ValueError('Different source-bound retention collection')
    rows=[row for row in collection['records'] if row['gamma']==gamma
          and row['theta_fraction']==[theta_numerator,8]]
    if len(rows)!=1 or rows[0]['continuous_certificate'] is not True:
        raise ValueError('Expected one previously retained circle certificate')
    prior=rows[0];rid=prior['run_id'];stem=rid+f'-theta-{theta_numerator}-8'
    witness=load_json((repo/'results/continuous_fiber_feasibility'/(rid+'.json')).read_bytes())
    retention_path=retention_root/(stem+'.json')
    retention=load_json(retention_path.read_bytes())
    record=next(row for row in index['runs'] if row['run_id']==rid)
    if record['seed']!=0 or record['gamma']!=gamma:
        raise ValueError('Different source model in the decision protocol')
    with tempfile.TemporaryDirectory(prefix='feature-topology-decision-') as temporary:
        temporary=Path(temporary)
        archive,arrays=temporary/'input.tar.gz',temporary/'analysis_data.npz'
        tag='primary-inputs-v3-2026-09-20'
        download(tag,record['asset'],archive)
        directory=verify_input(archive,record,temporary/'inputs')
        download(tag,index['analysis_data']['asset'],arrays)
        if digest(arrays)!=index['analysis_data']['sha256']:
            raise ValueError('Decision source embedding asset changed')
        with np.load(arrays,allow_pickle=False) as frozen:embedding=frozen['q'].copy()
        checkpoint,selected=matched_checkpoint(directory,.1)
        if (checkpoint is None or digest(checkpoint)!=retention['binding']['checkpoint_sha256']
                or selected['step']!=witness['selected_step']):
            raise ValueError('Decision and retention checkpoints differ')
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        if state.get('step')!=selected['step']:
            raise ValueError('Source classifier step differs from its stored loss record')
        ctx.prec=128
        network=classifier_from_state(state,record,embedding,theta_numerator)
        binding=dict(run_id=rid,gamma=gamma,seed=0,selected_step=selected['step'],
            checkpoint_sha256=digest(checkpoint),input_archive_sha256=record['archive_sha256'],
            input_index_sha256=digest(index_path),analysis_data_sha256=digest(arrays),
            retention_certificate_sha256=digest(retention_path),
            retention_replay_sha256=digest(retention_root/(stem+'.replay.json')),
            protocol_sha256=digest(protocol_path),theta_fraction=[theta_numerator,8],
            target_class=network.target_class,
            source_sha256={str(path.relative_to(repo)):digest(path) for path in source_files(repo)})
        result_path=output/(stem+'.json')
        if replay:
            result=load_json(result_path.read_bytes())
            if result.get('binding')!=binding:
                raise ValueError('Classifier decision result is not bound to these source inputs')
            positive=result.get('decision_certificate') is True
            negative=result.get('counterexample_certificate') is True
            if positive or negative:
                replay_decision(network,result)
            elif (result.get('status')!='inconclusive'
                  or result.get('decision_certificate') is not False
                  or result.get('counterexample_certificate') is not False):
                raise ValueError('Unknown classifier decision attempt status')
            receipt=dict(schema='feature-topology.nuisance-decision-replay.v1',run_id=rid,
                theta_fraction=[theta_numerator,8],target_class=network.target_class,
                result_sha256=digest(result_path),decision_certificate=positive,
                counterexample_certificate=negative,standalone_replay_passed=positive or negative,
                status='passed' if positive or negative else 'not_a_certificate',
                scope='Separate-process execution of the same Arb verifier, not an independent implementation or Lean proof')
            publish(output,{stem+'.replay.json':(json.dumps(receipt,indent=2)+'\n').encode()})
            return receipt
        budget=protocol['budget']
        result=verify_decision(network,maximum_boxes=budget['maximum_boxes'],
            maximum_depth=budget['maximum_depth'],seconds=budget['seconds_per_domain'])
        result.update(schema='feature-topology.nuisance-decision-verification.v1',binding=binding,
            arithmetic=dict(python_flint=flint.__version__,precision_bits=ctx.prec),
            host=dict(python=platform.python_version(),torch=torch.__version__,
                      source_commit=os.environ.get('GITHUB_SHA'),workflow_run=os.environ.get('GITHUB_RUN_ID')))
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
    print(json.dumps({key:value for key,value in result.items() if key in [
        'status','decision_certificate','counterexample_certificate','checked_boxes',
        'certified_leaves','unresolved_intervals','standalone_replay_passed']}))
