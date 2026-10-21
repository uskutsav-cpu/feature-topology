"""Complete the recorded finite task-preserving fiber study from saved weights."""
import argparse
import copy
import json
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.evaluate_production_noise import binding,close_numeric,load_arrays,plan,source_files as noise_sources
from scripts.recover_release_evidence import digest,load_json,publish
from src.metrics.task_preserving_fibers import evaluate_family
from src.training.train import build


PROTOCOL='configs/task_preserving_fiber_completion_v1.json'


def task_plan(repo):
    protocol=load_json((repo/PROTOCOL).read_bytes())
    if (protocol['models']!=210 or protocol['grid_side']!=100 or protocol['collision_tolerance']!=1e-10
            or protocol['training_seeds']!=list(range(5))
            or protocol['gammas']!=[.125,.5,1.,4.,16.,64.,128.]
            or protocol['relevance_levels']!=[0.,.05,.1,.25,.5,1.]
            or {k:protocol['adversarial_search'][k] for k in ['starts','steps','optimizer','learning_rate','seed',
                'minimum_phi_arc','maximum_phi_arc']}!=dict(starts=16,steps=64,optimizer='torch.optim.Adam',
                    learning_rate=.03,seed=1729,minimum_phi_arc=np.pi/8,maximum_phi_arc=np.pi)):
        raise ValueError('Different recorded task-preserving completion design')
    rows=[row for row in plan(repo) if 'relevance' in row['roles']]
    actual={(r['relevance'],r['record']['gamma'],r['record']['seed']) for r in rows}
    expected={(level,gamma,seed) for level in protocol['relevance_levels']
        for gamma in protocol['gammas'] for seed in protocol['training_seeds']}
    if len(rows)!=210 or actual!=expected:raise ValueError('Incomplete task-preserving source grid')
    return rows,protocol


def source_files(repo):
    return sorted(set(noise_sources(repo)+[repo/name for name in [PROTOCOL,
        'docs/TASK_PRESERVING_FIBER_PROTOCOL.md','src/metrics/task_preserving_fibers.py',
        'scripts/evaluate_task_preserving_fibers.py','scripts/recover_release_evidence.py',
        'scripts/remote_variability_baselines.py','src/training/checkpoints.py']]))


def task_binding(repo,row,sources):
    old_protocol=load_json((repo/'configs/production_noise_completion_v1.json').read_bytes())
    value,checkpoint,summary=binding(repo,row,old_protocol,sources)
    value['task_preserving_protocol_sha256']=digest(repo/PROTOCOL)
    return value,checkpoint,summary


def evaluate(repo,row,protocol,sources,arrays):
    value,checkpoint,summary=task_binding(repo,row,sources)
    common=dict(schema='feature-topology.task-preserving-fiber-evaluation.v1',binding=value,
        source_training_status=summary['status'],continuum_certificate=False,scope=protocol['scope'])
    if checkpoint is None:return dict(common,status='unreached_common_risk',families=[])
    state=torch.load(checkpoint,weights_only=True,map_location='cpu');record=row['record']
    if (state.get('config')!=record['config'] or state.get('step')!=value['selected_step']
            or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state['model'].values())):
        raise ValueError('Task-preserving source state changed')
    model=build(record['config'],record['seed']);model.load_state_dict(state['model'],strict=True)
    model=model.double().eval().requires_grad_(False);q=torch.from_numpy(arrays['q']).double()
    task=evaluate_family(model,q,row['relevance'],True,protocol)
    if row['relevance']==0:
        control=copy.deepcopy(task);control.update(family='fixed_theta_original_factor_control',task_coordinate_maximum_error=None)
    else:control=evaluate_family(model,q,row['relevance'],False,protocol)
    return dict(common,status='evaluated',families=[task,control])


def run(repo,output,source_commit,replay=False):
    repo,output=Path(repo).resolve(),Path(output)
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Expected committed source identity')
    rows,protocol=task_plan(repo);sources={p.relative_to(repo).as_posix():digest(p) for p in source_files(repo)}
    host=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,threads=2,dtype='float64')
    cache={};complete=[];torch.set_num_threads(2)
    for row in rows:
        rid=row['record']['run_id'];path=output/(rid+'.json');receipt_path=output/(rid+'.replay.json')
        arrays=load_arrays(repo,row,cache);expected,_,_=task_binding(repo,row,sources)
        existing=load_json(path.read_bytes()) if path.exists() else None
        if existing is not None and (existing.get('binding')!=expected or existing.get('source_commit')!=source_commit
                                      or existing.get('host')!=host):
            raise ValueError('Different existing immutable task-preserving record')
        if replay:
            if existing is None:raise ValueError('Missing task-preserving result for replay')
            receipt=dict(schema='feature-topology.task-preserving-fiber-replay.v1',run_id=rid,
                result_sha256=digest(path),binding=expected,host=host,replay_verified=True,
                numeric_replay=existing['status']=='evaluated',
                scope='Same-host repeated numerical evaluation; not an independent verifier or new training replicate')
            if receipt_path.exists():
                if load_json(receipt_path.read_bytes())!=receipt:raise ValueError('Different task-preserving replay receipt')
            else:
                actual=evaluate(repo,row,protocol,sources,arrays);actual.update(source_commit=source_commit,host=host)
                close_numeric(actual,existing)
                publish(output,{receipt_path.name:(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()})
            result=existing
        else:
            result=existing
            if result is None:
                result=evaluate(repo,row,protocol,sources,arrays);result.update(source_commit=source_commit,host=host)
                publish(output,{path.name:(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()})
        complete.append(dict(run_id=rid,status=result['status'],result_sha256=digest(path)))
        print(json.dumps(dict(completed=len(complete),run_id=rid,gamma=row['record']['gamma'],
            seed=row['record']['seed'],relevance=row['relevance'],status=result['status'],replay=replay)),flush=True)
    stage='replay' if replay else 'generation'
    manifest=dict(schema='feature-topology.task-preserving-fiber-stage.v1',expected_models=210,
        complete_models=len(complete),stage=stage,source_commit=source_commit,
        protocol_sha256=digest(repo/PROTOCOL),records=complete)
    publish(output,{stage+'_stage.json':(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()})
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    run(args.repo,args.output,args.source_commit,args.replay)
