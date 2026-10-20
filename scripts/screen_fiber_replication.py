"""Construct the fixed readout witnesses for all recorded replication models."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest, load_json, publish
from src.metrics.continuous_fibers import fit_circular_readout, features_and_phi_derivative
from src.training.train import build


def replication_records(index, protocol):
    if (protocol.get('schema')!='feature-topology.fiber-replication-protocol.v1'
            or protocol.get('gammas')!=[.03125,.5,1.,8.]
            or protocol.get('training_seeds')!=list(range(1,10))
            or protocol.get('theta_numerators')!=[1,3,5,7]
            or protocol.get('theta_denominator')!=8
            or protocol.get('models')!=36 or protocol.get('planned_domains')!=144):
        raise ValueError('Different recorded replication grid')
    expected={(gamma,seed) for gamma in protocol['gammas'] for seed in protocol['training_seeds']}
    rows=[row for row in index['runs'] if (row['gamma'],row['seed']) in expected]
    if (len(rows)!=36 or {(row['gamma'],row['seed']) for row in rows}!=expected
            or len({row['run_id'] for row in rows})!=36):
        raise ValueError('Missing or duplicate replication source model')
    for row in rows:
        config=row['config']
        if (config.get('gamma')!=row['gamma'] or config.get('seed')!=row['seed']
                or config.get('centered') is not True or config.get('relevance',0)!=0
                or config.get('swap',False) is not False or config.get('manifold','torus')!='torus'):
            raise ValueError('Replication model is not the unchanged primary task')
    return sorted(rows,key=lambda row:(row['gamma'],row['seed']))


def source_files(repo):
    return [repo/name for name in ['scripts/screen_fiber_replication.py',
        'src/metrics/continuous_fibers.py','src/models/mlp.py','src/training/train.py',
        'scripts/evaluate_nuisance_shift.py','configs/fiber_replication_v1.json',
        'docs/FIBER_REPLICATION_PROTOCOL.md']]


def screen_one(repo, record, arrays, protocol_sha, index_sha, sources, grid):
    gx, latent, embedding=grid
    directory=repo/'results/main/runs'/record['run_id']
    for name in ['config.json','summary.json']:
        if digest(directory/name)!=record['files'][name]:
            raise ValueError('Replication source training identity changed')
    checkpoint, selected=matched_checkpoint(directory,.1)
    if checkpoint is None or digest(checkpoint)!=record['files'][checkpoint.name]:
        raise ValueError('Replication has no source-bound first stored common-threshold checkpoint')
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if (state.get('config')!=record['config'] or state.get('step')!=selected['step']
            or any(value.dtype!=torch.float32 or not torch.isfinite(value).all()
                   for value in state['model'].values())):
        raise ValueError('Replication selected inference state changed')
    model=build(record['config'],record['seed'])
    model.load_state_dict(state['model'],strict=True)
    network=model.network.double().eval()
    axis=(np.arange(64)+.5)*2*np.pi/64
    validation=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    with torch.no_grad():
        features=np.concatenate([network.representations(batch)[-1].numpy()
            for batch in torch.from_numpy(gx.astype(np.float64)).split(1024)])
        readout,intercept,fit=fit_circular_readout(features,latent[:,1])
        derivative=np.concatenate([features_and_phi_derivative(network,batch,embedding)[1].numpy()
            for batch in torch.from_numpy(validation).split(512)])
    phi=validation[:,1]
    errors=np.linalg.norm(derivative@readout.T-np.column_stack([-np.sin(phi),np.cos(phi)]),axis=1)
    if not np.isfinite(errors).all() or not np.isfinite(readout).all():
        raise ValueError('Nonfinite fixed replication witness or derivative screen')
    passed=bool(errors.max()<2/np.pi)
    return dict(schema='feature-topology.fiber-replication-screen.v1',
        run_id=record['run_id'],gamma=record['gamma'],seed=record['seed'],
        status='eligible_fixed_witness' if passed else 'fixed_witness_screen_failed',
        selected_step=selected['step'],actual_training_loss=selected['training_loss'],
        checkpoint=checkpoint.name,checkpoint_sha256=digest(checkpoint),
        input_archive_sha256=record['archive_sha256'],input_index_sha256=index_sha,
        analysis_data_sha256=digest(arrays),protocol_sha256=protocol_sha,source_sha256=sources,
        samples=len(validation),empirical_max_derivative_error=float(errors.max()),
        empirical_q95_derivative_error=float(np.quantile(errors,.95)),
        empirical_screen_passed=passed,continuous_certificate=False,**fit,
        readout_hex=[[float(value).hex() for value in row] for row in readout],
        intercept_hex=[float(value).hex() for value in intercept],
        scope='Fixed approximate readout and sampled screen only; failure is not proof of noninjectivity')


def run(repo, arrays, output, source_commit):
    repo,arrays,output=Path(repo).resolve(),Path(arrays).resolve(),Path(output)
    if not re.fullmatch('[0-9a-f]{40}',source_commit):
        raise ValueError('Expected the committed screening source identity')
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/fiber_replication_v1.json'
    protocol=load_json(protocol_path.read_bytes())
    records=replication_records(index,protocol)
    if digest(arrays)!=index['analysis_data']['sha256']:
        raise ValueError('Replication embedding/grid asset changed')
    with np.load(arrays,allow_pickle=False) as frozen:
        grid=[frozen[name].copy() for name in ['gx','z','q']]
    sources={path.relative_to(repo).as_posix():digest(path) for path in source_files(repo)}
    torch.set_num_threads(2)
    rows=[]
    for record in records:
        path=output/(record['run_id']+'.json')
        if path.exists():
            result=load_json(path.read_bytes())
            if (result.get('source_sha256')!=sources or result.get('protocol_sha256')!=digest(protocol_path)
                    or result.get('input_index_sha256')!=digest(index_path)
                    or result.get('source_commit')!=source_commit
                    or result.get('run_id')!=record['run_id']):
                raise ValueError('Different existing immutable replication screen')
        else:
            result=screen_one(repo,record,arrays,digest(protocol_path),digest(index_path),sources,grid)
            result.update(source_commit=source_commit,host=dict(system=platform.system(),
                machine=platform.machine(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
            publish(output,{path.name:(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()})
        rows.append(result)
        print(json.dumps({key:result[key] for key in ['run_id','gamma','seed','status','empirical_max_derivative_error']}),flush=True)
    manifest=dict(schema='feature-topology.fiber-replication-screen-collection.v1',
        expected_models=36,complete_models=36,planned_domains=144,
        eligible_models=sum(row['empirical_screen_passed'] for row in rows),
        screen_failed_models=sum(not row['empirical_screen_passed'] for row in rows),
        training_seeds=protocol['training_seeds'],protocol_sha256=digest(protocol_path),
        source_commit=source_commit,records=rows,
        scope='Complete numerical witness screening; no continuum certificate or whole-study completion claim')
    publish(output,{'collection.json':(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()})
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--arrays',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    result=run(args.repo,args.arrays,args.output,args.source_commit)
    print(json.dumps({key:result[key] for key in ['complete_models','eligible_models','screen_failed_models']}))
