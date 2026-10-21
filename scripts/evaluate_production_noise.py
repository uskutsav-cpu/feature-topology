"""Complete the original main/relevance noisy-decoding component from saved weights."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import sklearn
import torch

from scripts.compute_metrics import ANALYSIS_DEFAULTS, analysis_condition
from scripts.evaluate_nuisance_shift import matched_checkpoint
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.remote_variability_baselines import download
from src.metrics.fibers import noise_decodability
from src.training.train import build


def plan(repo):
    repo=Path(repo)
    protocol=load_json((repo/'configs/production_noise_completion_v1.json').read_bytes())
    if (protocol.get('unique_models')!=305 or protocol.get('main_models')!=130
            or protocol.get('relevance_models')!=210 or protocol.get('shared_models')!=35
            or protocol.get('noise_levels')!=[0.,.001,.01,.1]
            or protocol.get('main_training_seeds')!=list(range(10))
            or protocol.get('relevance_training_seeds')!=list(range(5))):
        raise ValueError('Different recorded production-noise completion design')
    registry=load_json((repo/'configs/completion/production_cohorts_v3.json').read_bytes())
    wanted={'primary','relevance-0.05','relevance-0.1','relevance-0.25','relevance-0.5','relevance-1.0'}
    cohorts=[row for row in registry['cohorts'] if row['name'] in wanted]
    if len(cohorts)!=6 or {row['name'] for row in cohorts}!=wanted:
        raise ValueError('Missing or duplicate primary/relevance noise source cohort')
    rows=[]
    for cohort in cohorts:
        index_path=repo/cohort['index'];index=load_json(index_path.read_bytes())
        for record in index['runs']:
            config=record['config'];relevance=float(config.get('relevance',0.))
            roles=['main'] if cohort['name']=='primary' else ['relevance']
            if (cohort['name']=='primary' and record['gamma'] in protocol['relevance_gammas']
                    and record['seed'] in protocol['relevance_training_seeds']):roles.append('relevance')
            if (config.get('gamma')!=record['gamma'] or config.get('seed')!=record['seed']
                    or config.get('width',256)!=256 or config.get('depth',4)!=4
                    or config.get('dimension',16)!=16 or config.get('centered') is not True
                    or config.get('manifold','torus')!='torus' or config.get('swap',False) is not False
                    or config.get('nuisance_condition','iid')!='iid'
                    or config.get('relevance_mode','periodic')!='periodic'
                    or (cohort['name']=='primary' and relevance!=0)
                    or (cohort['name']!='primary' and cohort['name']!=f'relevance-{relevance}')):
                raise ValueError('Noise source is not the unchanged primary/relevance task')
            array_path=('results/continuous_fiber_inputs/analysis_data.npz' if cohort['name']=='primary'
                else f"results/production_noise_inputs/{Path(cohort['index']).stem}.npz")
            rows.append(dict(record=record,index=cohort['index'],index_sha256=digest(index_path),
                input_tag=cohort['input_tag'],cohort=cohort['name'],relevance=relevance,roles=roles,
                analysis_data=index['analysis_data'],array_path=array_path))
    expected={(0.,float(gamma),seed) for gamma in protocol['main_gammas'] for seed in range(10)}
    expected|={(float(level),float(gamma),seed) for level in protocol['relevance_levels'] if level>0
               for gamma in protocol['relevance_gammas'] for seed in range(5)}
    observed=[(row['relevance'],float(row['record']['gamma']),row['record']['seed']) for row in rows]
    if (len(rows)!=305 or len(set(observed))!=305 or set(observed)!=expected
            or len({row['record']['run_id'] for row in rows})!=305
            or sum('main' in row['roles'] for row in rows)!=130
            or sum('relevance' in row['roles'] for row in rows)!=210):
        raise ValueError('Incomplete or duplicate production-noise model grid')
    return sorted(rows,key=lambda row:(row['relevance'],row['record']['gamma'],row['record']['seed']))


def source_files(repo):
    return [repo/name for name in ['scripts/evaluate_production_noise.py','src/metrics/fibers.py',
        'scripts/compute_metrics.py','scripts/evaluate_nuisance_shift.py','src/training/train.py',
        'src/models/mlp.py','src/data/torus.py','configs/production_noise_completion_v1.json',
        'configs/analysis_plan_v2.json','configs/completion/production_cohorts_v3.json',
        'docs/PRODUCTION_NOISE_COMPLETION.md']]


def validate_probe_arrays(arrays,config):
    condition={**ANALYSIS_DEFAULTS,**load_json(str(arrays['data_config']).encode())}
    if condition!=analysis_condition(config):
        raise ValueError('Noise probe archive has a different task condition')
    shapes={'px':(5000,16),'pz':(5000,2),'tx':(2000,16),'tz':(2000,2),'q':(16,4)}
    for name,shape in shapes.items():
        if arrays[name].shape!=shape or arrays[name].dtype!=np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError('Noise probe arrays differ from the frozen split or embedding')
    if set(map(tuple,arrays['pz'])) & set(map(tuple,arrays['tz'])):
        raise ValueError('Noise probe training and held-out latent samples overlap')


def load_arrays(repo,row,cache):
    key=row['index']
    if key in cache:return cache[key]
    path=repo/row['array_path'];path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix='noise-input-',dir=path.parent) as temporary:
            candidate=Path(temporary)/'analysis_data.npz'
            download(row['input_tag'],row['analysis_data']['asset'],candidate)
            if digest(candidate)!=row['analysis_data']['sha256']:
                raise ValueError('Downloaded noise probe archive checksum mismatch')
            candidate.replace(path)
    if digest(path)!=row['analysis_data']['sha256']:
        raise ValueError('Physical noise probe archive changed')
    with np.load(path,allow_pickle=False) as archive:
        arrays={name:archive[name].copy() for name in ['data_config','px','pz','tx','tz','q']}
    validate_probe_arrays(arrays,row['record']['config'])
    cache[key]=arrays
    return arrays


def run_directory(repo,row):
    rid=row['record']['run_id']
    if row['cohort']=='primary':paths=[repo/'results/main/runs'/rid]
    else:paths=[path.parent for path in (repo/'results/relevance').glob(f'*/runs/{rid}/config.json')]
    if len(paths)!=1 or not paths[0].resolve().is_relative_to(repo):
        raise ValueError('Expected exactly one physical source run for production noise')
    return paths[0].resolve()


def binding(repo,row,protocol,sources):
    record=row['record'];directory=run_directory(repo,row)
    for name in ['config.json','summary.json']:
        if digest(directory/name)!=record['files'][name]:
            raise ValueError('Noise training metadata differs from immutable input members')
    config=load_json((directory/'config.json').read_bytes())
    summary=load_json((directory/'summary.json').read_bytes())
    if config!=record['config'] or summary['config']!=config or summary['run_id']!=record['run_id']:
        raise ValueError('Noise training identity changed')
    checkpoint,selected=matched_checkpoint(directory,.1)
    if checkpoint is not None and digest(checkpoint)!=record['files'].get(checkpoint.name):
        raise ValueError('Noise common-threshold checkpoint differs from source members')
    value=dict(run_id=record['run_id'],gamma=record['gamma'],seed=record['seed'],relevance=row['relevance'],
        roles=row['roles'],run_directory=directory.relative_to(repo).as_posix(),input_index=row['index'],
        input_index_sha256=row['index_sha256'],input_archive_sha256=record['archive_sha256'],
        config_sha256=digest(directory/'config.json'),training_summary_sha256=digest(directory/'summary.json'),
        analysis_data_path=row['array_path'],analysis_data_sha256=row['analysis_data']['sha256'],
        checkpoint=checkpoint.name if checkpoint is not None else None,
        checkpoint_sha256=digest(checkpoint) if checkpoint is not None else None,
        selected_step=selected['step'] if selected is not None else None,
        actual_training_loss=selected['training_loss'] if selected is not None else None,
        protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),source_sha256=sources)
    return value,checkpoint,summary


def evaluate(repo,row,protocol,sources,arrays):
    value,checkpoint,summary=binding(repo,row,protocol,sources)
    losses=[record['training_loss'] for record in summary['history']]
    if not all(np.isfinite(loss) for loss in losses):
        raise ValueError('Nonfinite stored loss in noise selection metadata')
    common=dict(schema='feature-topology.production-noise-evaluation.v1',binding=value,
        source_training_status=summary['status'],minimum_stored_training_loss=min(losses) if losses else None,
        noise_levels=protocol['noise_levels'],information_destruction_proved=False,scope=protocol['scope'])
    if checkpoint is None:
        return dict(common,status='unreached_common_risk',constant_training_representation=None,noise_rows=[])
    record=row['record'];state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if (state.get('config')!=record['config'] or state.get('step')!=value['selected_step']
            or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state['model'].values())):
        raise ValueError('Noise source inference state changed')
    model=build(record['config'],record['seed']);model.load_state_dict(state['model'],strict=True);model.eval()
    def features(name):
        with torch.no_grad():
            return np.concatenate([model.network.representations(part)[-1].numpy()
                for part in torch.from_numpy(arrays[name]).split(1024)])
    train,test=features('px'),features('tx')
    curve=noise_decodability(train,test,arrays['pz'][:,1],arrays['tz'][:,1],
        levels=protocol['noise_levels'],seed=record['seed'])
    if not all(np.isfinite(row[key]) for row in curve for key in row):
        raise ValueError('Nonfinite production-noise diagnostic')
    return dict(common,status='evaluated',constant_training_representation=curve[0]['training_rms']==0,
        noise_rows=curve)


def close_numeric(actual,expected):
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Noise replay structure changed')
        for key in actual:close_numeric(actual[key],expected[key])
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('Noise replay row count changed')
        for a,b in zip(actual,expected):close_numeric(a,b)
    elif (isinstance(actual,(int,float)) and not isinstance(actual,bool)
          and isinstance(expected,(int,float)) and not isinstance(expected,bool)):
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):
            raise ValueError('Noise replay numeric difference')
    elif type(actual)!=type(expected) or actual!=expected:
        raise ValueError('Noise replay value changed')


def run(repo,output,source_commit,replay=False):
    repo,output=Path(repo).resolve(),Path(output)
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Expected committed noise source identity')
    protocol=load_json((repo/'configs/production_noise_completion_v1.json').read_bytes())
    rows=plan(repo);sources={p.relative_to(repo).as_posix():digest(p) for p in source_files(repo)}
    host=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__)
    cache={};complete=[];torch.set_num_threads(2)
    for row in rows:
        rid=row['record']['run_id'];path=output/(rid+'.json');receipt_path=output/(rid+'.replay.json')
        arrays=load_arrays(repo,row,cache)
        expected,_,_=binding(repo,row,protocol,sources)
        existing=load_json(path.read_bytes()) if path.exists() else None
        if existing is not None and (existing.get('binding')!=expected or existing.get('source_commit')!=source_commit):
            raise ValueError('Different existing immutable production-noise record')
        if replay:
            if existing is None:raise ValueError('Missing production-noise result for replay')
            if existing.get('host')!=host:raise ValueError('Recorded noise replay requires the same host/runtime')
            if receipt_path.exists():
                receipt=load_json(receipt_path.read_bytes())
                if (receipt.get('result_sha256')!=digest(path) or receipt.get('binding')!=expected
                        or receipt.get('host')!=host or receipt.get('replay_verified') is not True):
                    raise ValueError('Different existing production-noise replay')
            else:
                actual=evaluate(repo,row,protocol,sources,arrays)
                close_numeric(actual,{k:v for k,v in existing.items() if k not in {'source_commit','host'}})
                receipt=dict(schema='feature-topology.production-noise-replay.v1',run_id=rid,
                    result_sha256=digest(path),binding=expected,replay_verified=True,
                    numeric_decoder_replay=actual['status']=='evaluated',
                    status='numeric_replay_passed' if actual['status']=='evaluated' else 'unreached_selection_verified',
                    host=host,scope='Same-host repeated numerical evaluation of the same implementation; not an independent verifier or new training replicate')
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
    manifest=dict(schema='feature-topology.production-noise-stage.v1',expected_unique_models=305,
        complete_unique_models=len(complete),stage='replay' if replay else 'generation',
        source_commit=source_commit,protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),records=complete)
    publish(output,{('replay_stage.json' if replay else 'generation_stage.json'):
        (json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()})
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();run(args.repo,args.output,args.source_commit,args.replay)
