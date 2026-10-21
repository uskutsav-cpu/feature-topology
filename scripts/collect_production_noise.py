"""Validate every planned production noise record and its same-runtime replay."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_production_noise import plan,source_files
from scripts.recover_release_evidence import digest,load_json,publish


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_cell(repo,row,result,receipt,result_sha,sources,protocol):
    record=row['record'];rid=record['run_id'];b=result.get('binding',{})
    directory=b.get('run_directory','')
    parts=Path(directory).parts
    valid_directory=(directory in {f'results/main/runs/{rid}',f'results/pilot/runs/{rid}'} if row['cohort']=='primary' else
        len(parts)==5 and parts[:2]==('results','relevance') and parts[3:]==('runs',rid)
        and re.fullmatch('[0-9a-f]{16}',parts[2]) is not None)
    if not valid_directory or Path(directory).is_absolute() or '..' in parts:
        raise ValueError('Noise source run directory changed or escaped')
    selected={key:b.get(key) for key in ['checkpoint','checkpoint_sha256','selected_step','actual_training_loss']}
    expected=dict(run_id=rid,gamma=record['gamma'],seed=record['seed'],relevance=row['relevance'],roles=row['roles'],
        run_directory=directory,input_index=row['index'],input_index_sha256=row['index_sha256'],
        input_archive_sha256=record['archive_sha256'],config_sha256=record['files']['config.json'],
        training_summary_sha256=record['files']['summary.json'],analysis_data_path=row['array_path'],
        analysis_data_sha256=row['analysis_data']['sha256'],**selected,
        protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),source_sha256=sources)
    if (result.get('schema')!='feature-topology.production-noise-evaluation.v1' or b!=expected
            or result.get('noise_levels')!=protocol['noise_levels']
            or result.get('information_destruction_proved') is not False or result.get('scope')!=protocol['scope']
            or result.get('source_training_status') not in {'converged','budget_exhausted','diverged'}
            or ('status' in record and result.get('source_training_status')!=record['status'])):
        raise ValueError('Noise source, protocol, scope or training identity changed')
    evaluated=result.get('status')=='evaluated'
    minimum=result.get('minimum_stored_training_loss')
    if evaluated:
        if (selected['checkpoint'] not in record['files']
                or record['files'][selected['checkpoint']]!=selected['checkpoint_sha256']
                or type(selected['selected_step']) is not int or selected['selected_step']<0
                or not finite(selected['actual_training_loss']) or not 0<=selected['actual_training_loss']<=.1
                or not finite(minimum) or not 0<=minimum<=selected['actual_training_loss']
                or type(result.get('constant_training_representation')) is not bool):
            raise ValueError('Noise matched-risk checkpoint binding changed')
        name=selected['checkpoint']
        if name!='final.pt' and name!=f"step_{selected['selected_step']:07d}.pt":
            raise ValueError('Noise checkpoint filename and step differ')
        curve=result.get('noise_rows',[])
        if len(curve)!=4:raise ValueError('Incomplete noise-level curve')
        rms=curve[0].get('training_rms')
        for level,values in zip(protocol['noise_levels'],curve):
            if (set(values)!={'noise_level','training_rms','coordinate_noise_sd','angular_cosine','angular_mae'}
                    or not all(finite(value) for value in values.values()) or values['noise_level']!=level
                    or values['training_rms']!=rms or rms<0 or not -1<=values['angular_cosine']<=1
                    or not 0<=values['angular_mae']<=math.pi
                    or not math.isclose(values['coordinate_noise_sd'],level*rms/math.sqrt(256),rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError('Noise curve/scaling is invalid or differs from the recorded method')
        if result['constant_training_representation'] is not (rms==0):
            raise ValueError('Noise constant-representation status changed')
    elif (result.get('status')!='unreached_common_risk' or any(v is not None for v in selected.values())
            or result.get('source_training_status')=='converged' or result.get('noise_rows')!=[]
            or result.get('constant_training_representation') is not None
            or (minimum is not None and (not finite(minimum) or minimum<=.1))):
        raise ValueError('Unreached noise risk was substituted or contradicted')
    if (receipt.get('schema')!='feature-topology.production-noise-replay.v1' or receipt.get('run_id')!=rid
            or receipt.get('result_sha256')!=result_sha or receipt.get('binding')!=b
            or receipt.get('host')!=result.get('host') or not isinstance(result.get('host'),dict)
            or receipt.get('replay_verified') is not True or receipt.get('numeric_decoder_replay') is not evaluated
            or receipt.get('status')!=('numeric_replay_passed' if evaluated else 'unreached_selection_verified')
            or receipt.get('scope')!='Same-host repeated numerical evaluation of the same implementation; not an independent verifier or new training replicate'):
        raise ValueError('Missing or different production-noise replay receipt')


def collect(repo,artifacts,output=None,source_commit=None):
    repo,artifacts=Path(repo).resolve(),Path(artifacts)
    rows=plan(repo);protocol=load_json((repo/'configs/production_noise_completion_v1.json').read_bytes())
    sources={p.relative_to(repo).as_posix():digest(p) for p in source_files(repo)}
    required={row['record']['run_id']+suffix for row in rows for suffix in ['.json','.replay.json']}
    required|={'generation_stage.json','replay_stage.json'}
    present={p.name for p in artifacts.glob('*.json')}
    if not required<=present or present-required-{'collection.json'}:
        raise ValueError('Production-noise collection incomplete or has unplanned records')
    files={};records=[];stages=[];commits=set();arrays={}
    for row in rows:
        rid=row['record']['run_id'];path=artifacts/(rid+'.json');replay=artifacts/(rid+'.replay.json')
        result,receipt=load_json(path.read_bytes()),load_json(replay.read_bytes())
        commit=result.get('source_commit','')
        if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit) or (source_commit is not None and commit!=source_commit):
            raise ValueError('Different production-noise source commit')
        commits.add(commit)
        validate_cell(repo,row,result,receipt,digest(path),sources,protocol)
        b=result['binding'];arrays[b['analysis_data_path']]=b['analysis_data_sha256']
        records.append(dict(run_id=rid,gamma=b['gamma'],seed=b['seed'],relevance=b['relevance'],roles=b['roles'],
            status=result['status'],checkpoint=b['checkpoint'],checkpoint_sha256=b['checkpoint_sha256'],
            selected_step=b['selected_step'],actual_training_loss=b['actual_training_loss'],
            constant_training_representation=result['constant_training_representation'],noise_rows=result['noise_rows'],
            result_sha256=digest(path),replay_sha256=digest(replay)))
        stages.append(dict(run_id=rid,status=result['status'],result_sha256=digest(path)))
        files[path.name],files[replay.name]=path.read_bytes(),replay.read_bytes()
    if len(commits)!=1:raise ValueError('Production-noise records mix source commits')
    commit=next(iter(commits))
    for stage,name in [('generation','generation_stage.json'),('replay','replay_stage.json')]:
        expected=dict(schema='feature-topology.production-noise-stage.v1',expected_unique_models=305,
            complete_unique_models=305,stage=stage,source_commit=commit,
            protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),records=stages)
        if load_json((artifacts/name).read_bytes())!=expected:
            raise ValueError('Production-noise stage accounting changed or is incomplete')
        files[name]=(artifacts/name).read_bytes()
    evaluated=sum(row['status']=='evaluated' for row in records)
    manifest=dict(schema='feature-topology.production-noise-collection.v1',expected_unique_models=305,
        complete_unique_models=305,main_models=130,relevance_models=210,shared_models=35,
        evaluated_models=evaluated,unreached_models=305-evaluated,noise_levels=protocol['noise_levels'],
        numeric_replays=evaluated,selection_only_replays=305-evaluated,source_commit=commit,
        protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),analysis_data_sha256=arrays,
        records=records,scope=protocol['scope'],
        interpretation='All specified saved models accounted for; noise levels and probe draws are not additional training-seed replicates. No information-destruction or independent-verifier claim.')
    files['collection.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    if output is not None:publish(Path(output),files)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--artifacts',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();result=collect(args.repo,args.artifacts,args.output,args.source_commit)
    print(json.dumps({k:result[k] for k in ['complete_unique_models','evaluated_models','unreached_models','numeric_replays']}))
