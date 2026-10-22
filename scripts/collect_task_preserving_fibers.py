"""Strict accounting of every recorded task-preserving model and numerical replay."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_task_preserving_fibers import PROTOCOL,source_files,task_plan
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.replay_task_preserving_fibers_v2 import POLICY


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def near(actual,expected):
    return finite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12)


def norm_summary(values,scale):
    if (set(values)!={'minimum','q01','median','normalized_minimum','normalized_q01'}
            or not all(finite(values[key]) and values[key]>=0 for key in ['minimum','q01','median'])
            or not values['minimum']<=values['q01']<=values['median']):
        raise ValueError('Invalid finite task-fiber margin summary')
    for key in ['minimum','q01']:
        if ((scale>0 and not near(values['normalized_'+key],values[key]/scale))
                or (scale==0 and values['normalized_'+key] is not None)):
            raise ValueError('Task-fiber margin normalization changed')


def validate_family(family,level,task,protocol):
    expected_name='fixed_u_task_preserving' if task else 'fixed_theta_original_factor_control'
    scale=family.get('centered_representation_rms');dense=family.get('dense_pairs',{})
    error=family.get('task_coordinate_maximum_error')
    if (family.get('family')!=expected_name or family.get('grid_side')!=100 or family.get('grid_points')!=10000
            or not finite(scale) or scale<0 or not finite(family.get('classifier_grid_accuracy'))
            or not 0<=family['classifier_grid_accuracy']<=1
            or type(family.get('exact_zero_preactivations')) is not int
            or not 0<=family['exact_zero_preactivations']<=10000*4*256
            or (task and (not finite(error) or not 0<=error<=1e-12)) or (not task and error is not None)
            or dense.get('pairs')!=495000 or dense.get('collision_tolerance')!=1e-10
            or not finite(dense.get('minimum_representation_distance')) or dense['minimum_representation_distance']<0
            or type(dense.get('exact_float_collision_pairs')) is not int
            or type(dense.get('tolerance_collision_pairs')) is not int
            or not 0<=dense['exact_float_collision_pairs']<=dense['tolerance_collision_pairs']<=495000
            or dense.get('qualification')!='Finite grid only; no continuum lower bound or real-valued collision certificate'):
        raise ValueError('Task-fiber grid, family, scope or collision accounting changed')
    for key in ['local_per_phi','local_per_unit_flat_torus_tangent','local_per_unit_embedded_input_tangent']:
        norm_summary(family[key],scale)
    for key in ['phi_arc','flat_torus_ambient']:norm_summary(dense[key],scale)
    witness=dense['minimum_phi_ratio_witness']
    if (type(witness.get('base_index')) is not int or not 0<=witness['base_index']<100
            or len(witness.get('pair_indices',[]))!=2
            or any(type(i) is not int for i in witness['pair_indices'])
            or not 0<=witness['pair_indices'][0]<witness['pair_indices'][1]<100
            or not near(witness.get('phi_arc_ratio'),dense['phi_arc']['minimum'])
            or not finite(witness.get('representation_distance')) or witness['representation_distance']<0):
        raise ValueError('Task-fiber dense minimum witness changed')
    expected_latent=[];base=(witness['base_index']+.5)*2*math.pi/100
    for index in witness['pair_indices']:
        phi=(index+.5)*2*math.pi/100
        expected_latent.append([base-level*math.sin(phi) if task else base,phi])
    latent=witness.get('latent_pair',[])
    if (len(latent)!=2 or any(len(z)!=2 for z in latent)
            or any(not near(a,b) for actual,expected in zip(latent,expected_latent) for a,b in zip(actual,expected))):
        raise ValueError('Dense witness left the declared fiber/grid')
    arc=abs(latent[1][1]-latent[0][1])%(2*math.pi);arc=min(arc,2*math.pi-arc)
    if not near(witness['phi_arc_ratio'],witness['representation_distance']/arc):
        raise ValueError('Dense minimum witness ratio changed')
    search=family.get('adversarial_search',{});candidate=search.get('best_candidate')
    if scale==0:
        if search!=dict(status='constant_grid_scale',evaluated_candidates=0,best_candidate=None):
            raise ValueError('Constant-grid search division was not retained explicitly')
        return
    if (search.get('status')!='fixed_budget_complete' or search.get('evaluated_candidates')!=1040
            or search.get('qualification')!='Smallest evaluated separated-pair ratio; upper bound on an infimum, not a certificate'
            or not isinstance(candidate,dict) or type(candidate.get('update')) is not int
            or not 0<=candidate['update']<=64 or type(candidate.get('start')) is not int
            or not 0<=candidate['start']<16):
        raise ValueError('Adversarial search budget or evidence scope changed')
    b=candidate.get('base');arc=candidate.get('phi_arc');z=candidate.get('latent_pair',[])
    if (not finite(b) or not finite(arc) or not math.pi/8<=arc<=math.pi
            or len(z)!=2 or any(len(p)!=2 or not all(finite(v) for v in p) for p in z)
            or not near(z[1][1]-z[0][1],arc)
            or any(not near(p[0]+level*math.sin(p[1]) if task else p[0],b) for p in z)):
        raise ValueError('Adversarial candidate left the exact same-base separated-pair domain')
    distance=candidate.get('representation_distance')
    delta=[min(abs(a-c)%(2*math.pi),2*math.pi-abs(a-c)%(2*math.pi)) for a,c in zip(*z)]
    ambient=math.hypot(*delta)
    if (not finite(distance) or distance<0
            or not near(candidate.get('phi_arc_ratio'),distance/arc)
            or not near(candidate.get('normalized_phi_arc_ratio'),distance/(arc*scale))
            or not near(candidate.get('flat_torus_ambient_ratio'),distance/ambient)
            or not near(candidate.get('normalized_flat_torus_ambient_ratio'),distance/(ambient*scale))):
        raise ValueError('Adversarial ratio/scaling changed')
    coords=[p[0]+level*math.sin(p[1]) for p in z]
    task_delta=abs(coords[0]-coords[1])%(2*math.pi);task_delta=min(task_delta,2*math.pi-task_delta)
    if (len(candidate.get('task_coordinate_pair',[]))!=2
            or any(not near(a,c) for a,c in zip(candidate['task_coordinate_pair'],coords))
            or not near(candidate.get('task_coordinate_wrapped_difference'),task_delta)
            or (task and task_delta>1e-12)):
        raise ValueError('Adversarial task-coordinate identity changed')


def validate_cell(repo,row,result,receipt,result_sha,sources,protocol):
    record=row['record'];rid=record['run_id'];b=result.get('binding',{});directory=b.get('run_directory','')
    parts=Path(directory).parts
    valid=(directory in {f'results/main/runs/{rid}',f'results/pilot/runs/{rid}'} if row['cohort']=='primary'
        else len(parts)==5 and parts[:2]==('results','relevance') and parts[3:]==('runs',rid)
        and re.fullmatch('[0-9a-f]{16}',parts[2]) is not None)
    if not valid:raise ValueError('Task-fiber source run directory changed or escaped')
    selected={key:b.get(key) for key in ['checkpoint','checkpoint_sha256','selected_step','actual_training_loss']}
    expected=dict(run_id=rid,gamma=record['gamma'],seed=record['seed'],relevance=row['relevance'],roles=row['roles'],
        run_directory=directory,input_index=row['index'],input_index_sha256=row['index_sha256'],
        input_archive_sha256=record['archive_sha256'],config_sha256=record['files']['config.json'],
        training_summary_sha256=record['files']['summary.json'],analysis_data_path=row['array_path'],
        analysis_data_sha256=row['analysis_data']['sha256'],**selected,
        protocol_sha256=digest(repo/'configs/production_noise_completion_v1.json'),source_sha256=sources,
        task_preserving_protocol_sha256=digest(repo/PROTOCOL))
    if (result.get('schema')!='feature-topology.task-preserving-fiber-evaluation.v1' or b!=expected
            or result.get('continuum_certificate') is not False or result.get('scope')!=protocol['scope']
            or result.get('source_training_status') not in {'converged','budget_exhausted','diverged'}
            or ('status' in record and result.get('source_training_status')!=record['status'])):
        raise ValueError('Task-fiber source, protocol or proof scope changed')
    evaluated=result.get('status')=='evaluated'
    if evaluated:
        if (selected['checkpoint'] not in record['files']
                or record['files'][selected['checkpoint']]!=selected['checkpoint_sha256']
                or type(selected['selected_step']) is not int or selected['selected_step']<0
                or not finite(selected['actual_training_loss']) or not 0<=selected['actual_training_loss']<=.1
                or (selected['checkpoint']!='final.pt' and selected['checkpoint']!=f"step_{selected['selected_step']:07d}.pt")
                or len(result.get('families',[]))!=2):
            raise ValueError('Task-fiber selected risk/checkpoint or paired families changed')
        for family,task in zip(result['families'],[True,False]):validate_family(family,row['relevance'],task,protocol)
        if row['relevance']==0:
            left,right=copy.deepcopy(result['families'])
            for value in [left,right]:
                value.pop('family');value.pop('task_coordinate_maximum_error')
            if left!=right:raise ValueError('Lambda-zero family computations must coincide exactly')
    elif (result.get('status')!='unreached_common_risk' or result.get('families')!=[]
            or any(v is not None for v in selected.values()) or result.get('source_training_status')=='converged'):
        raise ValueError('Unreached task-fiber risk was substituted')
    host=result.get('host',{})
    replay_commit=receipt.get('replay_source_commit','')
    expected_receipt=dict(schema='feature-topology.task-preserving-fiber-replay.v2',run_id=rid,
        result_sha256=result_sha,binding=b,host=host,replay_verified=True,numeric_replay=evaluated,
        replay_source_commit=replay_commit,replay_policy=POLICY,
        replay_worker_sha256=digest(repo/'scripts/replay_task_preserving_fibers_v2.py'),
        scope='Same-host repeated numerical evaluation; not an independent verifier or new training replicate')
    if (receipt!=expected_receipt or host.get('dtype')!='float64' or host.get('threads')!=2
            or not isinstance(replay_commit,str) or not re.fullmatch('[0-9a-f]{40}',replay_commit)
            or not all(isinstance(host.get(k),str) and host[k] for k in ['system','machine','python','torch','numpy'])):
        raise ValueError('Task-fiber same-host numerical replay binding changed')


def collect(repo,artifacts,output=None,source_commit=None):
    repo,artifacts=Path(repo).resolve(),Path(artifacts);rows,protocol=task_plan(repo)
    sources={p.relative_to(repo).as_posix():digest(p) for p in source_files(repo)}
    required={r['record']['run_id']+suffix for r in rows for suffix in ['.json','.replay.json']}
    required|={'generation_stage.json','replay_stage.json'}
    present={p.name for p in artifacts.glob('*.json')}
    if not required<=present or present-required-{'collection.json'}:raise ValueError('Task-fiber collection incomplete or unplanned')
    files={};records=[];stages=[];commits=set();replay_commits=set();arrays={}
    for row in rows:
        rid=row['record']['run_id'];path=artifacts/(rid+'.json');replay=artifacts/(rid+'.replay.json')
        result,receipt=load_json(path.read_bytes()),load_json(replay.read_bytes());commit=result.get('source_commit','')
        if (not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit)
                or (source_commit is not None and commit!=source_commit)):raise ValueError('Different task-fiber source commit')
        commits.add(commit);validate_cell(repo,row,result,receipt,digest(path),sources,protocol);b=result['binding']
        replay_commits.add(receipt['replay_source_commit'])
        arrays[b['analysis_data_path']]=b['analysis_data_sha256']
        records.append(dict(run_id=rid,gamma=b['gamma'],seed=b['seed'],relevance=b['relevance'],status=result['status'],
            checkpoint=b['checkpoint'],checkpoint_sha256=b['checkpoint_sha256'],selected_step=b['selected_step'],
            actual_training_loss=b['actual_training_loss'],families=result['families'],
            result_sha256=digest(path),replay_sha256=digest(replay)))
        stages.append(dict(run_id=rid,status=result['status'],result_sha256=digest(path)))
        files[path.name],files[replay.name]=path.read_bytes(),replay.read_bytes()
    if len(commits)!=1 or len(replay_commits)!=1:raise ValueError('Mixed task-fiber generation/replay source commits')
    commit=next(iter(commits));replay_commit=next(iter(replay_commits))
    for stage in ['generation','replay']:
        name=stage+'_stage.json';expected=dict(schema='feature-topology.task-preserving-fiber-stage.v1',
            expected_models=210,complete_models=210,stage=stage,source_commit=commit,
            protocol_sha256=digest(repo/PROTOCOL),records=stages)
        if stage=='replay':expected.update(replay_source_commit=replay_commit,
            replay_worker_sha256=digest(repo/'scripts/replay_task_preserving_fibers_v2.py'),replay_policy=POLICY)
        if load_json((artifacts/name).read_bytes())!=expected:raise ValueError('Incomplete task-fiber stage accounting')
        files[name]=(artifacts/name).read_bytes()
    evaluated=sum(r['status']=='evaluated' for r in records)
    shared=sum(r['status']=='evaluated' and r['relevance']==0 for r in records)
    manifest=dict(schema='feature-topology.task-preserving-fiber-collection.v1',expected_models=210,complete_models=210,
        evaluated_models=evaluated,unreached_models=210-evaluated,numeric_replays=evaluated,
        selection_only_replays=210-evaluated,training_seeds=list(range(5)),source_commit=commit,
        replay_source_commit=replay_commit,replay_policy=POLICY,
        paired_family_views=2*evaluated,distinct_family_evaluations=2*evaluated-shared,
        lambda_zero_family_reuse=shared,
        protocol_sha256=digest(repo/PROTOCOL),analysis_data_sha256=arrays,records=records,scope=protocol['scope'])
    files['collection.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    if output is not None:publish(Path(output),files)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--artifacts',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--source-commit',required=True);args=parser.parse_args()
    result=collect(args.repo,args.artifacts,args.output,args.source_commit)
    print(json.dumps({k:result[k] for k in ['complete_models','evaluated_models','unreached_models','numeric_replays']}))
