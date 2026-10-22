"""Place the unchanged task-fiber calculation on bounded hosted shards."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
EXECUTION='configs/task_fiber_hosted_execution_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_pins(repo):
    plan=json.loads((repo/EXECUTION).read_text())
    if (plan['shards']!=8 or plan['max_parallel']!=4 or plan['math_changed'] is not False
            or plan['statistics_changed'] is not False):raise ValueError('Different hosted execution plan')
    for name,expected in plan['scientific_source_sha256'].items():
        if sha(repo/name)!=expected:raise ValueError('Protected scientific source changed: '+name)
    if sha(repo/'scripts/replay_task_preserving_fibers_v2.py')!=plan['replay_worker_sha256']:
        raise ValueError('Different recorded metadata-only replay fix')
    return plan


def source_identity():
    commit=os.environ.get('GITHUB_SHA','');workflow=os.environ.get('GITHUB_RUN_ID','')
    if not re.fullmatch('[0-9a-f]{40}',commit) or not workflow.isdigit():
        raise ValueError('Hosted source/run identity is required')
    return commit,workflow


def runtime():
    import numpy as np
    import torch
    return dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
        torch=str(torch.__version__),numpy=str(np.__version__),threads=2,dtype='float64')


@contextmanager
def prepared_model(repo,row):
    from scripts.remote_variability_baselines import download
    from scripts.remote_nuisance_evaluation import verify_input
    from scripts.evaluate_task_preserving_fibers import source_files,task_binding
    rid=row['record']['run_id']
    reference=json.loads((repo/'results/production_noise_completion_v1'/(rid+'.json')).read_text())['binding']
    relative=Path(reference['run_directory'])
    if relative.is_absolute() or '..' in relative.parts or not re.fullmatch('[0-9a-f]{16}',relative.name):
        raise ValueError('Unsafe archived canonical model directory')
    destination=repo/relative
    if destination.exists() or destination.is_symlink():raise ValueError('Hosted input destination already exists')
    alias=None
    with tempfile.TemporaryDirectory(prefix='task-fiber-input-') as temp:
        temporary=Path(temp);archive=temporary/'input.tar.gz'
        download(row['input_tag'],row['record']['asset'],archive)
        directory=verify_input(archive,row['record'],temporary/'verified')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(directory,destination)
        try:
            if row['cohort']=='primary' and relative!=Path('results/main/runs')/rid:
                alias=repo/'results/main/runs'/rid;alias.parent.mkdir(parents=True,exist_ok=True)
                if alias.exists() or alias.is_symlink():raise ValueError('Hosted logical model alias already exists')
                alias.symlink_to(os.path.relpath(destination,alias.parent),target_is_directory=True)
            sources={p.relative_to(repo).as_posix():sha(p) for p in source_files(repo)}
            actual,_,_=task_binding(repo,row,sources)
            for key,value in reference.items():
                if key!='source_sha256' and actual.get(key)!=value:
                    raise ValueError('Hosted source differs from the checked canonical checkpoint: '+key)
            yield actual
        finally:
            if alias is not None and alias.is_symlink():alias.unlink()
            shutil.rmtree(destination)


def cell(repo,output,rid,replay):
    import torch
    from scripts.evaluate_task_preserving_fibers import task_plan,source_files,task_binding,evaluate
    from scripts.evaluate_production_noise import load_arrays
    from scripts.replay_task_preserving_fibers_v2 import compare_result,POLICY
    from scripts.recover_release_evidence import load_json,publish
    verify_pins(repo);source,_=source_identity();rows,protocol=task_plan(repo)
    selected=[r for r in rows if r['record']['run_id']==rid]
    if len(selected)!=1:raise ValueError('Unplanned hosted task-fiber model')
    row=selected[0];sources={p.relative_to(repo).as_posix():sha(p) for p in source_files(repo)}
    binding,_,_=task_binding(repo,row,sources);host=runtime();torch.set_num_threads(2)
    path=output/(rid+'.json')
    actual=evaluate(repo,row,protocol,sources,load_arrays(repo,row,{}))
    if replay:
        stored=load_json(path.read_bytes())
        if stored['binding']!=binding:raise ValueError('Different hosted replay model binding')
        compare_result(actual,stored,source,host)
        receipt=dict(schema='feature-topology.task-preserving-fiber-replay.v2',run_id=rid,
            result_sha256=sha(path),binding=binding,host=host,replay_verified=True,
            numeric_replay=stored['status']=='evaluated',replay_source_commit=source,replay_policy=POLICY,
            replay_worker_sha256=sha(repo/'scripts/replay_task_preserving_fibers_v2.py'),
            scope='Same-host repeated numerical evaluation; not an independent verifier or new training replicate')
        publish(output,{rid+'.replay.json':(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()})
    else:
        actual.update(source_commit=source,host=host)
        publish(output,{rid+'.json':(json.dumps(actual,indent=2,allow_nan=False)+'\n').encode()})


def shard(repo,output,index):
    from scripts.evaluate_task_preserving_fibers import task_plan
    from scripts.collect_production_noise import collect as noise_collection
    from scripts.recover_release_evidence import publish
    plan=verify_pins(repo);source,workflow=source_identity()
    if not 0<=index<plan['shards']:raise ValueError('Invalid hosted shard')
    # Canonical paths above come only from the already source-checked collection.
    if noise_collection(repo,repo/'results/production_noise_completion_v1')!=json.loads(
            (repo/'results/production_noise_completion_v1/collection.json').read_text()):
        raise ValueError('The reference production-noise collection changed')
    rows=[r for r in task_plan(repo)[0] if int(r['record']['run_id'],16)%plan['shards']==index]
    completed=[]
    for row in rows:
        rid=row['record']['run_id']
        with prepared_model(repo,row) as binding:
            base=[sys.executable,str(Path(__file__).resolve()),'--repo',str(repo),'--output',str(output),'--cell',rid]
            subprocess.run(base,check=True);subprocess.run(base+['--replay'],check=True)
        receipt=dict(schema='feature-topology.task-fiber-hosted-execution-receipt.v1',run_id=rid,
            source_commit=source,workflow_run=workflow,shard=index,shards=plan['shards'],
            result_sha256=sha(output/(rid+'.json')),replay_sha256=sha(output/(rid+'.replay.json')),
            input_archive_sha256=row['record']['archive_sha256'],canonical_directory=binding['run_directory'],
            execution_plan_sha256=sha(repo/EXECUTION),
            orchestration_sha256=sha(Path(__file__)),scope=plan['scope'])
        payload=(json.dumps(receipt,indent=2)+'\n').encode()
        publish(output,{'execution/'+rid+'.json':payload})
        completed.append(dict(run_id=rid,result_sha256=receipt['result_sha256'],replay_sha256=receipt['replay_sha256'],
            execution_sha256=hashlib.sha256(payload).hexdigest()))
        print(json.dumps(dict(completed=len(completed),expected=len(rows),run_id=rid,shard=index)),flush=True)
    manifest=dict(schema='feature-topology.task-fiber-hosted-shard.v1',shard=index,shards=plan['shards'],
        source_commit=source,workflow_run=workflow,execution_plan_sha256=sha(repo/EXECUTION),
        expected=[r['record']['run_id'] for r in rows],complete=completed)
    publish(output,{'shard_manifest.json':(json.dumps(manifest,indent=2)+'\n').encode()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True);parser.add_argument('--shard',type=int)
    parser.add_argument('--cell');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    repo,output=Path(args.repo).resolve(),Path(args.output).resolve()
    if args.cell:cell(repo,output,args.cell,args.replay)
    elif args.shard is not None:shard(repo,output,args.shard)
    else:parser.error('Choose --shard or --cell')
