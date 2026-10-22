"""Replay unchanged task-fiber mathematics with canonical JSON runtime identity."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.evaluate_task_preserving_fibers import (PROTOCOL,source_files,task_binding,task_plan,evaluate)
from scripts.evaluate_production_noise import close_numeric,load_arrays
from scripts.recover_release_evidence import digest,load_json,publish

POLICY='Compare canonical JSON runtime identity separately; unchanged mathematical fields and original numeric tolerances'


def compare_result(actual,stored,source_commit,host):
    if (stored.get('source_commit')!=source_commit
            or json.dumps(stored.get('host'),sort_keys=True)!=json.dumps(host,sort_keys=True)):
        raise ValueError('Different task-fiber generation source or exact serialized runtime identity')
    close_numeric(actual,{k:v for k,v in stored.items() if k not in {'source_commit','host'}})


def run(repo,output,source_commit,replay_source_commit):
    repo,output=Path(repo).resolve(),Path(output)
    if not all(re.fullmatch('[0-9a-f]{40}',v) for v in [source_commit,replay_source_commit]):
        raise ValueError('Expected committed generation/replay identities')
    rows,protocol=task_plan(repo);sources={p.relative_to(repo).as_posix():digest(p) for p in source_files(repo)}
    host=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
        torch=str(torch.__version__),numpy=str(np.__version__),threads=2,dtype='float64')
    worker_sha=digest(repo/'scripts/replay_task_preserving_fibers_v2.py')
    cache={};complete=[];torch.set_num_threads(2)
    for row in rows:
        rid=row['record']['run_id'];path=output/(rid+'.json');receipt_path=output/(rid+'.replay.json')
        stored=load_json(path.read_bytes());expected,_,_=task_binding(repo,row,sources)
        if stored['binding']!=expected:raise ValueError('Different task-fiber source binding')
        receipt=dict(schema='feature-topology.task-preserving-fiber-replay.v2',run_id=rid,
            result_sha256=digest(path),binding=expected,host=host,replay_verified=True,
            numeric_replay=stored['status']=='evaluated',replay_worker_sha256=worker_sha,
            replay_source_commit=replay_source_commit,replay_policy=POLICY,
            scope='Same-host repeated numerical evaluation; not an independent verifier or new training replicate')
        if receipt_path.exists():
            if load_json(receipt_path.read_bytes())!=receipt:raise ValueError('Different immutable task-fiber V2 replay receipt')
        else:
            actual=evaluate(repo,row,protocol,sources,load_arrays(repo,row,cache))
            compare_result(actual,stored,source_commit,host)
            publish(output,{receipt_path.name:(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()})
        complete.append(dict(run_id=rid,status=stored['status'],result_sha256=digest(path)))
        print(json.dumps(dict(completed=len(complete),run_id=rid,gamma=row['record']['gamma'],
            seed=row['record']['seed'],relevance=row['relevance'],status=stored['status'],replay=True)),flush=True)
    manifest=dict(schema='feature-topology.task-preserving-fiber-stage.v1',expected_models=210,
        complete_models=len(complete),stage='replay',source_commit=source_commit,
        protocol_sha256=digest(repo/PROTOCOL),records=complete,replay_source_commit=replay_source_commit,
        replay_worker_sha256=worker_sha,replay_policy=POLICY)
    publish(output,{'replay_stage.json':(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()})
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--replay-source-commit',required=True);args=parser.parse_args()
    run(args.repo,args.output,args.source_commit,args.replay_source_commit)
