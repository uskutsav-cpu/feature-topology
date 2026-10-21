"""Collect all hosted task-fiber shards with exact execution provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_task_preserving_fibers import collect
from scripts.evaluate_task_preserving_fibers import PROTOCOL,task_plan
from scripts.remote_task_preserving_completion import EXECUTION,verify_pins
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.replay_task_preserving_fibers_v2 import POLICY


def expected_ids(repo,shard):
    return [r['record']['run_id'] for r in task_plan(repo)[0] if int(r['record']['run_id'],16)%8==shard]


def validate_execution(repo,root,workflow_run=None,source_commit=None):
    repo,root=Path(repo),Path(root);plan=verify_pins(repo);rows={r['record']['run_id']:r for r in task_plan(repo)[0]}
    first=load_json((root/'execution/shards/shard-0.json').read_bytes())
    workflow=str(workflow_run) if workflow_run is not None else first.get('workflow_run','')
    source=source_commit if source_commit is not None else first.get('source_commit','')
    if not isinstance(workflow,str) or not workflow.isdigit() or not isinstance(source,str) or not re.fullmatch('[0-9a-f]{40}',source):
        raise ValueError('Invalid hosted task-fiber source/workflow identity')
    consumed={};expected_execution={'audit.json'};seen=[]
    for shard in range(8):
        name=f'execution/shards/shard-{shard}.json';path=root/name;manifest=load_json(path.read_bytes());ids=expected_ids(repo,shard)
        header=dict(schema='feature-topology.task-fiber-hosted-shard.v1',shard=shard,shards=8,
            source_commit=source,workflow_run=workflow,execution_plan_sha256=digest(repo/EXECUTION),expected=ids)
        if ({k:v for k,v in manifest.items() if k!='complete'}!=header or len(manifest.get('complete',[]))!=len(ids)):
            raise ValueError('Incomplete or different hosted task-fiber shard')
        consumed[name]=digest(path);expected_execution.add(f'shards/shard-{shard}.json')
        for rid,record in zip(ids,manifest['complete']):
            result_path=root/(rid+'.json');replay_path=root/(rid+'.replay.json');execution_path=root/'execution'/(rid+'.json')
            result,replay=load_json(result_path.read_bytes()),load_json(replay_path.read_bytes())
            expected_record=dict(run_id=rid,result_sha256=digest(result_path),replay_sha256=digest(replay_path),
                execution_sha256=digest(execution_path))
            if (record!=expected_record or result.get('source_commit')!=source or replay.get('replay_source_commit')!=source):
                raise ValueError('Hosted task-fiber result/replay source or checksum changed')
            expected=dict(schema='feature-topology.task-fiber-hosted-execution-receipt.v1',run_id=rid,
                source_commit=source,workflow_run=workflow,shard=shard,shards=8,
                result_sha256=expected_record['result_sha256'],replay_sha256=expected_record['replay_sha256'],
                input_archive_sha256=rows[rid]['record']['archive_sha256'],
                canonical_directory=result['binding']['run_directory'],execution_plan_sha256=digest(repo/EXECUTION),
                orchestration_sha256=digest(repo/'scripts/remote_task_preserving_completion.py'),scope=plan['scope'])
            if load_json(execution_path.read_bytes())!=expected:raise ValueError('Different hosted task-fiber execution receipt')
            consumed[rid+'.json']=expected_record['result_sha256'];consumed[rid+'.replay.json']=expected_record['replay_sha256']
            consumed['execution/'+rid+'.json']=expected_record['execution_sha256'];expected_execution.add(rid+'.json');seen.append(rid)
    if len(seen)!=210 or len(set(seen))!=210 or set(seen)!=set(rows):raise ValueError('Hosted task-fiber cohort incomplete or duplicated')
    actual={p.relative_to(root/'execution').as_posix() for p in (root/'execution').rglob('*') if p.is_file()}
    if actual-expected_execution:raise ValueError('Unexpected hosted task-fiber execution evidence')
    return dict(schema='feature-topology.task-fiber-hosted-execution-audit.v1',complete_models=210,
        complete_shards=8,workflow_run=workflow,source_commit=source,
        execution_plan_sha256=digest(repo/EXECUTION),scientific_source_sha256=plan['scientific_source_sha256'],
        collection_validator_sha256=digest(repo/'scripts/collect_hosted_task_fibers.py'),
        input_sha256=consumed,scope=plan['scope'])


def collect_hosted(repo,artifacts,output,workflow_run,source_commit):
    repo,artifacts,output=Path(repo).resolve(),Path(artifacts).resolve(),Path(output).resolve()
    manifests=list(artifacts.rglob('shard_manifest.json'))
    if len(manifests)!=8:raise ValueError('Expected exactly eight complete hosted task-fiber artifacts')
    payloads={};seen=set()
    for manifest_path in manifests:
        value=load_json(manifest_path.read_bytes());shard=value.get('shard')
        if type(shard) is not int or not 0<=shard<8 or shard in seen:raise ValueError('Duplicate or unexpected hosted shard')
        seen.add(shard);ids=expected_ids(repo,shard)
        if (value.get('expected')!=ids or value.get('source_commit')!=source_commit
                or value.get('workflow_run')!=str(workflow_run)):raise ValueError('Different hosted shard campaign')
        folder=manifest_path.parent;required={'shard_manifest.json'}
        required.update(rid+suffix for rid in ids for suffix in ['.json','.replay.json'])
        required.update('execution/'+rid+'.json' for rid in ids)
        present={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        if present!=required:raise ValueError('Hosted shard files incomplete or unplanned')
        for name in required:
            path=folder/name
            if not path.resolve().is_relative_to(artifacts):raise ValueError('Hosted artifact path escaped')
            target=f'execution/shards/shard-{shard}.json' if name=='shard_manifest.json' else name
            if target in payloads:raise ValueError('Hosted artifacts duplicate a model')
            payloads[target]=path.read_bytes()
    records=[]
    for row in task_plan(repo)[0]:
        rid=row['record']['run_id'];result=load_json(payloads[rid+'.json'])
        records.append(dict(run_id=rid,status=result['status'],result_sha256=hashlib.sha256(payloads[rid+'.json']).hexdigest()))
    for stage in ['generation','replay']:
        value=dict(schema='feature-topology.task-preserving-fiber-stage.v1',expected_models=210,
            complete_models=210,stage=stage,source_commit=source_commit,protocol_sha256=digest(repo/PROTOCOL),records=records)
        if stage=='replay':value.update(replay_source_commit=source_commit,
            replay_worker_sha256=digest(repo/'scripts/replay_task_preserving_fibers_v2.py'),replay_policy=POLICY)
        payloads[stage+'_stage.json']=(json.dumps(value,indent=2)+'\n').encode()
    publish(output,payloads)
    result=collect(repo,output,output,source_commit=source_commit)
    audit=validate_execution(repo,output,workflow_run,source_commit)
    publish(output,{'execution/audit.json':(json.dumps(audit,indent=2)+'\n').encode()})
    return dict(complete_models=result['complete_models'],numeric_replays=result['numeric_replays'],
        workflow_run=str(workflow_run),source_commit=source_commit,full_study_frozen=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--artifacts',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workflow-run',required=True);parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();print(json.dumps(collect_hosted(args.repo,args.artifacts,args.output,args.workflow_run,args.source_commit)))
