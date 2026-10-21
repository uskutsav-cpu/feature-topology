"""Reconstruct explicit CIFAR source provenance from authenticated public evidence.

This never claims the identity of an unavailable local controller ledger and
never validates training/tensors or marks a production collection complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.remote_cifar import cell_id,config_for,parse_cell
from scripts.recover_release_evidence import digest,load_json,publish

REPOSITORY='uskutsav-cpu/feature-topology'
STATUS=re.compile(r'status-([0-9a-f]{16})-([0-9]+)-([0-9]+)\.json')


def descriptor(cell):
    return (cell['stage'],cell['dataset'],cell['gamma'],
        cell['multiplier'] if cell['stage']=='calibration' else cell['seed'])


def job_descriptor(name):
    if not name.startswith('cell (') or not name.endswith(')'):return None
    parts=name[6:-1].split(', ')
    if len(parts) not in {4,5}:raise ValueError('Unrecognized CIFAR worker job name')
    stage,dataset,gamma=parts[:3];gamma=float(gamma)
    if stage=='calibration' and len(parts)==4:
        return (stage,dataset,gamma,float(parts[3])),None
    if stage=='production' and len(parts)==5:
        return (stage,dataset,gamma,int(parts[3])),parts[4]
    raise ValueError('Unrecognized CIFAR stage/job fields')


def resolve_job(name,cohort_cells):
    parsed=job_descriptor(name)
    if parsed is None:return None
    key,rate=parsed
    if key[0]=='calibration':
        return parse_cell(dict(stage=key[0],dataset=key[1],gamma=key[2],multiplier=key[3]))
    if rate.startswith('0x'):
        return parse_cell(dict(stage=key[0],dataset=key[1],gamma=key[2],seed=key[3],lr_hex=rate))
    # Decimal display names are not exact LR evidence: the original matrix
    # transport rounded the gamma-128 rate. Use typed source-status cells only.
    candidates={cell_id(c):c for c in cohort_cells if descriptor(c)==key}
    if len(candidates)!=1:raise ValueError('Decimal job LR has no unique typed source-status identity')
    return next(iter(candidates.values()))


def validate_status(path,source,dataset):
    row=load_json(path.read_bytes());match=STATUS.fullmatch(path.name)
    cell=parse_cell(row.get('cell',{}));identifier=cell_id(cell)
    if (match is None or row.get('schema')!='feature-topology.remote-cifar.v1'
            or row.get('cell_id')!=identifier or match.group(1)!=identifier
            or row.get('host',{}).get('commit')!=source
            or str(row['host'].get('workflow_run'))!=match.group(2)
            or cell['dataset']!=dataset or row.get('config')!=config_for(cell)
            or not re.fullmatch('[0-9a-f]{64}',str(row.get('archive_sha256','')))):
        raise ValueError('CIFAR status source/configuration identity mismatch: '+path.name)
    return row


def api(endpoint):
    return load_json(subprocess.check_output(['gh','api',f'repos/{REPOSITORY}/{endpoint}']))


def reconstruct(repo,dataset,source,tag,output):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    if (not re.fullmatch('[0-9a-f]{40}',source) or not re.fullmatch('[0-9A-Za-z._-]+',tag)
            or dataset not in {'CIFAR10','CIFAR100'} or not output.is_relative_to(repo)
            or (output.exists() and any(output.iterdir()))):
        raise ValueError('Expected safe source identity and a new output directory inside the repository')
    output.mkdir(parents=True,exist_ok=True);raw=output/'api';raw.mkdir();statuses=output/'statuses';statuses.mkdir()
    evidence={}
    def retain(name,value):
        path=raw/name;payload=(json.dumps(value,indent=2)+'\n').encode()
        publish(raw,{name:payload});evidence[path.relative_to(repo).as_posix()]=hashlib.sha256(payload).hexdigest()
    tag_ref=api('git/ref/tags/'+tag);retain('tag_ref.json',tag_ref);target=tag_ref['object']
    for _ in range(4):
        if target['type']=='commit':break
        if target['type']!='tag':raise ValueError('Release ref does not resolve to a source commit')
        annotated=api('git/tags/'+target['sha']);retain('annotated-'+target['sha']+'.json',annotated);target=annotated['object']
    if target.get('type')!='commit' or target['sha']!=source:raise ValueError('CIFAR release tag changed source')
    release=api('releases/tags/'+tag);retain('release.json',release)
    asset_rows={a['name']:a for a in release['assets'] if a['name'].startswith('status-')}
    subprocess.run(['gh','release','download',tag,'--repo',REPOSITORY,'--dir',str(statuses),
        '--pattern','status-*.json'],check=True,capture_output=True,text=True)
    if {p.name for p in statuses.glob('*.json')}!=set(asset_rows):raise ValueError('Release status inventory changed during download')
    typed=[]
    for path in sorted(statuses.glob('*.json')):
        asset=asset_rows[path.name];actual=digest(path)
        if asset.get('digest')!='sha256:'+actual:raise ValueError('Authenticated release status digest differs')
        row=validate_status(path,source,dataset);typed.append(row['cell'])
        evidence[path.relative_to(repo).as_posix()]=actual
    runs=api('actions/workflows/cifar_v3.yml/runs?head_sha='+source+'&per_page=100');retain('runs.json',runs)
    if runs.get('total_count')!=len(runs.get('workflow_runs',[])):raise ValueError('Workflow pagination is required; no truncated provenance accepted')
    if any(r['head_sha']!=source for r in runs['workflow_runs']):raise ValueError('Different source returned in workflow inventory')
    def jobs_for(run):
        value=api(f"actions/runs/{run['id']}/jobs?per_page=100")
        if value.get('total_count')!=len(value.get('jobs',[])):raise ValueError('Job pagination is required')
        return run,value
    with ThreadPoolExecutor(max_workers=4) as pool:observed=list(pool.map(jobs_for,runs['workflow_runs']))
    attempts=[];excluded=[];unresolved=[];active=[]
    for run,jobs in observed:
        retain(str(run['id'])+'-jobs.json',jobs);stages={}
        for job in jobs['jobs']:
            parsed=job_descriptor(job['name'])
            if parsed is None or parsed[0][1]!=dataset:continue
            try:cell=resolve_job(job['name'],typed)
            except ValueError as error:
                unresolved.append(dict(workflow_run=str(run['id']),job_id=job['id'],job_name=job['name'],reason=str(error)))
                continue
            stages.setdefault(cell['stage'],set()).add(cell_id(cell))
        if not stages:
            excluded.append(dict(workflow_run=str(run['id']),reason='No assignable worker cell for this dataset; raw run/job evidence retained'))
        for stage,identifiers in sorted(stages.items()):
            if run['status']!='completed' or not run.get('conclusion'):
                active.append(str(run['id']));continue
            attempts.append(dict(dataset=dataset,stage=stage,workflow_run=str(run['id']),url=run['html_url'],
                cell_ids=sorted(identifiers),started=run['created_at'],finished=run['updated_at'],conclusion=run['conclusion']))
    observation=dict(schema='feature-topology.cifar-provenance-reconstruction.v1',dataset=dataset,source_commit=source,
        result_tag=tag,complete_source_ledger=not unresolved and not active and bool(attempts),
        terminal_attempts=attempts,active_workflows=sorted(set(active)),unresolved_jobs=unresolved,excluded_runs=excluded,
        evidence_sha256=evidence,scope='Authenticated API/status reconstruction only; no original-controller byte identity, tensor verification, training completion or full-study freeze claim')
    publish(output,{'observation.json':(json.dumps(observation,indent=2)+'\n').encode()})
    if observation['complete_source_ledger']:
        reconstruction=dict(method=observation['schema'],result_tag=tag,evidence_sha256=evidence,
            observation_path=(output/'observation.json').relative_to(repo).as_posix(),
            observation_sha256=digest(output/'observation.json'),
            scope=observation['scope'],excluded_runs=excluded)
        ledger=dict(schema='feature-topology.remote-cifar-queue.v1',source_commit=source,attempts=attempts,reconstruction=reconstruction)
        publish(output,{'source_ledger.json':(json.dumps(ledger,indent=2)+'\n').encode()})
    return {k:observation[k] for k in ['dataset','source_commit','complete_source_ledger','active_workflows','unresolved_jobs']}


def validate_reconstruction(repo,state,dataset):
    """Keep every API/status byte used by an explicitly rebuilt source ledger."""
    reconstruction=state.get('reconstruction')
    if reconstruction is None:return []
    if (state.get('schema')!='feature-topology.remote-cifar-queue.v1'
            or reconstruction.get('method')!='feature-topology.cifar-provenance-reconstruction.v1'
            or not reconstruction.get('evidence_sha256')):
        raise ValueError('Invalid reconstructed CIFAR source provenance')
    repo=Path(repo).resolve();paths=[]
    def checked(relative,expected):
        path=repo/relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(repo) or not path.is_file() or digest(path)!=expected):
            raise ValueError('Reconstructed CIFAR evidence missing, changed or escaped')
        paths.append(path);return path
    observation_path=checked(reconstruction['observation_path'],reconstruction['observation_sha256'])
    observation=json.loads(observation_path.read_text())
    if (observation.get('schema')!=reconstruction['method'] or observation.get('dataset')!=dataset
            or observation.get('source_commit')!=state.get('source_commit')
            or observation.get('result_tag')!=reconstruction.get('result_tag')
            or observation.get('complete_source_ledger') is not True
            or observation.get('active_workflows')!=[] or observation.get('unresolved_jobs')!=[]
            or observation.get('terminal_attempts')!=state.get('attempts')
            or observation.get('evidence_sha256')!=reconstruction['evidence_sha256']
            or observation.get('excluded_runs')!=reconstruction.get('excluded_runs')):
        raise ValueError('Reconstructed CIFAR ledger contradicts its retained observation')
    for relative,expected in reconstruction['evidence_sha256'].items():checked(relative,expected)
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--dataset',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(reconstruct(args.repo,args.dataset,args.source_commit,args.tag,args.output)))
