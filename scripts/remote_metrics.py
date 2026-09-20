"""Checksum-verified CI metric workers with resumable result archives."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def selected(index,run_id):
    if not re.fullmatch('[0-9a-f]{16}',run_id):raise ValueError('Invalid run ID')
    return next(r for r in index['runs'] if r['run_id']==run_id)


def unpack_input(index,run_id,incoming,work):
    record=selected(index,run_id)
    incoming,work=Path(incoming),Path(work)
    archive=incoming/record['asset']
    arrays=incoming/index['analysis_data']['asset']
    if digest(archive)!=record['archive_sha256'] or digest(arrays)!=index['analysis_data']['sha256']:
        raise ValueError('Input archive/array checksum mismatch')
    destination=work/'main/runs'/run_id
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        expected={run_id+'/'+name for name in record['files']}
        if {m.name for m in tar.getmembers()}!=expected or len(tar.getmembers())!=len(expected):
            raise ValueError('Input archive membership mismatch')
        for member in tar.getmembers():
            if not member.isfile():raise ValueError('Only regular checkpoint files are allowed')
            name=member.name.split('/')[1]
            payload=tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest()!=record['files'][name]:
                raise ValueError('Checkpoint member checksum mismatch')
            target=destination/name
            if target.exists() and digest(target)!=record['files'][name]:
                raise ValueError('Local checkpoint changed')
            if not target.exists():target.write_bytes(payload)
    return destination,arrays


def restore(archive,index,run_id,work,provenance):
    with tarfile.open(archive,'r:gz') as tar:
        report=json.load(tar.extractfile('job_manifest.json'))
        record=selected(index,run_id)
        if (report['run_id']!=run_id or report['input_archive_sha256']!=record['archive_sha256']
            or report['analysis_data_sha256']!=index['analysis_data']['sha256']
            or report.get('metric_provenance')!=provenance):
            return False
        if {m.name for m in tar.getmembers()}!={*report['files'],'job_manifest.json'}:
            raise ValueError('Resume archive membership mismatch')
        for name,expected in report['files'].items():
            allowed=(name.startswith(f'main/runs/{run_id}/metrics/') or name.startswith('ph_cache/'))
            relative=Path(name)
            if not allowed or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe resume archive path')
            member=tar.getmember(name)
            if not member.isfile():raise ValueError('Only regular resume files are allowed')
            payload=tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Resume member checksum mismatch')
            target=Path(work)/name
            if target.exists():
                # Completed rows/cache entries from an earlier compatible attempt remain immutable.
                if digest(target)!=expected and not name.endswith('/host.json'):
                    raise ValueError(f'Conflicting compatible resume artifacts: {name}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
    return True


def pack(index,run_id,work,output,job_id):
    record=selected(index,run_id)
    work,output=Path(work),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    run=work/'main/runs'/run_id
    profile=index['metric_options']
    # Same canonical serialization as src.training.checkpoints.fingerprint.
    from src.training.checkpoints import fingerprint
    directory=run/'metrics'/fingerprint(profile)
    provenance_path=directory/'provenance/execution.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    host=dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
              commit=os.environ.get('GITHUB_SHA'),workflow_run=os.environ.get('GITHUB_RUN_ID'))
    (directory/'provenance').mkdir(parents=True,exist_ok=True)
    (directory/'provenance/host.json').write_text(json.dumps(host,sort_keys=True,indent=2)+'\n')
    expected=[name[:-3] for name in record['files'] if name.endswith('.pt')]
    completed=[]
    for checkpoint in expected:
        path=directory/(checkpoint+'.json')
        if not path.exists():continue
        row=json.loads(path.read_text())
        if row.get('checkpoint_sha256')!=record['files'][checkpoint+'.pt']:
            raise ValueError('Output row belongs to different checkpoint bytes')
        if len(row.get('layers',[]))!=record['config'].get('depth',4):continue
        valid=True
        ph_required=(profile.get('ph_schedule','all')=='all'
                     or checkpoint in {'step_0000000','final'})
        for layer in row['layers']:
            valid &= (not ph_required or (len(layer.get('persistence',[]))==profile['ph_repeats']
                      and all('H2' in p and p['size']==profile['ph_size'] for p in layer['persistence'])))
            suffixes=['tangents','ph'] if ph_required else ['tangents']
            valid &= all((directory/f"{checkpoint}_layer{layer['layer']}_{suffix}.npz").exists() for suffix in suffixes)
        if valid:completed.append(checkpoint)
    files=sorted([p for p in directory.rglob('*') if p.is_file()]+list((work/'ph_cache').glob('*/repeat_*.npz')))
    report=dict(schema='feature-topology.remote-metrics.v1',run_id=run_id,job_id=job_id,
                complete=set(completed)==set(expected),completed=completed,expected=expected,
                input_archive_sha256=record['archive_sha256'],analysis_data_sha256=index['analysis_data']['sha256'],
                metric_provenance=provenance,host=host,
                files={str(p.relative_to(work)):digest(p) for p in files})
    if not re.fullmatch('[0-9A-Za-z_-]+',job_id):raise ValueError('Invalid job ID')
    archive=output/f'metrics-{run_id}-{job_id}.tar.gz'
    with archive.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|') as tar:
            for path in files:
                info=tarfile.TarInfo(str(path.relative_to(work)));info.size=path.stat().st_size;info.mode=0o644
                with path.open('rb') as source:tar.addfile(info,source)
            payload=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
            info=tarfile.TarInfo('job_manifest.json');info.size=len(payload);info.mode=0o644
            tar.addfile(info,io.BytesIO(payload))
    report['archive_sha256']=digest(archive)
    (output/f'status-{run_id}-{job_id}.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(run_id=run_id,complete=report['complete'],completed=len(completed),expected=len(expected))),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['plan','compute','pack'])
    parser.add_argument('--index',default='configs/completion/primary_inputs.json')
    parser.add_argument('--run-id')
    parser.add_argument('--run-ids',default='[]')
    parser.add_argument('--incoming',default='incoming')
    parser.add_argument('--work',default='metric-work')
    parser.add_argument('--resume',default='resume')
    parser.add_argument('--output',default='outgoing')
    parser.add_argument('--job-id',default='local')
    args=parser.parse_args()
    index=json.loads(Path(args.index).read_text())
    if args.command=='plan':
        ids=json.loads(args.run_ids) or [r['run_id'] for r in index['runs'][:24]]
        if not isinstance(ids,list) or len(ids)!=len(set(ids)):raise ValueError('Run IDs must be a unique JSON list')
        if len(ids)>24:raise ValueError('Use batches of at most 24 runs to remain within hosted-runner queue limits')
        for rid in ids:selected(index,rid)
        line='run_ids='+json.dumps(ids,separators=(',',':'))+'\n'
        if os.environ.get('GITHUB_OUTPUT'):
            with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:output.write(line)
        print(line,end='')
    elif args.command=='compute':
        from scripts.compute_metrics import compute,metric_provenance
        run,arrays=unpack_input(index,args.run_id,args.incoming,args.work)
        provenance=metric_provenance(arrays)
        for archive in sorted(Path(args.resume).glob(f'metrics-{args.run_id}-*.tar.gz')):
            print(json.dumps(dict(resume=archive.name,compatible=restore(archive,index,args.run_id,args.work,provenance))),flush=True)
        compute(run,index['metric_options'],analysis_data=arrays)
    else:
        pack(index,args.run_id,args.work,args.output,args.job_id)
