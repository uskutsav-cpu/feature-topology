"""Fail closed until every requested study and production trajectory is complete."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import inventory, sha256
from research_ext.catalog import load_catalog, PRODUCTION
from research_ext.image_audit import audit_images
from research_ext.exact import verify_polygon_certificate
from src.training.checkpoints import atomic_json, fingerprint


def readiness(repo):
    repo=Path(repo).resolve()
    result=inventory(repo)
    problems=list(result['failures'])
    expected_counts=dict(main=130,width=175,depth=140,relevance=210,swapped=35,cylinder=35,small_network=120)
    if {k:v['expected'] for k,v in result['studies'].items()}!=expected_counts:
        problems.append(dict(study='design',error='Required sweep designs missing or changed'))
    paths=set()
    requested_ids=set()
    for name,study in result['studies'].items():
        if study['accounted']!=study['expected']:
            problems.append(dict(study=name,error=f"Training coverage {study['accounted']}/{study['expected']}"))
        requested_ids.update(r['run_id'] for r in study['runs'])
    selected=[r for r in result['validated_runs'] if r['run_id'] in requested_ids]
    roots=sorted({str(Path(r['path']).parent) for r in selected})
    catalog=load_catalog(roots,profile=fingerprint(PRODUCTION)) if roots else None
    if catalog:
        problems.extend(i for i in catalog.issues if i['severity']=='error')
        info={r['run_id']:r for r in catalog.runs}
    else:
        info={}
    for r in selected:
        run=Path(r['path'])
        paths.update(run/p for p in ['summary.json','config.json'])
        if r['status']=='diverged':
            continue
        paths.add(run/'final.pt')
        checkpoints=list(run.glob('step_*.pt'))+[run/'final.pt']
        if not (run/'step_0000000.pt').exists():
            problems.append(dict(run=r['run_id'],error='Missing initial checkpoint'))
        paths.update(checkpoints)
        profiles=[p for p in info.get(r['run_id'],{}).get('profiles',[])
                  if p['directory_name']==fingerprint(PRODUCTION)]
        if not profiles or profiles[0]['missing_checkpoints'] or not profiles[0]['production_resolution']:
            problems.append(dict(run=r['run_id'],error='Full production trajectory metrics missing'))
            continue
        directory=run/'metrics'/fingerprint(PRODUCTION)
        execution=directory/'provenance/execution.json'
        if not execution.exists():
            problems.append(dict(run=r['run_id'],error='Metric execution provenance missing'))
            continue
        paths.update([directory/'options.json',execution])
        paths.update((directory/'provenance').glob('*.json'))
        for checkpoint in checkpoints:
            metric=directory/(checkpoint.stem+'.json')
            row=json.loads(metric.read_text())
            if row.get('checkpoint_sha256')!=sha256(checkpoint):
                problems.append(dict(run=r['run_id'],error=f'Checkpoint hash mismatch: {metric.name}'))
            if len(row.get('layers',[]))!=r['config'].get('depth',4):
                problems.append(dict(run=r['run_id'],error=f'Layer coverage mismatch: {metric.name}'))
            paths.add(metric)
            for layer in row.get('layers',[]):
                if len(layer.get('persistence',[]))!=20 or any('H2' not in v for v in layer['persistence']):
                    problems.append(dict(run=r['run_id'],error='Required H2 replicates missing'))
                for suffix in ['tangents','ph']:
                    artifact=directory/f"{checkpoint.stem}_layer{layer['layer']}_{suffix}.npz"
                    if not artifact.exists():
                        problems.append(dict(run=r['run_id'],error=f'Missing {artifact.name}'))
                    else:
                        paths.add(artifact)
    for name in ['rotated_digits','dsprites','cifar10','cifar100']:
        audit=audit_images(repo/'results'/name)
        if not audit['artifact_completion']:
            problems.append(dict(study=name,error='Image coverage incomplete',audit=audit))
        for path in (repo/'results'/name).glob('runs/*/*'):
            if path.is_file() and path.name!='resume.pt':
                paths.add(path)
    exact=repo/'results/exact_trained_circles/manifest.json'
    if not exact.exists():
        problems.append(dict(study='exact',error='Trained-network certificate manifest missing'))
    else:
        manifest=json.loads(exact.read_text())
        if not manifest.get('complete') or manifest.get('checkpoint_count')!=20 or len(manifest.get('records',[]))!=40:
            problems.append(dict(study='exact',error='Trained circle certificate coverage incomplete'))
        for record in manifest.get('records',[]):
            certificate=exact.parent/record['certificate_file']
            cert=json.loads(certificate.read_text())
            checkpoint=repo/'results/circle_quotient/runs'/record['run_id']/'final.pt'
            if (sha256(certificate)!=record['certificate_sha256'] or not verify_polygon_certificate(cert)
                or sha256(checkpoint)!=record['checkpoint_sha256']
                or cert.get('provenance',{}).get('checkpoint_sha256')!=record['checkpoint_sha256']):
                problems.append(dict(study='exact',error=f'Certificate replay failed: {certificate}'))
            paths.update([certificate,checkpoint,checkpoint.parent/'summary.json',checkpoint.parent/'config.json'])
        paths.add(exact)
    formal=repo/'results/completion/formal/formal_status.json'
    if not formal.exists() or not json.loads(formal.read_text()).get('lean_verified'):
        problems.append(dict(study='formal',error='Compiled theorem audit missing'))
    else:
        report=json.loads(formal.read_text())
        for name,expected in report['files'].items():
            if sha256(repo/'formal'/name)!=expected:
                problems.append(dict(study='formal',error=f'Audited source changed: {name}'))
        paths.update([formal,formal.parent/'lean_build.log'])
    for name in ['src','scripts','research_ext','formal','configs','docs','.github']:
        paths.update(p for p in (repo/name).rglob('*') if p.is_file()
                     and not any(part in {'.lake','__pycache__'} for part in p.parts))
    paths.update(p for p in repo.glob('*') if p.is_file() and p.suffix in {'.txt','.toml','.md'})
    paths.update((repo/'results').rglob('gamma_to_lr.json'))
    for name in ['main','width','depth','relevance','swapped','cylinder','small_network','rotated_digits','dsprites','cifar10','cifar100','circle_quotient']:
        path=repo/'results'/name/'manifest.json'
        if path.exists(): paths.add(path)
    design=repo/'results/dsprites/dataset_design.json'
    if design.exists(): paths.add(design)
    return dict(ready=not problems,problems=problems,
                studies={name:dict(expected=v['expected'],accounted=v['accounted']) for name,v in result['studies'].items()}),paths


def verify_manifest(repo,path):
    repo=Path(repo).resolve()
    manifest=json.loads(Path(path).read_text())
    if manifest.get('schema')!='feature-topology.frozen-results.v1' or not manifest.get('ready'):
        raise ValueError('Not a completed frozen-result manifest')
    for relative,expected in manifest['files'].items():
        source=repo/relative
        if not source.resolve().is_relative_to(repo) or sha256(source)!=expected:
            raise ValueError(f'Frozen file missing or changed: {relative}')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    repo=Path(args.repo).resolve()
    report,paths=readiness(repo)
    if report['ready'] and not args.check_only:
        report.update(schema='feature-topology.frozen-results.v1',
                      files={str(p.relative_to(repo)):sha256(p) for p in sorted(paths)})
    atomic_json(Path(args.output),report)
    print(json.dumps(dict(ready=report['ready'],problem_count=len(report['problems']),studies=report['studies']),indent=2))
    sys.exit(0 if report['ready'] else 2)
