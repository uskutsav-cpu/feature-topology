"""Fail closed until every requested study and production trajectory is complete."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import inventory, sha256
from research_ext.catalog import ABLATION_PRODUCTION, load_catalog, PRODUCTION
from research_ext.image_audit import audit_images
from research_ext.exact import verify_box_certificate,verify_polygon_certificate
from src.training.checkpoints import atomic_json, fingerprint
from scripts.dataset_provenance import verify_provenance


def nonfinite_paths(value,path=""):
    """Locate serialized nonfinite/null metric values without hiding them."""
    if value is None or (isinstance(value,float) and not math.isfinite(value)):
        return [path or "<root>"]
    if isinstance(value,dict):
        return [item for key,child in value.items()
                for item in nonfinite_paths(child,f"{path}.{key}" if path else key)]
    if isinstance(value,list):
        return [item for index,child in enumerate(value)
                for item in nonfinite_paths(child,f"{path}[{index}]")]
    return []


def exact_control_paths(repo):
    """Replay every deterministic analytic control and return frozen inputs."""
    root=Path(repo)/'results/research_ext/exact_controls'
    summary_path=root/'control_summary.json'
    summary=json.loads(summary_path.read_text())
    expected={
        'identity':(1,True),'scaled_rotated':(1,True),
        'projection':(0,False),'constant':(0,False),
        'relu_same_cycle_rank_collision':(1,False),
        'signed_relu_embedding':(1,True),
    }
    if summary.get('status')!='exact_controls_passed' or set(summary.get('controls',{}))!=set(expected)|{'positive_box'}:
        raise ValueError('Exact analytic control summary incomplete')
    paths=[summary_path]
    for name,(beta1,injective) in expected.items():
        path=root/f'{name}.json'
        certificate=json.loads(path.read_text())
        row=summary['controls'][name]
        if (not verify_polygon_certificate(certificate)
                or certificate.get('graph',{}).get('beta1')!=beta1
                or certificate.get('injective_on_polygon_subset') is not injective
                or row.get('beta1')!=beta1
                or row.get('injective_on_polygon_subset') is not injective
                or row.get('python_exact_replay_passed') is not True):
            raise ValueError(f'Exact analytic control replay failed: {name}')
        paths.append(path)
    box_path=root/'positive_box.json'
    box=json.loads(box_path.read_text())
    if (not verify_box_certificate(box)
            or summary['controls']['positive_box'].get('status')!='certified_injective_on_box'
            or summary['controls']['positive_box'].get('python_exact_replay_passed') is not True):
        raise ValueError('Exact positive-box control replay failed')
    paths.append(box_path)
    return paths


def validate_numeric_circle_quotient(report):
    """Validate the finite polygon/LP diagnostic without upgrading it to proof."""
    if (report.get('polygon_points')!=360
            or not isinstance(report.get('tolerance'),(int,float))
            or not math.isfinite(report['tolerance']) or report['tolerance']<=0
            or len(report.get('initial',[]))!=2 or len(report.get('final',[]))!=2):
        raise ValueError('Invalid numerical circle-quotient schema')
    for stage in ('initial','final'):
        for layer in report[stage]:
            required=('beta0','beta1','nodes','edges','affine_segments','pair_intersections','tolerance')
            if (any(key not in layer for key in required)
                    or nonfinite_paths({key:layer[key] for key in required})):
                raise ValueError('Invalid numerical circle-quotient values')
    return report


def production_queue_paths(repo):
    """Require final hosted-queue and collector ledgers for every cohort."""
    repo=Path(repo)
    registry_path=repo/'configs/completion/production_cohorts_v3.json'
    registry=json.loads(registry_path.read_text())
    cohorts={row['name']:row for row in registry['cohorts']}
    state_paths=[repo/'results/completion/production_queue_primary_v3.json',
                 repo/'results/completion/production_queue_ablation_v3.json']
    observed={}
    paths=[registry_path]
    for path in state_paths:
        state=json.loads(path.read_text())
        if state.get('schema')!='feature-topology.production-queue-state.v1':
            raise ValueError(f'Invalid production queue ledger: {path.name}')
        overlap=set(observed)&set(state.get('cohorts',{}))
        if overlap:
            raise ValueError(f'Duplicate queue cohorts: {sorted(overlap)}')
        observed.update(state.get('cohorts',{}))
        for attempt in state.get('attempts',[]):
            if (attempt.get('cohort') not in cohorts or not str(attempt.get('workflow_run','')).isdigit()
                    or not str(attempt.get('url','')).startswith('https://github.com/')
                    or not attempt.get('run_ids') or len(attempt['run_ids'])>24
                    or 'started' not in attempt or 'finished' not in attempt
                    or not attempt.get('conclusion')):
                raise ValueError(f'Incomplete hosted attempt provenance: {path.name}')
        paths.append(path)
    if set(observed)!=set(cohorts):
        raise ValueError(f'Hosted cohort ledger coverage {len(observed)}/{len(cohorts)}')
    for name,definition in cohorts.items():
        row=observed[name]
        expected=definition['expected_unique_profiles']
        if (row.get('expected')!=expected or row.get('complete')!=expected
                or row.get('missing')!=0 or row.get('invalid')!=[]):
            raise ValueError(f'Hosted cohort incomplete: {name}')
        report_path=repo/'results/completion/remote_collections'/f'{name}.json'
        report=json.loads(report_path.read_text())
        if (report.get('expected')!=expected or len(report.get('complete',[]))!=expected
                or report.get('missing')!=[] or report.get('invalid')!=[]):
            raise ValueError(f'Collector ledger incomplete: {name}')
        paths.append(report_path)
    return paths


def validate_ood_evaluation_row(row):
    """Validate finite predictive intervention measurements before freezing."""
    if (row.get('schema')!='feature-topology.nuisance-shift-evaluation.v1'
            or row.get('status')!='evaluated'
            or set(row.get('environments',{}))!={'iid','concentrated','spurious','unseen'}):
        raise ValueError('Invalid nuisance-shift evaluation schema')
    measured={'gamma':row.get('gamma'),'seed':row.get('seed'),
              'selected_step':row.get('selected_step'),
              'actual_training_loss':row.get('actual_training_loss'),
              'environments':row.get('environments')}
    invalid=nonfinite_paths(measured)
    if invalid:
        raise ValueError(f'Nonfinite nuisance-shift values: {invalid}')
    for name,values in row['environments'].items():
        accuracy,loss=values.get('accuracy'),values.get('loss')
        if (not isinstance(values.get('samples'),int) or isinstance(values.get('samples'),bool)
                or values['samples']<=0 or not isinstance(accuracy,(int,float))
                or isinstance(accuracy,bool) or not 0<=accuracy<=1
                or not isinstance(loss,(int,float)) or isinstance(loss,bool) or loss<0):
            raise ValueError(f'Invalid nuisance-shift measurement: {name}')
    return row


def readiness(repo):
    repo=Path(repo).resolve()
    # A freeze must independently replay the strict checkpoint-tensor audit;
    # it must not rely on a separately generated inventory report.
    result=inventory(repo,check_tensors=True)
    problems=list(result['failures'])
    expected_counts=dict(main=130,width=175,depth=140,relevance=210,swapped=35,cylinder=35,
                         small_network=120,ood=140)
    if {k:v['expected'] for k,v in result['studies'].items()}!=expected_counts:
        problems.append(dict(study='design',error='Required sweep designs missing or changed'))
    paths=set()
    requested_ids=set()
    primary_ids=set()
    for name,study in result['studies'].items():
        if study['accounted']!=study['expected']:
            problems.append(dict(study=name,error=f"Training coverage {study['accounted']}/{study['expected']}"))
        requested_ids.update(r['run_id'] for r in study['runs'])
        if name=='main':
            primary_ids.update(r['run_id'] for r in study['runs'])
    selected=[r for r in result['validated_runs'] if r['run_id'] in requested_ids]
    roots=sorted({str(Path(r['path']).parent) for r in selected})
    catalog=load_catalog(roots) if roots else None
    if catalog:
        problems.extend(i for i in catalog.issues if i['severity']=='error')
        info={r['run_id']:r for r in catalog.runs}
    else:
        info={}
    for r in selected:
        run=Path(r['path'])
        options=PRODUCTION if r['run_id'] in primary_ids else ABLATION_PRODUCTION
        options_id=fingerprint(options)
        paths.update(run/p for p in ['summary.json','config.json'])
        if r['status']=='diverged':
            continue
        paths.add(run/'final.pt')
        all_checkpoints=list(run.glob('step_*.pt'))+[run/'final.pt']
        checkpoints=(all_checkpoints if options['all_checkpoints']
                     else [run/'step_0000000.pt',run/'final.pt'])
        if not (run/'step_0000000.pt').exists():
            problems.append(dict(run=r['run_id'],error='Missing initial checkpoint'))
        paths.update(all_checkpoints)
        profiles=[p for p in info.get(r['run_id'],{}).get('profiles',[])
                  if p['directory_name']==options_id]
        if not profiles or profiles[0]['missing_checkpoints']:
            problems.append(dict(run=r['run_id'],error='Full production trajectory metrics missing'))
            continue
        directory=run/'metrics'/options_id
        execution=directory/'provenance/execution.json'
        if not execution.exists():
            problems.append(dict(run=r['run_id'],error='Metric execution provenance missing'))
            continue
        paths.update([directory/'options.json',execution])
        paths.update((directory/'provenance').glob('*.json'))
        for checkpoint in checkpoints:
            metric=directory/(checkpoint.stem+'.json')
            row=json.loads(metric.read_text())
            invalid_numbers=nonfinite_paths(row)
            if invalid_numbers:
                problems.append(dict(run=r['run_id'],error=f"Nonfinite metric values: {metric.name}",
                                     paths=invalid_numbers))
            if row.get('checkpoint_sha256')!=sha256(checkpoint):
                problems.append(dict(run=r['run_id'],error=f'Checkpoint hash mismatch: {metric.name}'))
            if len(row.get('layers',[]))!=r['config'].get('depth',4):
                problems.append(dict(run=r['run_id'],error=f'Layer coverage mismatch: {metric.name}'))
            paths.add(metric)
            for layer in row.get('layers',[]):
                ph_required=(options.get('ph_schedule','all')=='all'
                             or checkpoint.stem in {'step_0000000','final'})
                if ph_required and (len(layer.get('persistence',[]))!=options['ph_repeats']
                                    or any('H2' not in v for v in layer['persistence'])):
                    problems.append(dict(run=r['run_id'],error='Required H2 replicates missing'))
                suffixes=['tangents','ph'] if ph_required else ['tangents']
                for suffix in suffixes:
                    artifact=directory/f"{checkpoint.stem}_layer{layer['layer']}_{suffix}.npz"
                    if not artifact.exists():
                        problems.append(dict(run=r['run_id'],error=f'Missing {artifact.name}'))
                    else:
                        paths.add(artifact)
                        if suffix=='tangents':
                            try:
                                with np.load(artifact,allow_pickle=False) as arrays:
                                    if (set(arrays.files)!={'indices','sigma_min','factor_norms'}
                                            or any(not np.isfinite(arrays[name]).all()
                                                   for name in ['sigma_min','factor_norms'])):
                                        problems.append(dict(run=r['run_id'],
                                                             error=f'Invalid tangent archive: {artifact.name}'))
                            except (ValueError,OSError) as exc:
                                problems.append(dict(run=r['run_id'],
                                                     error=f'Unreadable tangent archive: {artifact.name}: {exc}'))
    for name in ['rotated_digits','dsprites','cifar10','cifar100']:
        audit=audit_images(repo/'results'/name)
        if not audit['artifact_completion']:
            problems.append(dict(study=name,error='Image coverage incomplete',audit=audit))
        for path in (repo/'results'/name).glob('runs/*/*'):
            if path.is_file() and path.name!='resume.pt':
                paths.add(path)
    dataset_provenance=repo/'results/completion/dataset_provenance.json'
    if not dataset_provenance.exists():
        problems.append(dict(study='datasets',error='Official dataset provenance missing'))
    else:
        try:
            paths.update(verify_provenance(repo,dataset_provenance))
        except (ValueError,KeyError,OSError) as exc:
            problems.append(dict(study='datasets',error=str(exc)))
    try:
        paths.update(image_replay_paths(repo))
    except (ValueError,KeyError,OSError) as exc:
        problems.append(dict(study='image_replay',error=str(exc)))
    try:
        paths.update(production_queue_paths(repo))
    except (ValueError,KeyError,OSError) as exc:
        problems.append(dict(study='hosted_metrics',error=str(exc)))
    ood_manifest=repo/'results/ood_evaluation/manifest.json'
    if not ood_manifest.exists():
        problems.append(dict(study='ood_evaluation',error='Nuisance-shift evaluation missing'))
    else:
        ood=json.loads(ood_manifest.read_text())
        expected_ood={r['run_id'] for r in result['studies'].get('ood',{}).get('runs',[])}
        observed={r.get('run_id') for r in ood.get('runs',[]) if r.get('status')=='evaluated'}
        if observed!=expected_ood:
            problems.append(dict(study='ood_evaluation',error=f'Evaluation coverage {len(observed)}/{len(expected_ood)}'))
        for row in ood.get('runs',[]):
            if row.get('status')!='evaluated':
                continue
            try:
                validate_ood_evaluation_row(row)
            except (ValueError,KeyError) as exc:
                problems.append(dict(study='ood_evaluation',error=f'{row.get("run_id")}: {exc}'))
            record=next((r for r in result['validated_runs'] if r['run_id']==row.get('run_id')),None)
            checkpoint=Path(record['path'])/row['checkpoint'] if record else None
            if checkpoint is None or not checkpoint.exists() or sha256(checkpoint)!=row.get('checkpoint_sha256'):
                problems.append(dict(study='ood_evaluation',error=f'Checkpoint mismatch: {row.get("run_id")}'))
            else:
                paths.add(checkpoint)
        paths.update(p for p in ood_manifest.parent.glob('*.json'))
    width_spec=repo/'configs/width_scaling_gate_v1.json'
    width_result=repo/'results/analysis/width_scaling_gate.json'
    if not width_result.exists():
        problems.append(dict(study='width_scaling',error='Finite-width terminology gate missing'))
    else:
        gate=json.loads(width_result.read_text())
        if (gate.get('schema')!='feature-topology.width-scaling-result.v1'
                or gate.get('status')!='complete'):
            problems.append(dict(study='width_scaling',error='Finite-width terminology gate incomplete'))
        if gate.get('specification_sha256')!=sha256(width_spec):
            problems.append(dict(study='width_scaling',error='Finite-width gate specification mismatch'))
        allowed=gate.get('phase_transition_language_allowed') is True
        expected_term='phase transition' if allowed else 'crossover'
        if gate.get('terminology')!=expected_term:
            problems.append(dict(study='width_scaling',error='Finite-width terminology decision inconsistent'))
        for relative,expected in gate.get('input_hashes',{}).items():
            source=repo/relative
            if (not source.resolve().is_relative_to(repo) or not source.exists()
                    or sha256(source)!=expected):
                problems.append(dict(study='width_scaling',error=f'Finite-width input changed: {relative}'))
            else:
                paths.add(source)
        paths.update([width_spec,width_result])
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
            quotient=checkpoint.parent/'quotient.json'
            if (sha256(certificate)!=record['certificate_sha256'] or not verify_polygon_certificate(cert)
                or sha256(checkpoint)!=record['checkpoint_sha256']
                or cert.get('provenance',{}).get('checkpoint_sha256')!=record['checkpoint_sha256']):
                problems.append(dict(study='exact',error=f'Certificate replay failed: {certificate}'))
            try:
                validate_numeric_circle_quotient(json.loads(quotient.read_text()))
            except (ValueError,KeyError,OSError) as exc:
                problems.append(dict(study='exact',error=f'Numerical quotient invalid: {record["run_id"]}: {exc}'))
            paths.update([certificate,checkpoint,quotient,checkpoint.parent/'summary.json',checkpoint.parent/'config.json'])
        paths.add(exact)
    try:
        paths.update(exact_control_paths(repo))
    except (ValueError,KeyError,OSError) as exc:
        problems.append(dict(study='exact_controls',error=str(exc)))
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
    for name in ['main','width','depth','relevance','swapped','cylinder','small_network','ood',
                 'rotated_digits','dsprites','cifar10','cifar100','circle_quotient']:
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


def image_replay_paths(repo):
    """Validate held-out replay reports and their checkpoint/metric bindings."""
    repo=Path(repo)
    paths=[]
    rotated_path=repo/'results/completion/rotated_digits_replay.json'
    rotated=json.loads(rotated_path.read_text())
    if (not rotated.get('complete') or len(rotated.get('records',[]))!=35
            or not all(row.get('passed') for row in rotated.get('records',[]))):
        raise ValueError('Rotated-digits held-out replay incomplete')
    for row in rotated['records']:
        run=repo/'results/rotated_digits/runs'/row['run_id']
        if (sha256(run/'final.pt')!=row.get('checkpoint_sha256')
                or sha256(run/'metrics.json')!=row.get('metrics_sha256')):
            raise ValueError(f"Rotated-digits replay binding changed: {row['run_id']}")
    paths.append(rotated_path)
    dsprites_path=repo/'results/completion/dsprites_replay.json'
    dsprites=json.loads(dsprites_path.read_text())
    if (dsprites.get('schema')!='feature-topology.dsprites-replay.v1'
            or not dsprites.get('complete') or dsprites.get('replayed_runs')!=35
            or len(dsprites.get('records',[]))!=35):
        raise ValueError('dSprites held-out replay incomplete')
    for row in dsprites['records']:
        checkpoint=repo/'results/dsprites/runs'/row['run_id']/'final.pt'
        if sha256(checkpoint)!=row.get('checkpoint_sha256'):
            raise ValueError(f"dSprites replay binding changed: {row['run_id']}")
    paths.append(dsprites_path)
    cifar_path=repo/'results/completion/cifar_validation.json'
    cifar=json.loads(cifar_path.read_text())
    datasets={row.get('dataset'):row for row in cifar.get('datasets',[])}
    if (cifar.get('schema')!='feature-topology.cifar-validation.v1' or not cifar.get('complete')
            or set(datasets)!={'CIFAR10','CIFAR100'}
            or any(row.get('validated_runs')!=35 or len(row.get('records',[]))!=35
                   for row in datasets.values())):
        raise ValueError('CIFAR held-out replay incomplete')
    for name,dataset in datasets.items():
        for row in dataset['records']:
            checkpoint=repo/'results'/name.lower()/'runs'/row['run_id']/'final.pt'
            if sha256(checkpoint)!=row.get('checkpoint_sha256'):
                raise ValueError(f"CIFAR replay binding changed: {row['run_id']}")
    paths.append(cifar_path)
    return paths


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
