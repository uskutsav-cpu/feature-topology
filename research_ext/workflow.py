"""Sequential, resumable orchestration around the existing research scripts.

No detached workers, shell expansion, destructive cleanup or automatic gate
bypasses. Completion markers are validated against input/output content hashes.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import glob
import os
import re
import shutil
import subprocess
import sys
from .io import read_json, atomic_json, digest, file_digest, exclusive_lock, utc_now, original_fingerprint, environment
from .catalog import PRODUCTION, load_catalog

GAMMAS = [2.**i for i in range(-5,8)]


@dataclass
class Task:
    name: str
    command: list[str]
    dependencies: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    guard: dict = field(default_factory=dict)


def make_plan(repo: str | Path, *, suite: str = 'primary', image_device: str = 'cpu') -> dict:
    if suite not in {'primary','ablations','formal','images','cifar'}:
        raise ValueError('Unknown suite')
    if image_device not in {'cpu','mps'}:
        raise ValueError('The upstream image runner supports cpu or mps only')
    repo = Path(repo).resolve()
    py = sys.executable
    prefix = [py, 'scripts/research.py']
    options_id = original_fingerprint(PRODUCTION)
    if suite in {'images','cifar'}:
        names = ['rotated_digits'] if suite == 'images' else ['CIFAR10','CIFAR100']
        tasks = []
        for name in names:
            output = 'results/'+name.lower()
            command = ([py,'scripts/run_image_validation.py','--output',output,'--device',image_device]
                       if name == 'rotated_digits' else
                       [py,'scripts/run_cifar.py','--dataset',name,'--output',output,'--device',image_device])
            tasks.append(Task(name.lower()+'-study', command,
                outputs=[output+'/gamma_to_lr.json',output+'/runs/*/summary.json',output+'/runs/*/metrics.json'],
                guard={'kind':'image_metrics','study':output}))
            tasks.append(Task(name.lower()+'-audit',prefix+['audit-images','--root',output,
                 '--output','results/research_ext/'+name.lower()+'_audit.json'],
                 dependencies=[name.lower()+'-study'],
                 inputs=[output+'/runs/*/summary.json',output+'/runs/*/metrics.json'],
                 outputs=['results/research_ext/'+name.lower()+'_audit.json']))
    elif suite == 'formal':
        tasks = [Task('exact-controls', prefix+['controls','--output','results/research_ext/exact_controls'],
                      outputs=['results/research_ext/exact_controls/*.json']),
                 Task('lean-check', prefix+['check-formal','--output','results/research_ext/formal'],
                      ['exact-controls'], outputs=['results/research_ext/formal/formal_status.json'],
                      guard={'kind':'json_true','path':'results/research_ext/formal/formal_status.json','key':'lean_verified'})]
    else:
        tasks = [
            Task('calibration', [py,'scripts/calibrate.py','--config','configs/synthetic/calibration_main.json',
                  '--output','results/calibration_main','--trials','results/calibration_pilot/trials'],
                 inputs=['configs/synthetic/calibration_main.json','results/pilot/gate.json'],
                 outputs=['results/calibration_main/gamma_to_lr.json'],
                 guard={'kind':'calibration','path':'results/calibration_main/gamma_to_lr.json','gammas':GAMMAS}),
            Task('main-training', [py,'scripts/run_main_sweep.py','--calibration','results/calibration_main/gamma_to_lr.json',
                  '--output','results/main','--pilot-gate','results/pilot/gate.json','--reuse-runs','results/pilot/runs'],
                 ['calibration'], ['results/calibration_main/gamma_to_lr.json','results/pilot/gate.json'],
                 ['results/main/manifest.json','results/main/runs/*/config.json','results/main/runs/*/summary.json',
                  'results/main/runs/*/final.pt','results/main/runs/*/step_*.pt'],
                 {'kind':'manifest','path':'results/main/manifest.json','gammas':GAMMAS,'seeds':list(range(10))}),
            Task('main-metrics', [py,'scripts/compute_metrics.py','--runs','results/main/runs','--all-checkpoints'],
                 ['main-training'], ['results/main/manifest.json','results/main/runs/*/summary.json',
                                     'results/main/runs/*/final.pt','results/main/runs/*/step_*.pt'],
                 [f'results/main/runs/*/metrics/{options_id}/*.json'],
                 {'kind':'metrics','study':'results/main','profile':options_id}),
            Task('primary-report', prefix+['analyze','--roots','results/main','--profile',options_id,
                  '--output','results/research_ext/primary_report','--plots'], ['main-metrics'],
                 [f'results/main/runs/*/metrics/{options_id}/*.json','results/main/runs/*/summary.json'],
                 ['results/research_ext/primary_report/analysis_result.json',
                  'results/research_ext/primary_report/metrics_long.csv',
                  'results/research_ext/primary_report/model_comparisons.json']),
            Task('numerical-controls', [py,'scripts/run_controls.py'], outputs=['results/controls/controls.json']),
            Task('exact-controls', prefix+['controls','--output','results/research_ext/exact_controls'],
                 outputs=['results/research_ext/exact_controls/*.json'])]
        if suite == 'ablations':
            for name in ['width','depth','relevance','swapped','cylinder','small_network']:
                output = f'results/extended/{name}'
                tasks.append(Task(f'{name}-training', [py,'scripts/run_ablation.py','--plan',f'configs/sweeps/{name}.json',
                    '--output',output], ['main-training'],
                    [f'configs/sweeps/{name}.json','results/calibration_main/gamma_to_lr.json','results/pilot/gate.json'],
                    [output+'/manifest.json',output+'/*/runs/*/summary.json']))
                tasks.append(Task(f'{name}-metrics', prefix+['nested-metrics','--root',output], [f'{name}-training'],
                    [output+'/manifest.json',output+'/*/runs/*/summary.json'], [output+f'/*/runs/*/metrics/{options_id}/*.json']))
                tasks.append(Task(f'{name}-report', prefix+['nested-analyze','--root',output,'--profile',options_id,
                    '--output',f'results/research_ext/{name}_report'], [f'{name}-metrics'],
                    [output+f'/*/runs/*/metrics/{options_id}/*.json',output+'/*/runs/*/summary.json'],
                    [f'results/research_ext/{name}_report/*/analysis_result.json']))
    return {'schema':'feature-topology.workflow.v1','repo':str(repo),'suite':suite,
            'tasks':[asdict(task) for task in tasks], 'min_free_bytes':2*1024**3,
            'note':'A plan is not completed computation. This runs sequentially and stops on failures.'}


def _files(repo: Path, patterns: list[str]) -> dict:
    result = {}
    for pattern in patterns:
        paths = sorted(p for p in repo.glob(pattern) if p.is_file())
        if not paths:
            raise FileNotFoundError(f'No artifact matches {pattern}')
        for path in paths:
            result[str(path.relative_to(repo))] = file_digest(path)
    return result


def _sources(repo: Path) -> str:
    hashes = {}
    for directory in ['src','scripts','research_ext','formal']:
        for pattern in ['**/*.py','**/*.lean','**/*.toml','lean-toolchain']:
            for path in (repo/directory).glob(pattern):
                if path.is_file() and '.lake' not in path.parts:
                    hashes[str(path.relative_to(repo))] = file_digest(path)
    return digest(hashes)


def _guard(repo: Path, guard: dict) -> dict:
    kind = guard.get('kind')
    if kind is None:
        return {'status':'outputs_exist'}
    if kind == 'json_true':
        if read_json(repo/guard['path']).get(guard['key']) is not True:
            raise ValueError(f"Required gate {guard['key']} is not true")
    elif kind == 'calibration':
        import math
        value=read_json(repo/guard['path'])
        if {float(k) for k in value['selection']} != set(guard['gammas']):
            raise ValueError('Calibration gamma coverage mismatch')
        if original_fingerprint(value['config']) != value['calibration_id']:
            raise ValueError('Calibration fingerprint mismatch')
        for row in value['selection'].values():
            if not math.isfinite(row['lr']) or row['lr'] <= 0 or not row['stable']:
                raise ValueError('Invalid frozen learning-rate selection')
    elif kind == 'manifest':
        rows=read_json(repo/guard['path'])
        observed=[(r['gamma'],r['seed']) for r in rows]
        expected={(g,s) for g in guard['gammas'] for s in guard['seeds']}
        if len(observed)!=len(set(observed)) or set(observed)!=expected:
            raise ValueError('Training manifest is incomplete or contains duplicate gamma/seed pairs')
        if any(r['status'] not in {'converged','budget_exhausted','diverged'} for r in rows):
            raise ValueError('A training run is still incomplete')
        # This is execution completion; nonconvergence remains a reported outcome.
        return {'status':'all_requested_runs_recorded','converged':sum(r['status']=='converged' for r in rows),
                'nonconverged':sum(r['status']!='converged' for r in rows)}
    elif kind == 'image_metrics':
        from .image_audit import audit_images
        result = audit_images(repo/guard['study'])
        if not result['artifact_completion']:
            raise ValueError('Image artifacts are incomplete or invalid; run audit-images for details')
        return {'status':'image_artifacts_complete','measured_runs':result['measured_runs'],
                'diverged_runs':result['diverged_runs']}
    elif kind == 'metrics':
        catalog=load_catalog([repo/guard['study']],profile=guard['profile'])
        catalog.require_clean()
        if not catalog.runs:
            raise ValueError('No training runs to validate')
        for run in catalog.runs:
            if run['status']=='diverged':
                continue
            profiles=[p for p in run['profiles'] if p['directory_name']==guard['profile']]
            if not profiles or any(p['missing_checkpoints'] or not p['metric_files'] for p in profiles):
                raise ValueError(f"Missing trajectory metrics for {run['run_id']}")
    else:
        raise ValueError(f'Unknown completion guard {kind}')
    return {'status':'guard_passed'}


def execute(plan: dict, *, state_root: str | Path | None = None,
            only: list[str] | None = None, timeout: float | None = None) -> dict:
    if plan.get('schema')!='feature-topology.workflow.v1':
        raise ValueError('Unsupported workflow schema')
    repo=Path(plan['repo']).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    tasks=[Task(**value) for value in plan['tasks']]
    lookup={task.name:task for task in tasks}
    if len(lookup)!=len(tasks) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+',t.name) for t in tasks):
        raise ValueError('Task names must be unique safe identifiers')
    requested=set(only if only is not None else lookup)
    if requested-lookup.keys():
        raise ValueError('Unknown requested task')
    visiting, ordered, visited=set(),[],set()
    def visit(name):
        if name in visiting: raise ValueError('Dependency cycle')
        if name in visited: return
        if name not in lookup: raise ValueError(f'Unknown dependency {name}')
        visiting.add(name)
        for dep in lookup[name].dependencies: visit(dep)
        visiting.remove(name);visited.add(name);ordered.append(lookup[name])
    for task in tasks:
        if task.name in requested: visit(task.name)
    if plan.get('suite') in {'primary','ablations'}:
        _guard(repo,{'kind':'json_true','path':'results/pilot/gate.json','key':'proceed_to_main'})
    root=Path(state_root or repo/'results/research_ext/workflow_state')
    root.mkdir(parents=True,exist_ok=True)
    events=[]
    with exclusive_lock(root/'workflow.lock'):
        source_id=_sources(repo)
        execution_environment=environment(repo)
        environment_id=digest({k:execution_environment.get(k) for k in ['python','platform','packages','torch']})
        for task in ordered:
            marker=root/f'{task.name}.json'
            record={'task':task.name,'command':task.command,'source_id':source_id,'environment':execution_environment,'started':utc_now()}
            try:
                inputs=_files(repo,task.inputs)
                key=digest({'task':asdict(task),'inputs':inputs,'source_id':source_id,'environment_id':environment_id})
                if marker.exists():
                    prior=read_json(marker)
                    try:
                        outputs=_files(repo,task.outputs)
                        guard=_guard(repo,task.guard)
                    except (OSError,ValueError,KeyError):
                        outputs=None
                    if (prior.get('status')=='completed' and prior.get('key')==key
                            and outputs is not None and prior.get('output_hashes')==outputs):
                        events.append({'task':task.name,'status':'validated_cache_hit'})
                        continue
                if shutil.disk_usage(repo).free < plan.get('min_free_bytes',2*1024**3):
                    raise RuntimeError('Free storage is below the reserved minimum')
                if not task.command or not all(isinstance(v,str) for v in task.command):
                    raise ValueError('Invalid argument-vector command')
                record.update(status='running',key=key,input_hashes=inputs)
                atomic_json(marker,record)
                with (root/f'{task.name}.log').open('a',encoding='utf-8') as log:
                    log.write('\n'+utc_now()+' '+repr(task.command)+'\n');log.flush()
                    completed=subprocess.run(task.command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                                             timeout=timeout,check=False)
                if completed.returncode:
                    raise RuntimeError(f'{task.name} exited {completed.returncode}; see {root/task.name}.log')
                outputs=_files(repo,task.outputs)
                guard=_guard(repo,task.guard)
                record.update(status='completed',finished=utc_now(),output_hashes=outputs,guard=guard)
                atomic_json(marker,record)
                events.append({'task':task.name,'status':'completed'})
            except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as exc:
                record.update(status='failed',finished=utc_now(),error=str(exc))
                atomic_json(marker,record)
                atomic_json(root/'last_execution.json',{'status':'failed','events':events,'failed_task':task.name})
                raise
    result={'status':'requested_tasks_completed','events':events,
            'note':'Task completion does not imply every hypothesis is established or every run converged.'}
    atomic_json(root/'last_execution.json',result)
    return result
