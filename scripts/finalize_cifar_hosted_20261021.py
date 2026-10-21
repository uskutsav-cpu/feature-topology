"""Finish unchanged CIFAR collection/provenance/replay on a hosted workspace.

No training dispatch, parameter change, publication or full-study readiness claim.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_remote_cifar import collect
from scripts.cifar_provenance import merge_collection_reports
from scripts.finalize_cifar_queue import promote_multisource
from scripts.reconstruct_cifar_provenance import reconstruct,validate_reconstruction
from scripts.run_cifar_remote_queue import calibration_cells,production_cells,finalize_calibration
from scripts.remote_cifar import cell_id
from src.training.checkpoints import atomic_json

SOURCE='06d5f685538e31542bbbd23a2c32cfde3c239aaf'
SPECIFICATION='configs/completion/cifar_sources_v3.json'
DATASETS={
    'CIFAR10':dict(workflow='39530430490',phase_tag='cifar10-v3-phcache-06d5f68',phase_seeds=[0,2,4],
        original_tag='cifar10-v3-hosted-d06123e',original_source='d06123ebdc4a23e8d4bec976e2fe92793b9795a0',
        historical=['cifar10_original','cifar10_exact_lr']),
    'CIFAR100':dict(workflow='39530446041',phase_tag='cifar100-v3-phcache-06d5f68',phase_seeds=list(range(5)),
        original_tag='cifar100-v3-hosted-b20f3c7',original_source='b20f3c77bd620389dbeb542f0b48789f754a2a8e',
        historical=['cifar100_original'])}


def component_cells(dataset,cells):
    spec=DATASETS[dataset]
    phase=[c for c in cells if c['gamma']==.125 and c['seed'] in spec['phase_seeds']]
    correction=[c for c in cells if dataset=='CIFAR10' and c['gamma']==128.]
    excluded={cell_id(c) for c in [*phase,*correction]}
    original=[c for c in cells if cell_id(c) not in excluded]
    identifiers=[cell_id(c) for group in [original,correction,phase] for c in group]
    if len(identifiers)!=35 or len(set(identifiers))!=35:raise ValueError('CIFAR source components are not a disjoint complete frozen grid')
    return original,correction,phase


def observed(workflow):
    for attempt in range(4):
        try:
            return json.loads(subprocess.check_output(['gh','run','view',workflow,'--repo','uskutsav-cpu/feature-topology',
                '--json','status,conclusion,headSha']))
        except subprocess.CalledProcessError:
            if attempt==3:raise
            time.sleep(10)


def transient_collection_failure(report):
    """Retry transport errors only; a content-validation error stays terminal."""
    markers=('connection reset by peer','connection timed out','i/o timeout',
             'tls handshake timeout','unexpected eof','http 502','http 503','http 504')
    errors=report.get('invalid',[])
    return bool(errors) and all(any(marker in str(row.get('error','')).lower()
                                   for marker in markers) for row in errors)


def main(repo, cifar100_workflow=None, output_root='results/completion/cloud_cifar_finalization_20261021'):
    repo=Path(repo).resolve();os.environ['GH_REPO']='uskutsav-cpu/feature-topology'
    root=(repo/output_root).resolve()
    if not root.is_relative_to(repo) or root==repo:raise ValueError('CIFAR execution output must be inside the repository')
    root.mkdir(parents=True,exist_ok=True)
    datasets={name:dict(spec) for name,spec in DATASETS.items()}
    if cifar100_workflow is not None:
        if not str(cifar100_workflow).isdigit():raise ValueError('Invalid CIFAR100 workflow identity')
        datasets['CIFAR100']['workflow']=str(cifar100_workflow)
    def progress(**values):
        value=dict(scope='CIFAR component only; not the full original/combined research freeze',full_study_frozen=False,**values)
        atomic_json(root/'progress.json',value);print(json.dumps(value),flush=True)
    def checked_collection(name,dataset,tag,source,cells):
        for attempt in range(1,4):
            report=collect(repo,tag,cells,root/'transport'/name,SPECIFICATION,install=True,expected_commit=source,bounded_cache=True)
            atomic_json(root/(name+f'.collection-attempt-{attempt}.json'),report)
            if not transient_collection_failure(report) or attempt==3:break
            progress(status='retrying_source_transport',component=name,attempt=attempt)
            time.sleep(5)
        path=root/(name+'.json');atomic_json(path,report)
        if report['missing'] or report['invalid'] or len(report['complete'])!=len(cells):
            raise ValueError('Incomplete unchanged CIFAR component '+name+'; retain partial outputs and resume only the same missing cells after inspecting terminal evidence')
        return path
    try:
        previous=None
        while True:
            try:states={dataset:observed(spec['workflow']) for dataset,spec in datasets.items()}
            except subprocess.CalledProcessError:
                progress(status='observation_retry');time.sleep(60);continue
            if any(value['headSha']!=SOURCE for value in states.values()):raise ValueError('Different scientific source in recorded CIFAR continuation')
            if states!=previous:progress(status='waiting_for_recorded_continuations',workflows=states);previous=states
            if all(value['status']=='completed' and value['conclusion'] for value in states.values()):break
            time.sleep(60)
        complete={}
        for dataset,spec in datasets.items():
            key=dataset.lower();progress(status='collecting_and_binding',dataset=dataset)
            frozen=json.loads((repo/'results'/key/'gamma_to_lr.json').read_text());cells=production_cells(dataset,frozen)
            original,correction,phase=component_cells(dataset,cells)
            calibration=checked_collection(key+'_calibration',dataset,spec['original_tag'],spec['original_source'],calibration_cells(dataset))
            # Recheck the original optimization-only map, never choose new rates.
            finalize_calibration(repo,dataset)
            components=[checked_collection(key+'_production_original',dataset,spec['original_tag'],spec['original_source'],original)]
            if dataset=='CIFAR10':
                components.append(checked_collection(key+'_production_exact_lr',dataset,'cifar10-v3-exact-lr-8816fc0',
                    '8816fc099e3d5135f191597ff50764ad87d80af0',correction))
            components.append(checked_collection(key+'_production_phcache',dataset,spec['phase_tag'],SOURCE,phase))
            phase_root=root/(key+'_phase_source')
            reconstruct(repo,dataset,SOURCE,spec['phase_tag'],phase_root)
            phase_ledger=phase_root/'source_ledger.json'
            if not phase_ledger.exists():raise ValueError('Terminal continuation source reconstruction is unresolved')
            historical=[repo/'results/completion/cifar_source_reconstruction_20261021'/name/'source_ledger.json' for name in spec['historical']]
            for ledger in [*historical,phase_ledger]:validate_reconstruction(repo,json.loads(ledger.read_text()),dataset)
            merged=root/(key+'_production_complete.json')
            merge_collection_reports(repo,dataset,'production',components,merged)
            promote_multisource(repo,dataset,[*historical,phase_ledger],{'calibration':calibration,'production':merged},
                repo/'results/completion'/f'{key}_remote_queue_v3.json')
            complete[dataset]=dict(production=35,calibration=49)
        progress(status='replaying_all_70_heldout_evaluations',collected=complete)
        subprocess.run([sys.executable,'scripts/verify_cifar.py','--repo',str(repo),
            '--output',str(repo/'results/completion/cifar_validation.json')],cwd=repo,check=True)
        validation=json.loads((repo/'results/completion/cifar_validation.json').read_text())
        if validation.get('complete') is not True or sum(d['validated_runs'] for d in validation['datasets'])!=70:
            raise ValueError('CIFAR held-out verification is incomplete')
        progress(status='cifar_component_complete',collected=complete,held_out_replays=70)
    except Exception as error:
        progress(status='attention_required',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--cifar100-workflow',help='Recorded exact-source continuation; scientific configuration and source stay fixed')
    parser.add_argument('--output-root',default='results/completion/cloud_cifar_finalization_20261021',help='In-repository execution namespace preserving earlier failed attempts')
    args=parser.parse_args();main(args.repo,args.cifar100_workflow,args.output_root)
