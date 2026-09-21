"""Produce final statistics and figures only from a verified results freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.freeze_results import exact_control_paths, verify_manifest, sha256
from research_ext.report import analyze
from src.training.checkpoints import fingerprint,atomic_json
from research_ext.catalog import ABLATION_PRODUCTION, PRODUCTION
from scripts.image_statistics import summarize_images
from scripts.ood_statistics import summarize_nuisance_shift
from scripts.relevance_statistics import summarize_relevance
from scripts.certification_statistics import summarize_certification
from scripts.width_statistics import plot_width_gate
from scripts.v3_results_document import render as render_v3_results


def verify_consumed_inputs(repo, manifest, input_hashes_path):
    """Require every catalog input actually read to belong to the freeze."""
    repo=Path(repo).resolve()
    frozen=manifest.get('files',{})
    consumed=json.loads(Path(input_hashes_path).read_text())
    for name,observed in consumed.items():
        supplied=Path(name)
        source=(supplied if supplied.is_absolute() else repo/supplied).resolve()
        if not source.is_relative_to(repo):
            raise ValueError(f'Analysis input escapes repository: {name}')
        relative=source.relative_to(repo).as_posix()
        expected=frozen.get(relative)
        if expected is None:
            raise ValueError(f'Analysis consumed input absent from frozen manifest: {relative}')
        if observed!=expected or sha256(source)!=expected:
            raise ValueError(f'Analysis consumed changed frozen input: {relative}')
    return consumed


def verify_frozen_paths(repo, manifest, paths):
    """Require each directly consumed analysis input to be freeze-bound.

    The generic synthetic report records its own input hashes.  The image,
    OOD, certification, and study-discovery paths are read by smaller helpers,
    so they pass through this explicit guard instead.  This also rejects an
    otherwise valid-looking unlisted run dropped into a globbed directory
    after the results freeze.
    """
    repo=Path(repo).resolve()
    frozen=manifest.get('files',{})
    verified={}
    for supplied in paths:
        source=Path(supplied).resolve()
        if not source.is_relative_to(repo):
            raise ValueError(f'Analysis input escapes repository: {supplied}')
        relative=source.relative_to(repo).as_posix()
        expected=frozen.get(relative)
        if expected is None:
            raise ValueError(f'Analysis consumed input absent from frozen manifest: {relative}')
        observed=sha256(source) if source.is_file() else None
        if observed!=expected:
            raise ValueError(f'Analysis consumed changed frozen input: {relative}')
        verified[relative]=observed
    return verified


def main(args):
    repo=Path(args.repo).resolve()
    manifest=verify_manifest(repo,args.manifest)
    output=Path(args.output).resolve()
    if not output.is_relative_to(repo):
        raise ValueError('Final analysis output must be inside the repository')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Final analysis output must be empty to exclude stale artifacts')
    reports=[]
    relevance_groups=[]
    plan=json.loads((repo/'configs/completion/analysis_plan.json').read_text())
    primary_index=json.loads((repo/'configs/completion/primary_inputs_v3.json').read_text())
    verify_frozen_paths(repo,manifest,[repo/'configs/completion/analysis_plan.json',
                                      repo/'configs/completion/primary_inputs_v3.json'])
    primary_ids={row['run_id'] for row in primary_index['runs']}
    for name in [*plan['synthetic_studies'], 'ood']:
        study=repo/'results'/name
        if name=='main':
            groups=[dict(path=str(study),gammas=plan['primary']['gammas'],seeds=plan['primary']['seeds'],
                         run_ids=sorted(primary_ids),profile=PRODUCTION,source='primary')]
        else:
            groups=[]
            study_manifest=study/'manifest.json'
            verify_frozen_paths(repo,manifest,[study_manifest])
            for group in json.loads(study_manifest.read_text()):
                run_ids={row['run_id'] for row in group['runs']}
                shared=run_ids & primary_ids
                if shared and shared != run_ids:
                    raise ValueError(f"Partially shared primary group is ambiguous: {name}/{group['path']}")
                # Portable group identity; never follow an old machine's absolute path.
                path=repo/'results/main' if shared else study/Path(group['path']).name
                groups.append(dict(path=str(path),gammas=sorted({r['gamma'] for r in group['runs']}),
                                   seeds=sorted({r['seed'] for r in group['runs']}),run_ids=sorted(run_ids),
                                   profile=PRODUCTION if shared else ABLATION_PRODUCTION,
                                   source='shared_primary' if shared else 'ablation',
                                   data_condition=group['condition']))
        for group in groups:
            destination=output/name/(Path(group['path']).name if group['source']!='shared_primary'
                                     else 'shared_primary')
            result=analyze([group['path']],destination,target=.1,profile=fingerprint(group['profile']),
                gammas=group['gammas'],seeds=group['seeds'],repeats=2000,
                transition_repeats=300,rules=[],make_plots=True,run_ids=group['run_ids'])
            verify_consumed_inputs(repo,manifest,destination/'input_hashes.json')
            reports.append(dict(study=name,group=Path(group['path']).name,source=group['source'],
                                run_count=len(group['run_ids']),profile=fingerprint(group['profile']),
                                output=destination.relative_to(repo).as_posix()))
            if name=='relevance':
                relevance_groups.append((group['data_condition']['relevance'],destination))
    image_inputs=[]
    for study in ['rotated_digits','dsprites','cifar10','cifar100']:
        for summary_path in sorted((repo/'results'/study).glob('runs/*/summary.json')):
            image_inputs.append(summary_path)
            if json.loads(summary_path.read_text()).get('status')!='diverged':
                image_inputs.append(summary_path.parent/'metrics.json')
    verify_frozen_paths(repo,manifest,image_inputs)
    summarize_images(repo,output/'images')
    relevance_statistics=summarize_relevance(relevance_groups,output/'relevance_dependence')
    ood_manifest=repo/'results/ood_evaluation/manifest.json'
    verify_frozen_paths(repo,manifest,[ood_manifest])
    ood_statistics=summarize_nuisance_shift(ood_manifest,output/'ood_evaluation')
    trained_manifest=repo/'results/exact_trained_circles/manifest.json'
    circle_manifest=repo/'results/circle_quotient/manifest.json'
    certification_inputs=[*exact_control_paths(repo),trained_manifest,circle_manifest,
                          repo/'results/completion/formal/formal_status.json']
    certification_inputs.extend(
        repo/'results/exact_trained_circles'/row['certificate_file']
        for row in json.loads(trained_manifest.read_text())['records'])
    certification_inputs.extend(
        repo/'results/circle_quotient/runs'/row['run_id']/'quotient.json'
        for row in json.loads(circle_manifest.read_text()))
    verify_frozen_paths(repo,manifest,certification_inputs)
    certification=summarize_certification(repo,output/'certification')
    width_gate_path=repo/'results/analysis/width_scaling_gate.json'
    verify_frozen_paths(repo,manifest,[width_gate_path])
    width_gate=json.loads(width_gate_path.read_text())
    if (width_gate.get('schema')!='feature-topology.width-scaling-result.v1'
            or width_gate.get('status')!='complete'):
        raise ValueError('Frozen finite-width terminology result is incomplete')
    width_scaling={key:width_gate[key] for key in [
        'metric','terminology','phase_transition_language_allowed','clauses','scaling','scope']}
    plot_width_gate(width_gate,output/'width_scaling')
    claim_audit=render_v3_results(repo,output,sha256(Path(args.manifest)),reports,
                                  relevance_statistics,ood_statistics,
                                  certification,width_scaling)
    verify_manifest(repo,args.manifest)
    files={p.relative_to(repo).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'analysis_manifest.json',dict(schema='feature-topology.final-analysis.v1',reports=reports,
                relevance_statistics=relevance_statistics,ood_statistics=ood_statistics,
                certification=certification,width_scaling=width_scaling,
                claim_audit=claim_audit,
                frozen_manifest_sha256=sha256(Path(args.manifest)),files=files,
                frozen_input_files=len(manifest['files']),
                scope='Separate condition-level seed analyses and distinct image-study schemas; exact certificates retain their own domain-specific claims.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',default='results/final_analysis')
    main(parser.parse_args())
