"""Produce final statistics and figures only from a verified results freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.freeze_results import verify_manifest, sha256
from research_ext.report import analyze
from src.training.checkpoints import fingerprint,atomic_json
from research_ext.catalog import ABLATION_PRODUCTION, PRODUCTION
from scripts.image_statistics import summarize_images


def main(args):
    repo=Path(args.repo).resolve()
    manifest=verify_manifest(repo,args.manifest)
    output=Path(args.output).resolve()
    if not output.is_relative_to(repo):
        raise ValueError('Final analysis output must be inside the repository')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Final analysis output must be empty to exclude stale artifacts')
    reports=[]
    plan=json.loads((repo/'configs/completion/analysis_plan.json').read_text())
    for name in [*plan['synthetic_studies'], 'ood']:
        study=repo/'results'/name
        if name=='main':
            groups=[dict(path=str(study),gammas=plan['primary']['gammas'],seeds=plan['primary']['seeds'])]
        else:
            groups=[]
            for group in json.loads((study/'manifest.json').read_text()):
                # Portable group identity; never follow an old machine's absolute path.
                path=study/Path(group['path']).name
                groups.append(dict(path=str(path),gammas=sorted({r['gamma'] for r in group['runs']}),
                                   seeds=sorted({r['seed'] for r in group['runs']})))
        for group in groups:
            destination=output/name/Path(group['path']).name
            profile=PRODUCTION if name=='main' else ABLATION_PRODUCTION
            result=analyze([group['path']],destination,target=.1,profile=fingerprint(profile),
                gammas=group['gammas'],seeds=group['seeds'],repeats=2000,
                transition_repeats=300,rules=[],make_plots=True)
            reports.append(dict(study=name,group=Path(group['path']).name,output=destination.relative_to(repo).as_posix()))
    summarize_images(repo,output/'images')
    verify_manifest(repo,args.manifest)
    files={p.relative_to(repo).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'analysis_manifest.json',dict(schema='feature-topology.final-analysis.v1',reports=reports,
                frozen_manifest_sha256=sha256(Path(args.manifest)),files=files,
                frozen_input_files=len(manifest['files']),
                scope='Separate condition-level seed analyses and distinct image-study schemas; exact certificates retain their own domain-specific claims.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',default='results/final_analysis')
    main(parser.parse_args())
