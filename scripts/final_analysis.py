"""Produce final statistics and figures only from a verified results freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.freeze_results import verify_manifest
from research_ext.report import analyze
from src.training.checkpoints import fingerprint,atomic_json
from research_ext.catalog import PRODUCTION


def main(args):
    repo=Path(args.repo).resolve()
    manifest=verify_manifest(repo,args.manifest)
    output=Path(args.output)
    reports=[]
    plan=json.loads((repo/'configs/completion/analysis_plan.json').read_text())
    for name in plan['synthetic_studies']:
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
            result=analyze([group['path']],destination,target=.1,profile=fingerprint(PRODUCTION),
                gammas=group['gammas'],seeds=group['seeds'],repeats=2000,
                transition_repeats=300,rules=[],make_plots=True)
            reports.append(dict(study=name,group=Path(group['path']).name,output=str(destination)))
    atomic_json(output/'analysis_manifest.json',dict(reports=reports,
                frozen_input_files=len(manifest['files']),
                scope='Separate condition-level seed analyses; image studies and exact certificates require their distinct schemas.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',default='results/final_analysis')
    main(parser.parse_args())
