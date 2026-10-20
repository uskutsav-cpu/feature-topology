"""Extend a passed full original-study freeze with the additional research audit.

This gate cannot replace, skip, or relax the original physical-input gate. It
adds source checks and replays the diagnostic analysis. A negative predictive
finding remains a completed comparison; no novelty or journal readiness follows
from a successful evidence freeze.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.collect_additional_evaluations import validate_cell
from scripts.diagnostic_comparison import comparison_rows, run_comparison
from scripts.freeze_results import verify_manifest
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import evaluation_plan


def close_replay(actual, expected, path='result'):
    """Compare all replay fields, allowing only cross-platform floating roundoff."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Analysis replay keys differ: '+path)
        for key in actual:
            close_replay(actual[key], expected[key], path+'.'+key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Analysis replay lengths differ: '+path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            close_replay(left, right, f'{path}[{index}]')
    elif (isinstance(actual, (float, int)) and not isinstance(actual, bool)
          and isinstance(expected, (float, int)) and not isinstance(expected, bool)):
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
            raise ValueError('Analysis replay numeric difference: '+path)
    elif isinstance(actual, bool) or isinstance(expected, bool):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('Analysis replay value differs: '+path)
    elif actual != expected:
        raise ValueError('Analysis replay value differs: '+path)


def freeze(repo, original_manifest, evidence, output):
    repo = Path(repo).resolve()
    original_manifest = Path(original_manifest).resolve()
    original = verify_manifest(repo, original_manifest)
    plan = evaluation_plan(repo)
    protocol_path = repo/'configs/strengthening_diagnostic_protocol_v1.json'
    protocol = load_json(protocol_path.read_bytes())
    paths = set()
    for kind, folder in [('ood', 'ood_evaluation'), ('variability', 'additional_variability')]:
        root = repo/'results'/folder
        collection = load_json((root/'collection.json').read_bytes())
        if (collection.get('kind') != kind or collection.get('expected') != 140
                or collection.get('complete') != 140 or collection.get('missing') != []
                or collection.get('invalid') != []):
            raise ValueError('Additional evaluation collection incomplete: '+kind)
        for row in plan:
            rid = row['record']['run_id']
            result_path, receipt_path = root/(rid+'.json'), root/'receipts'/(rid+'.json')
            receipt = load_json(receipt_path.read_bytes())
            validate_cell(row, load_json(result_path.read_bytes()), receipt, digest(result_path), kind, repo)
            if kind == 'variability':
                for name, expected in receipt['source_sha256'].items():
                    if digest(repo/name) != expected:
                        raise ValueError('Additional metric source changed: '+name)
            elif receipt['evaluator_sha256'] != digest(repo/'scripts/evaluate_nuisance_shift.py'):
                raise ValueError('Nuisance evaluator source changed')
        paths.update(root.rglob('*.json'))
    aliases = repo/'results/checkpoint_aliases'
    rows, hashes, missing = comparison_rows(repo, evidence, repo/'results/ood_evaluation',
                                           repo/'results/additional_variability', aliases, protocol)
    if len(rows) != 140 or missing:
        raise ValueError('Additional comparison has missing cells or undefined predictors')
    comparison_root = repo/'results/additional_diagnostic_comparison'
    result = load_json((comparison_root/'comparison.json').read_bytes())
    stored_rows = load_json((comparison_root/'comparison_rows.json').read_bytes())
    close_replay(rows, stored_rows, 'comparison_rows')
    if (result.get('status') != 'complete_additional_analysis'
            or result.get('protocol_sha256') != digest(protocol_path)
            or result.get('cells') != 140 or result.get('missing') != []
            or result.get('input_sha256') != hashes):
        raise ValueError('Additional comparison is not bound to the complete current source table')
    replay = run_comparison(rows, protocol)
    for key, value in replay.items():
        close_replay(value, result[key], key)
    paths.update(aliases.rglob('*.json'))
    paths.update(comparison_root.rglob('*.json'))
    files = dict(original['files'])
    for path in paths:
        if not path.resolve().is_relative_to(repo):
            raise ValueError('Additional frozen file escapes repository')
        files[str(path.relative_to(repo))] = digest(path)
    for name, expected in hashes.items():
        supplied = Path(name)
        if supplied.is_absolute() or '..' in supplied.parts or files.get(name) != expected:
            raise ValueError('Additional analysis consumed an input outside the combined freeze: '+name)
    combined = dict(original)
    combined.update(files=files, original_freeze_sha256=digest(original_manifest),
                    additional_research=dict(schema='feature-topology.additional-research-freeze.v1',
                        protocol_sha256=digest(protocol_path), cells=140,
                        exact_checkpoint_alias_audits=140, analysis_replay_passed=True,
                        incremental_benefit_supported={row['outcome']: row['incremental_benefit_supported']
                                                       for row in result['contrasts']},
                        scope='Complete additional evidence audit; no novelty/acceptance guarantee'))
    output = Path(output)
    if not output.resolve().is_relative_to(repo):
        raise ValueError('Combined freeze must be saved inside the repository')
    publish(output.parent, {output.name: (json.dumps(combined, indent=2, allow_nan=False)+'\n').encode()})
    return dict(ready=True, cells=140, original_files=len(original['files']), combined_files=len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.original_manifest, args.evidence, args.output)))
