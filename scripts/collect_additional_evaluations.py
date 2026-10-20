"""Validate and collect hosted evaluations against the frozen 140-cell plan."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import evaluation_plan


def validate_cell(row, result, receipt, result_sha, kind, repo):
    record = row['record']
    if (result.get('run_id') != record['run_id'] or receipt.get('run_id') != record['run_id']
            or result.get('seed') != record['seed'] or result.get('gamma') != record['gamma']
            or result.get('training_nuisance_condition') != row['training_condition']
            or receipt.get('input_index') != row['index']
            or receipt.get('input_tag') != row['input_tag']
            or receipt.get('input_index_sha256') != row['index_sha256']
            or receipt.get('input_archive_sha256') != record['archive_sha256']
            or receipt.get('source_member_sha256') != record['files']
            or receipt.get('evaluation_sha256') != result_sha
            or result.get('checkpoint_sha256') != record['files'].get(result.get('checkpoint'))):
        raise ValueError('Evaluation identity/checksum differs from frozen sources')
    if kind == 'ood':
        if (result.get('schema') != 'feature-topology.nuisance-shift-evaluation.v1'
                or result.get('status') != 'evaluated' or result.get('target_loss') != .1
                or set(result.get('environments', {})) != {'iid', 'concentrated', 'spurious', 'unseen'}):
            raise ValueError('Incomplete or different nuisance evaluation')
        for value in result['environments'].values():
            if (value.get('samples') != 5000 or not 0 <= value.get('accuracy', -1) <= 1
                    or not isinstance(value.get('loss'), (int, float))
                    or not math.isfinite(value['loss']) or value['loss'] < 0):
                raise ValueError('Invalid nuisance environment measurement')
    elif kind == 'variability':
        index = load_json((Path(repo)/row['index']).read_bytes())
        protocol_sha = digest(Path(repo)/'configs/strengthening_diagnostic_protocol_v1.json')
        if (result.get('schema') != 'feature-topology.additional-variability-baselines.v1'
                or receipt.get('analysis_data_sha256') != index['analysis_data']['sha256']
                or receipt.get('protocol_sha256') != protocol_sha
                or result.get('protocol_sha256') != protocol_sha
                or result.get('layer') != record['config']['depth']
                or result.get('metrics', {}).get('samples') != 10000):
            raise ValueError('Different variability protocol, layer, or analysis grid')
    else:
        raise ValueError('Unknown evaluation kind')


def collect(repo, artifact_root, output, kind):
    repo, artifact_root = Path(repo), Path(artifact_root)
    plan = evaluation_plan(repo)
    expected = {row['record']['run_id']: row for row in plan}
    results, files, source_files = [], {}, {}
    for rid, row in sorted(expected.items()):
        candidates = [path for path in artifact_root.rglob(rid+'.json')
                      if path.parent.name != 'receipts']
        if len(candidates) != 1:
            raise ValueError(f'Expected one evaluation for {rid}, found {len(candidates)}')
        path = candidates[0]
        receipt_path = path.parent/'receipts'/path.name
        result, receipt = load_json(path.read_bytes()), load_json(receipt_path.read_bytes())
        validate_cell(row, result, receipt, digest(path), kind, repo)
        # The checkpoint name must agree with the recorded selection; final.pt is
        # allowed by the original helper and checked against training metadata by
        # the downstream diagnostic table join.
        if result['checkpoint'] != 'final.pt' and result['checkpoint'] != f"step_{result['selected_step']:07d}.pt":
            raise ValueError('Selected step/checkpoint name mismatch')
        results.append(result)
        files[path.name] = path.read_bytes()
        files['receipts/'+path.name] = receipt_path.read_bytes()
        source_files[str(path.relative_to(artifact_root))] = digest(path)
        source_files[str(receipt_path.relative_to(artifact_root))] = digest(receipt_path)
    actual = {p.stem for p in artifact_root.rglob('*.json')
              if p.parent.name != 'receipts' and p.name != 'shard_manifest.json'}
    if actual != set(expected):
        raise ValueError('Unexpected cells outside the frozen evaluation plan')
    if kind == 'ood':
        manifest = dict(schema='feature-topology.nuisance-shift-manifest.v1', target_loss=.1,
                        samples_per_environment=5000,
                        evaluation_environments=['iid', 'concentrated', 'spurious', 'unseen'], runs=results)
    else:
        manifest = dict(schema='feature-topology.additional-variability-manifest.v1', runs=results)
    files['manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    files['collection.json'] = (json.dumps(dict(
        schema='feature-topology.hosted-evaluation-collection.v1', kind=kind,
        expected=len(expected), complete=len(results), missing=[], invalid=[],
        input_index_sha256={row['index']: row['index_sha256'] for row in plan},
        artifact_file_sha256=source_files,
        scope='Source-bound hosted evaluations; not a declaration of a full results freeze'),
        indent=2, allow_nan=False)+'\n').encode()
    publish(output, files)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=['ood', 'variability'], required=True)
    args = parser.parse_args()
    manifest = collect(args.repo, args.artifacts, args.output, args.kind)
    print(json.dumps(dict(kind=args.kind, complete=len(manifest['runs']))))
