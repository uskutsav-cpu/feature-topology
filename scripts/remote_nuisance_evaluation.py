"""Complete the frozen 140-cell nuisance-shift evaluation from saved inputs.

This changes execution placement only. The existing evaluator selects the same
first stored checkpoint at loss <= 0.1, evaluates the four existing environments,
and uses the existing 5,000 examples/environment and fixed evaluation seeds.
Training checkpoints are never updated. Each source archive/member is verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_release_evidence import digest, load_json, publish, safe_name


def evaluation_plan(repo):
    repo = Path(repo)
    design = load_json((repo/'configs/analysis_plan_v2.json').read_bytes())
    gammas = design['primary_design']['gammas']
    seeds = design['primary_design']['training_seeds']
    registry = load_json((repo/'configs/completion/production_cohorts_v3.json').read_bytes())
    rows = []
    for cohort in registry['cohorts']:
        if cohort['name'] not in {'primary', 'ood-spurious', 'ood-concentrated', 'ood-unseen'}:
            continue
        index = load_json((repo/cohort['index']).read_bytes())
        for record in index['runs']:
            if record['gamma'] not in gammas or record['seed'] not in seeds:
                continue
            config = record['config']
            condition = config.get('nuisance_condition', 'iid')
            if (config.get('relevance', 0) != 0 or config.get('swap', False)
                    or config.get('manifold', 'torus') != 'torus'
                    or condition not in design['nuisance_shift']['training_conditions']):
                raise ValueError('Nuisance evaluation contains a different scientific task')
            rows.append(dict(record=record, input_tag=cohort['input_tag'],
                             index=cohort['index'], index_sha256=digest(repo/cohort['index']),
                             training_condition=condition))
    expected = {(condition, float(gamma), int(seed))
                for condition in design['nuisance_shift']['training_conditions']
                for gamma in gammas for seed in seeds}
    observed = [(row['training_condition'], float(row['record']['gamma']),
                 int(row['record']['seed'])) for row in rows]
    if (len(rows) != 140 or len(set(observed)) != len(rows) or set(observed) != expected
            or len({row['record']['run_id'] for row in rows}) != len(rows)):
        raise ValueError('Frozen nuisance-evaluation plan is incomplete or duplicated')
    return sorted(rows, key=lambda row: row['record']['run_id'])


def verify_input(archive_path, record, target):
    """Extract only after validating the complete archive and every member."""
    if digest(archive_path) != record['archive_sha256']:
        raise ValueError('Nuisance input archive checksum mismatch')
    expected = {f"{record['run_id']}/{name}": value for name, value in record['files'].items()}
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Nuisance input archive membership mismatch')
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError('Nuisance input must contain regular files only')
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected[member.name]:
                raise ValueError('Nuisance input member checksum mismatch')
        # Every source member has passed verification before anything is written.
        for member in members:
            destination = target.joinpath(*safe_name(member.name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, destination.open('wb') as output:
                while chunk := stream.read(1024*1024):
                    output.write(chunk)
    directory = target/record['run_id']
    config = load_json((directory/'config.json').read_bytes())
    summary = load_json((directory/'summary.json').read_bytes())
    if config != record['config'] or summary.get('config') != config or summary.get('run_id') != record['run_id']:
        raise ValueError('Nuisance input training identity mismatch')
    return directory


def evaluate_cell(row, output):
    import torch
    from scripts.evaluate_nuisance_shift import evaluate_run
    torch.set_num_threads(2)
    record = row['record']
    output = Path(output)
    result_path = output/f"{record['run_id']}.json"
    receipt_path = output/'receipts'/f"{record['run_id']}.json"
    if result_path.exists() and receipt_path.exists():
        receipt = load_json(receipt_path.read_bytes())
        if (receipt.get('input_archive_sha256') == record['archive_sha256']
                and receipt.get('evaluation_sha256') == digest(result_path)):
            return load_json(result_path.read_bytes())
        raise ValueError('Existing nuisance evaluation differs from its source-bound receipt')
    with tempfile.TemporaryDirectory(prefix='feature-topology-ood-') as temporary:
        temporary = Path(temporary)
        archive = temporary/'input.tar.gz'
        url = ('https://github.com/uskutsav-cpu/feature-topology/releases/download/'
               + urllib.parse.quote(row['input_tag'], safe='')+'/'
               + urllib.parse.quote(record['asset'], safe=''))
        request = urllib.request.Request(url, headers={'User-Agent': 'Feature-Topology-OOD-evaluation'})
        with urllib.request.urlopen(request, timeout=90) as stream, archive.open('wb') as saved:
            while chunk := stream.read(1024*1024):
                saved.write(chunk)
        directory = verify_input(archive, record, temporary/'inputs')
        result = evaluate_run(directory, target=.1, samples=5000)
        if result.get('status') == 'evaluated':
            checkpoint = result['checkpoint']
            if result['checkpoint_sha256'] != record['files'][checkpoint]:
                raise ValueError('Evaluated checkpoint is not a frozen input')
    payload = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    receipt = dict(schema='feature-topology.hosted-nuisance-evaluation.v1',
                   run_id=record['run_id'], input_tag=row['input_tag'],
                   input_index=row['index'], input_index_sha256=row['index_sha256'],
                   input_archive_sha256=record['archive_sha256'],
                   source_member_sha256=record['files'],
                   evaluation_sha256=hashlib.sha256(payload).hexdigest(),
                   source_commit=os.environ.get('GITHUB_SHA'),
                   workflow_run=os.environ.get('GITHUB_RUN_ID'),
                   host=dict(platform=platform.platform(), python=platform.python_version(),
                             torch=torch.__version__),
                   evaluator_sha256=digest(Path(__file__).with_name('evaluate_nuisance_shift.py')),
                   scope='Same frozen checkpoint selection, data seeds, samples, and evaluator; no training')
    publish(output, {result_path.name: payload,
                     f"receipts/{record['run_id']}.json":
                         (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid evaluation shard')
    rows = evaluation_plan(args.repo)
    selected = [row for row in rows if int(row['record']['run_id'], 16) % args.shards == args.shard]
    results = []
    for row in selected:
        result = evaluate_cell(row, args.output)
        results.append(result)
        print(json.dumps({k: result.get(k) for k in ['run_id', 'status', 'gamma', 'seed']}), flush=True)
    manifest = dict(schema='feature-topology.nuisance-shift-evaluation-shard.v1',
                    shard=args.shard, shards=args.shards, expected=len(selected),
                    target_loss=.1, samples_per_environment=5000,
                    evaluation_environments=['iid', 'concentrated', 'spurious', 'unseen'], runs=results)
    publish(Path(args.output), {'shard_manifest.json': (json.dumps(manifest, indent=2)+'\n').encode()})


if __name__ == '__main__':
    main()
