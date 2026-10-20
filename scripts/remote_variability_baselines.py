"""Compute additional VCI baselines on the frozen grid and matched checkpoint.

This is the separately versioned strengthening comparison, not a replacement of
any original metric profile. The original checkpoint and frozen analysis arrays
are verified before use. No weights are trained and no OOD outcome is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import evaluation_plan, verify_input


def download(tag, asset, destination):
    url = ('https://github.com/uskutsav-cpu/feature-topology/releases/download/'
           + urllib.parse.quote(tag, safe='')+'/'+urllib.parse.quote(asset, safe=''))
    request = urllib.request.Request(url, headers={'User-Agent': 'Feature-Topology-VCI-baselines'})
    with urllib.request.urlopen(request, timeout=90) as stream, destination.open('wb') as output:
        while chunk := stream.read(1024*1024):
            output.write(chunk)


def evaluate_cell(row, output, repo):
    import numpy as np
    import torch
    from scripts.compute_metrics import ANALYSIS_DEFAULTS, analysis_condition
    from scripts.evaluate_nuisance_shift import matched_checkpoint
    from src.metrics.variability import variability_baselines
    from src.training.train import build

    torch.set_num_threads(2)
    repo, output = Path(repo), Path(output)
    record = row['record']
    index = load_json((repo/row['index']).read_bytes())
    protocol = repo/'configs/strengthening_diagnostic_protocol_v1.json'
    with tempfile.TemporaryDirectory(prefix='feature-topology-vci-') as temporary:
        temporary = Path(temporary)
        archive, arrays = temporary/'inputs.tar.gz', temporary/'analysis_data.npz'
        download(row['input_tag'], record['asset'], archive)
        directory = verify_input(archive, record, temporary/'inputs')
        download(row['input_tag'], index['analysis_data']['asset'], arrays)
        if digest(arrays) != index['analysis_data']['sha256']:
            raise ValueError('Frozen analysis grid checksum mismatch')
        with np.load(arrays, allow_pickle=False) as frozen:
            condition = {**ANALYSIS_DEFAULTS, **json.loads(str(frozen['data_config']))}
            if condition != analysis_condition(record['config']):
                raise ValueError('Frozen analysis grid has a different task condition')
            x, labels = torch.from_numpy(frozen['gx'].copy()), frozen['gy'].copy()
        if len(x) != 10000 or x.shape[1] != record['config'].get('dimension', 16):
            raise ValueError('Additional baselines require the original 100-by-100 grid')
        checkpoint, selected = matched_checkpoint(directory, .1)
        if checkpoint is None:
            raise ValueError('Frozen comparison cell did not reach the specified risk')
        checkpoint_sha = digest(checkpoint)
        if checkpoint_sha != record['files'][checkpoint.name]:
            raise ValueError('Selected checkpoint checksum mismatch')
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if state.get('config') != record['config'] or state.get('step') != selected['step']:
            raise ValueError('Checkpoint state differs from training identity')
        model = build(record['config'], record['seed'])
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            features = np.concatenate([model.network.representations(chunk)[-1].numpy()
                                       for chunk in x.split(1024)])
        metrics = variability_baselines(features, labels)
    result = dict(schema='feature-topology.additional-variability-baselines.v1',
                  run_id=record['run_id'], gamma=record['gamma'], seed=record['seed'],
                  training_nuisance_condition=row['training_condition'],
                  selected_step=selected['step'], actual_training_loss=selected['training_loss'],
                  checkpoint=checkpoint.name, checkpoint_sha256=checkpoint_sha,
                  layer=record['config']['depth'], metrics=metrics,
                  protocol_sha256=digest(protocol),
                  scope='Last trainable hidden representation on the frozen IID grid; no training')
    payload = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    receipt = dict(schema='feature-topology.additional-variability-receipt.v1',
                   run_id=record['run_id'], input_tag=row['input_tag'], input_index=row['index'],
                   input_index_sha256=row['index_sha256'],
                   input_archive_sha256=record['archive_sha256'],
                   source_member_sha256=record['files'],
                   analysis_data_sha256=index['analysis_data']['sha256'],
                   protocol_sha256=digest(protocol), evaluation_sha256=hashlib.sha256(payload).hexdigest(),
                   source_commit=os.environ.get('GITHUB_SHA'), workflow_run=os.environ.get('GITHUB_RUN_ID'),
                   source_sha256={str(path.relative_to(repo)): digest(path) for path in [
                       repo/'scripts/remote_variability_baselines.py',
                       repo/'src/metrics/variability.py', repo/'scripts/evaluate_nuisance_shift.py']},
                   host=dict(platform=platform.platform(), python=platform.python_version(),
                             torch=torch.__version__, numpy=np.__version__))
    publish(output, {record['run_id']+'.json': payload,
                     'receipts/'+record['run_id']+'.json':
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
    for row in selected:
        result = evaluate_cell(row, args.output, args.repo)
        print(json.dumps({key: result[key] for key in ['run_id', 'gamma', 'seed']}), flush=True)
    publish(Path(args.output), {'shard_manifest.json': (json.dumps(dict(
        schema='feature-topology.additional-variability-shard.v1', shard=args.shard,
        shards=args.shards, expected=len(selected), completed=len(selected)))+'\n').encode()})


if __name__ == '__main__':
    main()
