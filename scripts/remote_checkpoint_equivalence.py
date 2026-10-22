"""Check exact inference-state equality of frozen checkpoint serialization aliases.

The metric profiles can use final.pt while the original risk-selection helper
uses step_N.pt. Their archive hashes differ because serialization containers and
training metadata differ. Step identity alone is insufficient: this audit checks
every model-state tensor, including the centered predictor's frozen branch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_release_evidence import digest, publish
from scripts.remote_nuisance_evaluation import evaluation_plan, verify_input
from scripts.remote_variability_baselines import download


def inference_state_digest(model_state):
    h = hashlib.sha256()
    for key, tensor in sorted(model_state.items()):
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        h.update(len(header).to_bytes(8, 'big'))
        h.update(header)
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compare_states(selected, final, config, step):
    import torch
    for state in [selected, final]:
        if state.get('config') != config or state.get('step') != step:
            raise ValueError('Checkpoint alias task/step mismatch')
    left, right = selected['model'], final['model']
    if set(left) != set(right):
        raise ValueError('Checkpoint alias model-state keys differ')
    for key in left:
        if (left[key].dtype != right[key].dtype or left[key].shape != right[key].shape
                or not torch.isfinite(left[key]).all() or not torch.isfinite(right[key]).all()
                or not torch.equal(left[key], right[key])):
            raise ValueError('Checkpoint alias model-state tensors differ: '+key)
    a, b = inference_state_digest(left), inference_state_digest(right)
    if a != b:
        raise ValueError('Checkpoint alias tensor-content digests differ')
    return a


def audit_cell(row, output):
    import torch
    from scripts.evaluate_nuisance_shift import matched_checkpoint
    record = row['record']
    with tempfile.TemporaryDirectory(prefix='feature-topology-alias-') as temporary:
        temporary = Path(temporary)
        archive = temporary/'input.tar.gz'
        download(row['input_tag'], record['asset'], archive)
        directory = verify_input(archive, record, temporary/'inputs')
        selected_path, selected = matched_checkpoint(directory, .1)
        if selected_path is None:
            raise ValueError('Alias audit cell did not reach target')
        final_path = directory/'final.pt'
        selected_state = torch.load(selected_path, map_location='cpu', weights_only=True)
        final_state = torch.load(final_path, map_location='cpu', weights_only=True)
        tensor_sha = compare_states(selected_state, final_state, record['config'], selected['step'])
        result = dict(schema='feature-topology.checkpoint-alias-equivalence.v1',
                      run_id=record['run_id'], selected_step=selected['step'],
                      selected_checkpoint=selected_path.name, selected_checkpoint_sha256=digest(selected_path),
                      final_checkpoint_sha256=digest(final_path),
                      exact_model_state_equality=True, inference_state_sha256=tensor_sha,
                      tensor_count=len(final_state['model']),
                      frozen_reference_included=any(k.startswith('initial.') for k in final_state['model']),
                      input_index=row['index'], input_index_sha256=row['index_sha256'],
                      input_archive_sha256=record['archive_sha256'],
                      source_commit=os.environ.get('GITHUB_SHA'), workflow_run=os.environ.get('GITHUB_RUN_ID'),
                      auditor_sha256=digest(Path(__file__)), torch=torch.__version__,
                      scope='Exact inference tensors only; optimizer/serialization metadata are not equated')
    publish(Path(output), {record['run_id']+'.json':
                          (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid alias-audit shard')
    rows = evaluation_plan(args.repo)
    selected = [row for row in rows if int(row['record']['run_id'], 16) % args.shards == args.shard]
    for row in selected:
        result = audit_cell(row, args.output)
        print(json.dumps(dict(run_id=result['run_id'], exact_equality=True)), flush=True)


if __name__ == '__main__':
    main()
