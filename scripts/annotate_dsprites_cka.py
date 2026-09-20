"""Annotate mathematically undefined dSprites CKA without inventing a value.

The original metric file is retained byte-for-byte beside the annotated file.
Only CKA fields whose stored value is null are touched.  The annotation is
derived by replaying the seeded initial network and the hash-bound final
checkpoint, and the ledger records both pre/post hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_dsprites import build, prepare
from scripts.image_cka import cka_diagnostics
from src.training.checkpoints import atomic_json


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def feature_layers(model, images, device):
    with torch.no_grad():
        batches=[model.network.representations(batch.to(device)) for batch in images.split(128)]
    return [torch.cat([batch[layer].cpu() for batch in batches]).numpy()
            for layer in range(3)]


def annotate(repo, device):
    repo=Path(repo).resolve()
    root=repo/'results/dsprites'
    data,dataset_id=prepare(repo/'data/dsprites/dsprites.npz',root)
    records=[]
    for metric_path in sorted(root.glob('runs/*/metrics.json')):
        metric=json.loads(metric_path.read_text())
        affected=[index for index,layer in enumerate(metric.get('layers',[]))
                  if layer.get('cka_drift') is None]
        if not affected:
            continue
        run=metric_path.parent
        config=json.loads((run/'config.json').read_text())
        if config.get('dataset_id')!=dataset_id or metric.get('dataset_id')!=dataset_id:
            raise ValueError(f'Dataset identity mismatch: {run}')
        checkpoint=run/'final.pt'
        if metric.get('checkpoint_sha256')!=sha256(checkpoint):
            raise ValueError(f'Checkpoint binding mismatch: {run}')

        torch.manual_seed(config['seed'])
        initial_model=build(config).to(device).eval()
        final_model=build(config).to(device).eval()
        final_model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        before=feature_layers(initial_model,data['test']['x'],device)
        after=feature_layers(final_model,data['test']['x'],device)

        current_hash=sha256(metric_path)
        backup=run/'metrics.pre_cka_annotation.json'
        if not backup.exists():
            shutil.copyfile(metric_path,backup)
        original_hash=sha256(backup)
        original=json.loads(backup.read_text())
        if any(original.get('layers',[{}])[index].get('cka_drift') is not None for index in affected):
            raise ValueError(f'Existing backup is not the undefined source artifact: {backup}')
        declared=[]
        diagnostics=[]
        for index in affected:
            result=cka_diagnostics(before[index],after[index])
            if result['score'] is not None or result['status']=='defined':
                raise ValueError(f'Stored undefined CKA replays as defined: {run}, layer {index+1}')
            layer=metric['layers'][index]
            layer.update(
                cka_status=result['status'],
                cka_initial_centered_gram_norm=result['initial_centered_gram_norm'],
                cka_current_centered_gram_norm=result['current_centered_gram_norm'])
            path=f'layers[{index}].cka_drift'
            declared.append(path)
            diagnostics.append(dict(layer=index+1,path=path,**result))
        metric['explicit_undefined_metrics']=sorted(declared)
        metric['cka_undefined_annotation']='Exact zero-variance replay; CKA remains null and is not numerically imputed.'
        atomic_json(metric_path,metric)
        records.append(dict(run_id=run.name,checkpoint_sha256=sha256(checkpoint),
                            original_metrics_sha256=original_hash,
                            backup=backup.relative_to(repo).as_posix(),
                            backup_sha256=sha256(backup),
                            annotated_metrics_sha256=sha256(metric_path),
                            input_metrics_sha256=current_hash,
                            diagnostics=diagnostics))
    return dict(schema='feature-topology.dsprites-cka-undefined.v1',dataset_id=dataset_id,
                annotated_runs=len(records),records=records,
                interpretation='Null CKA is retained only when exact replay finds a zero centered-Gram norm; it is a structural collapse outcome, not a numeric effect estimate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=annotate(args.repo,args.device)
    atomic_json(Path(args.output),result)
    print(json.dumps(dict(annotated_runs=result['annotated_runs'])))
