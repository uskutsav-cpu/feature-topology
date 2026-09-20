"""Replay held-out classification for saved dSprites production checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.run_dsprites import build,GAMMAS,split_ids,SOURCE_BLOB
from scripts.completion_inventory import inspect_run,sha256
from scripts.image_cka import cka_diagnostics
from src.training.checkpoints import atomic_json,fingerprint


def verify(repo,allow_partial=False):
    repo=Path(repo)
    root=repo/'results/dsprites'
    design=json.loads((root/'dataset_design.json').read_text())
    identity=design.pop('dataset_id')
    if fingerprint(design)!=identity or design['official_git_blob']!=SOURCE_BLOB:
        raise ValueError('Invalid frozen dataset identity')
    with np.load(repo/'data/dsprites/dsprites.npz',allow_pickle=False) as archive:
        classes=archive['latents_classes']
    expected_splits=split_ids(classes)
    if any(expected_splits[k].tolist()!=v for k,v in design['partitions'].items()):
        raise ValueError('Frozen split differs from declared identity split')
    ids=np.asarray(design['partitions']['test'])
    labels=classes[ids,1]
    images=np.load(repo/'data/dsprites/official_imgs.npy',mmap_mode='r',allow_pickle=False)
    records=[]
    for metric_path in sorted(root.glob('runs/*/metrics.json')):
        run=metric_path.parent
        checked=inspect_run(run,True)
        config=checked['config']
        if config['dataset_id']!=identity:
            raise ValueError('Checkpoint dataset differs from frozen design')
        state=torch.load(run/'final.pt',map_location='cpu',weights_only=True)
        model=build(config).eval()
        model.load_state_dict(state['model'])
        total_loss=correct=0
        with torch.no_grad():
            for start in range(0,len(ids),32):
                x=torch.from_numpy(np.array(images[ids[start:start+32]],dtype=np.float32)[:,None])
                y=torch.from_numpy(labels[start:start+32].astype(np.int64))
                logits=model(x)
                total_loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
                correct+=(logits.argmax(1)==y).sum().item()
        actual=dict(loss=total_loss/len(ids),accuracy=correct/len(ids))
        metric=json.loads(metric_path.read_text())
        if metric['checkpoint_sha256']!=checked['final_sha256']:
            raise ValueError('Metric checkpoint hash mismatch')
        saved=metric['test']
        if abs(actual['loss']-saved['loss'])>=1e-4 or abs(actual['accuracy']-saved['accuracy'])>1/len(ids):
            raise ValueError(f'Held-out replay mismatch: {run.name}: {actual} vs {saved}')
        undefined=[]
        affected=[index for index,layer in enumerate(metric.get('layers',[]))
                  if layer.get('cka_drift') is None]
        if affected:
            torch.manual_seed(config['seed'])
            initial=build(config).eval()
            digest=hashlib.sha256()
            for name,value in initial.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            if digest.hexdigest()!=metric.get('initial_model_sha256'):
                raise ValueError(f'Initial-model identity mismatch: {run.name}')
            initial_parts=[]; final_parts=[]
            with torch.no_grad():
                for start in range(0,len(ids),32):
                    x=torch.from_numpy(np.array(images[ids[start:start+32]],dtype=np.float32)[:,None])
                    initial_parts.append(initial.network.representations(x))
                    final_parts.append(model.network.representations(x))
            before=[torch.cat([part[layer] for part in initial_parts]).numpy() for layer in range(3)]
            after=[torch.cat([part[layer] for part in final_parts]).numpy() for layer in range(3)]
            for index in affected:
                result=cka_diagnostics(before[index],after[index])
                layer=metric['layers'][index]
                path=f'layers[{index}].cka_drift'
                if (result['score'] is not None or layer.get('cka_status')!=result['status']
                        or not np.isclose(layer.get('cka_initial_centered_gram_norm',np.nan),
                                          result['initial_centered_gram_norm'],rtol=1e-6,atol=1e-12)
                        or not np.isclose(layer.get('cka_current_centered_gram_norm',np.nan),
                                          result['current_centered_gram_norm'],rtol=1e-6,atol=1e-12)):
                    raise ValueError(f'Undefined CKA replay mismatch: {run.name}, layer {index+1}')
                undefined.append(dict(path=path,status=result['status'],
                    initial_centered_gram_norm=result['initial_centered_gram_norm'],
                    current_centered_gram_norm=result['current_centered_gram_norm']))
        if set(metric.get('explicit_undefined_metrics',[]))!={row['path'] for row in undefined}:
            raise ValueError(f'Undefined CKA declaration mismatch: {run.name}')
        records.append(dict(run_id=run.name,gamma=config['gamma'],seed=config['seed'],
                            checkpoint_sha256=checked['final_sha256'],replayed=actual,saved=saved,
                            explicit_undefined_cka=undefined))
    pairs={(r['gamma'],r['seed']) for r in records}
    expected={(g,s) for g in GAMMAS for s in range(5)}
    complete=pairs==expected and len(records)==35
    if len(pairs)!=len(records) or not pairs<=expected:
        raise ValueError('Unexpected or duplicate production run')
    if not complete and not allow_partial:
        raise ValueError(f'Full replay requires 35 measured runs; found {len(records)}')
    return dict(schema='feature-topology.dsprites-replay.v1',complete=complete,
                replayed_runs=len(records),test_examples=len(ids),dataset_id=identity,
                dataset_design_sha256=sha256(root/'dataset_design.json'),records=records,
                scope='CPU replay of held-out classification; probe and PH values not recomputed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=verify(args.repo,args.allow_partial)
    atomic_json(Path(args.output),result)
    print(json.dumps(dict(complete=result['complete'],replayed_runs=result['replayed_runs'])))
