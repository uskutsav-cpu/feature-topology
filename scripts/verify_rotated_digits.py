"""Replay held-out classification from every recovered digit checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.data.images import rotated_digits
from src.models.cnn import CNN
from src.models.mlp import ScaledModel
from src.training.train import evaluate
from src.training.checkpoints import atomic_json
from scripts.completion_inventory import sha256


def verify(root):
    torch.set_num_threads(2)
    data=rotated_digits(seed=123,rotations=4)
    identities={name:set(data[name][3]) for name in ['train','validation','test']}
    if any(identities[a]&identities[b] for a,b in [('train','test'),('train','validation'),('validation','test')]):
        raise ValueError('Base identity leakage')
    records=[]
    for path in sorted(Path(root).glob('runs/*/final.pt')):
        state=torch.load(path,map_location='cpu',weights_only=True)
        config=state['config']
        model=ScaledModel(CNN(width=config.get('width',16)),config['gamma'],centered=True)
        model.load_state_dict(state['model'])
        loss,accuracy=evaluate(model,*data['test'][:2],batch=256)
        saved=json.loads((path.parent/'metrics.json').read_text())
        passed=abs(loss-saved['test_loss'])<1e-4 and abs(accuracy-saved['test_accuracy'])<=1/len(data['test'][0])
        records.append(dict(run_id=path.parent.name,gamma=config['gamma'],seed=config['seed'],
                   checkpoint_sha256=sha256(path),metrics_sha256=sha256(path.parent/'metrics.json'),
                   replay_test_loss=loss,replay_test_accuracy=accuracy,
                   saved_test_loss=saved['test_loss'],saved_test_accuracy=saved['test_accuracy'],passed=passed))
        print(json.dumps(dict(run_id=path.parent.name,passed=passed)),flush=True)
    expected={(g,s) for g in [.125,.5,1.,4.,16.,64.,128.] for s in range(5)}
    complete=len(records)==35 and {(r['gamma'],r['seed']) for r in records}==expected
    return dict(complete=complete and all(r['passed'] for r in records),records=records,
                identity_split_disjoint=True,loss_tolerance=1e-4,
                accuracy_tolerance=1/len(data['test'][0]),
                scope='Classification replay from saved checkpoints on original held-out digit identities; saved probe/PH values are not recomputed here.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=verify(args.root)
    atomic_json(Path(args.output),result)
    sys.exit(0 if result['complete'] else 2)
