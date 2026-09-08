"""Independent CPU replication using frozen, previously calibrated learning rates.

Never writes to results/main or modifies primary calibration. All failures remain
in the manifest. Use --limit for explicit subsets; a rerun reuses completed runs.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.checkpoints import atomic_json
from src.training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/validation/cpu_replication.json')
    parser.add_argument('--output', default='results/continuation')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    study_path = root/'study_config.json'
    if study_path.exists() and json.loads(study_path.read_text()) != cfg:
        raise RuntimeError('Study configuration changed; use a new output directory')
    atomic_json(study_path, cfg)
    environment = dict(
        python=platform.python_version(), platform=platform.platform(),
        torch=str(torch.__version__), numpy=np.__version__, device='cpu',
        source_commit=cfg['source_commit'],
        note='Separate environment: do not pool seeds with the original macOS/PyTorch production results.')
    environment_path = root/'environment.json'
    if environment_path.exists():
        old = json.loads(environment_path.read_text())
        if any(old.get(key) != environment[key] for key in ('python', 'torch', 'numpy', 'device')):
            raise RuntimeError('Numerical environment changed; use a new output directory')
    else:
        atomic_json(environment_path, environment)
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    registry = {row['run_id']: row for row in manifest}
    attempted = 0
    for gamma in cfg['gammas']:
        for seed in cfg['seeds']:
            if args.limit is not None and attempted >= args.limit:
                return
            config = dict(cfg['training'], gamma=gamma, seed=seed,
                          lr=cfg['learning_rates'][str(gamma)],
                          calibration_id=cfg['calibration_id'], study=cfg['study'])
            result = train(config, root/'runs')
            attempted += 1
            final = root/'runs'/result['run_id']/'final.pt'
            row = dict(run_id=result['run_id'], gamma=gamma, seed=seed,
                       status=result['status'],
                       final_sha256=hashlib.sha256(final.read_bytes()).hexdigest() if final.exists() else None)
            if result['status'] == 'converged' and not final.exists():
                raise RuntimeError(f'Missing final checkpoint for {result["run_id"]}; restore the checkpoint archive or use a new output directory')
            registry[row['run_id']] = row
            manifest = sorted(registry.values(), key=lambda r: (r['gamma'], r['seed']))
            atomic_json(manifest_path, manifest)
            print(json.dumps(dict(row, completed=len(manifest))), flush=True)


if __name__ == '__main__':
    main()
