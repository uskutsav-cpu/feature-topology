import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train
from src.training.checkpoints import atomic_json


def run_sweep(calibration, output, seeds, max_steps, reuse_runs=None, gammas=None):
    frozen = json.loads(Path(calibration).read_text())
    original = frozen["config"]
    runs = []
    reusable = []
    if reuse_runs:
        for path in Path(reuse_runs).glob("*/summary.json"):
            result = json.loads(path.read_text())
            if result["status"] == "converged" and (path.parent/"final.pt").exists():
                reusable.append((path.parent, result))
    def signature(c):
        defaults = dict(dimension=16, width=256, depth=4, manifold="torus", swap=False,
                        relevance=0., relevance_mode="periodic", centered=True)
        return {k:v for k,v in {**defaults, **c}.items() if k not in ["max_steps", "calibration_id"]}
    for gamma in (gammas if gammas is not None else original["gammas"]):
        for seed in seeds:
            c = {k:v for k,v in original.items() if k not in ["gammas", "seeds", "lr_multipliers", "calibration_seeds", "base_lr"]}
            c.update(gamma=gamma, seed=seed, lr=frozen["selection"][str(gamma)]["lr"],
                     max_steps=max_steps, calibration_id=frozen["calibration_id"])
            match = next(((path, r) for path, r in reusable if signature(r["config"]) == signature(c)), None)
            if match:
                path, result = match
                destination = Path(output)/"runs"/result["run_id"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.symlink_to(os.path.relpath(path.resolve(), destination.parent.resolve()), target_is_directory=True)
            else:
                result = train(c, Path(output)/"runs")
            runs.append(dict(run_id=result["run_id"], gamma=gamma, seed=seed, status=result["status"]))
            atomic_json(Path(output)/"manifest.json", runs)
            print(json.dumps(runs[-1]), flush=True)
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument("--reuse-runs")
    a = p.parse_args()
    run_sweep(a.calibration, a.output, a.seeds, a.max_steps, a.reuse_runs)
