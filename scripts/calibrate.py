import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.train import train
from src.training.fls import reference_lr
from src.training.checkpoints import atomic_json, fingerprint


def calibrate(config, root, trials_root=None):
    root = Path(root)
    destination = root/"gamma_to_lr.json"
    calibration_id = fingerprint(config)
    if destination.exists():
        frozen = json.loads(destination.read_text())
        if frozen["calibration_id"] != calibration_id:
            raise RuntimeError("Frozen calibration differs: use a new experiment directory")
        return frozen
    selection = {}
    for gamma in config["gammas"]:
        candidates = []
        for multiplier in config.get("lr_multipliers", [.125, .25, .5, 1, 2, 4, 8]):
            lr = reference_lr(gamma, config.get("depth", 4), config.get("base_lr", .05))*multiplier
            runs = []
            for seed in config.get("calibration_seeds", [900]):
                c = {k:v for k,v in config.items() if k not in ["gammas", "seeds", "lr_multipliers", "calibration_seeds", "base_lr"]}
                c.update(gamma=gamma, lr=lr, seed=seed)
                result = train(c, Path(trials_root) if trials_root else root/"trials", save=False)
                runs.append(result)
                last = result["history"][-1] if result["history"] else {}
                print(json.dumps(dict(gamma=gamma, lr=lr, seed=seed, status=result["status"],
                                      loss=last.get("training_loss"), step=last.get("step"))), flush=True)
            stable = all(r["status"] != "diverged" for r in runs)
            reached = all(r["status"] == "converged" for r in runs)
            loss = np.mean([r["history"][-1]["training_loss"] for r in runs]) if stable else 1e20
            step = np.mean([r["history"][-1]["step"] for r in runs])
            candidates.append(dict(lr=lr, stable=stable, reached=reached, loss=float(loss), step=float(step)))
        eligible = [r for r in candidates if r["stable"]]
        if not eligible:
            raise RuntimeError(f"No stable learning rate for gamma={gamma}")
        winner = min(eligible, key=lambda r:(not r["reached"], r["step"] if r["reached"] else r["loss"], r["loss"]))
        selection[str(gamma)] = dict(**winner, candidates=candidates)
        atomic_json(root/"calibration_progress.json", selection)
    frozen = dict(calibration_id=calibration_id, config=config, selection=selection,
                  policy="Stable target-reaching candidates ranked by steps, then loss; otherwise lowest training loss, flagged unreached")
    atomic_json(destination, frozen)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--trials", help="Shared trial cache; reuse identical pilot trials")
    args = parser.parse_args()
    calibrate(json.loads(Path(args.config).read_text()), args.output, args.trials)
