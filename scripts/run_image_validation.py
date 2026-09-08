"""Calibrated real-image rotation validation using offline sklearn digits."""
import argparse
import copy
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from src.data.images import rotated_digits
from src.models.cnn import CNN
from src.models.mlp import ScaledModel
from src.training.checkpoints import fingerprint, atomic_json, save_checkpoint
from src.training.train import evaluate
from src.training.fls import reference_lr
from src.metrics.geometry import cka, effective_rank
from src.metrics.probes import evaluate_probes
from src.metrics.persistence import persistence


def image_train(config, root, data, save=True, device="cpu"):
    root = Path(root)/fingerprint(config)
    done = root/"summary.json"
    if done.exists():
        return json.loads(done.read_text())
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/"config.json", config)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    model = ScaledModel(CNN(width=config.get("width", 16)), config["gamma"], centered=True).to(device)
    optimizer = torch.optim.SGD(model.network.parameters(), lr=config["lr"])
    rng = torch.Generator().manual_seed(config["seed"]+100)
    x, y, _, _ = data["train"]
    vx, vy, _, _ = data["validation"]
    x, y, vx, vy = x.to(device), y.to(device), vx.to(device), vy.to(device)
    history, start = [], 0
    resume = root/"resume.pt"
    if resume.exists():
        s = torch.load(resume, weights_only=False, map_location="cpu")
        model.load_state_dict(s["model"]); optimizer.load_state_dict(s["optimizer"])
        rng.set_state(s["rng"]); history = s["history"]; start = s["step"]
    status = "budget_exhausted"
    for step in range(start, config["max_steps"]+1):
        if (step == 0 or step % 100 == 0 or step == config["max_steps"]) and (not history or history[-1]["step"] != step):
            loss, acc = evaluate(model, x, y, batch=256)
            vl, va = evaluate(model, vx, vy, batch=256)
            if not np.isfinite(loss) or loss > 1e6:
                status = "diverged"; break
            history.append(dict(step=step, training_loss=loss, training_accuracy=acc,
                                validation_loss=vl, validation_accuracy=va))
            if save:
                save_checkpoint(resume, dict(model={k:v.cpu() for k,v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
                                             rng=rng.get_state(), history=history, step=step))
            if loss <= config["target_loss"]:
                status = "converged"; break
        if step == config["max_steps"]:
            break
        ids = torch.randint(len(x), (64,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[ids.to(device)]), y[ids.to(device)])
        if not torch.isfinite(loss):
            status = "diverged"; break
        loss.backward(); optimizer.step()
    result = dict(config=config, status=status, history=history, run_id=root.name, execution_device=device)
    if save and status != "diverged":
        save_checkpoint(root/"final.pt", dict(model={k:v.cpu() for k,v in model.state_dict().items()}, config=config, step=step))
    atomic_json(done, result)
    return result


def features(model, x):
    with torch.no_grad():
        return torch.cat([model.network.representations(a)[-1] for a in x.split(256)]).numpy()


def analyze(config, root, data):
    root = Path(root)/fingerprint(config)
    if (root/"metrics.json").exists():
        return
    torch.manual_seed(config["seed"])
    model = ScaledModel(CNN(width=config.get("width", 16)), config["gamma"], centered=True)
    gx, gy, gp, base_ids = data["grid"]
    h0 = features(model, gx)
    model.load_state_dict(torch.load(root/"final.pt", weights_only=False, map_location="cpu")["model"])
    h = features(model, gx)
    x, y, p, _ = data["train"]; tx, ty, tp, _ = data["test"]
    train_h, test_h = features(model, x), features(model, tx)
    probes = evaluate_probes(train_h, test_h, y.numpy(), ty.numpy(), p, tp, config["seed"])
    test_loss, test_accuracy = evaluate(model, tx, ty, batch=256)
    loop_rows = []
    for identity in np.unique(base_ids):
        ids = base_ids == identity
        stats, _ = persistence(h[ids], size=36, repeats=1, maxdim=1)
        initial_stats, _ = persistence(h0[ids], size=36, repeats=1, maxdim=1)
        angles = gp[ids]
        values = h[ids]
        derivative = (np.roll(values, -1, axis=0)-np.roll(values, 1, axis=0))/(2*np.pi/18)
        loop_rows.append(dict(base_id=int(identity), initial_ph=initial_stats[0], final_ph=stats[0],
                              median_rotation_finite_difference_norm=float(np.median(np.linalg.norm(derivative, axis=1)))))
    np.savez_compressed(root/"representations.npz", initial=h0, final=h, phi=gp, base_ids=base_ids)
    atomic_json(root/"metrics.json", dict(test_loss=test_loss, test_accuracy=test_accuracy,
          cka_drift=1-cka(h0, h), effective_rank=effective_rank(h), probes=probes, rotation_loops=loop_rows,
          limitation="Discrete rotated handwritten digits; interpolation, digit symmetry and rotation-label ambiguity preclude exact S1/injectivity claims"))


def main(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    data = rotated_digits(seed=123, rotations=4)
    gammas = [.125, .5, 1., 4., 16., 64., 128.]
    frozen_path = root/"gamma_to_lr.json"
    if not frozen_path.exists():
        selection = {}
        for gamma in gammas:
            trials = []
            for mult in [.25, 1., 4.]:
                c = dict(gamma=gamma, lr=reference_lr(gamma, 3, .05)*mult, seed=900,
                         max_steps=args.calibration_steps, target_loss=.5, width=16)
                r = image_train(c, root/"calibration", data, save=True, device=args.device)
                print(json.dumps(dict(gamma=gamma, lr=c["lr"], status=r["status"])), flush=True)
                if r["status"] != "diverged":
                    trials.append((r["status"] != "converged", r["history"][-1]["step"] if r["status"] == "converged" else r["history"][-1]["training_loss"], c["lr"]))
            if not trials:
                raise RuntimeError(f"No stable image rate at {gamma}")
            selection[str(gamma)] = min(trials)[2]
            atomic_json(root/"calibration_progress.json", selection)
        atomic_json(frozen_path, dict(selection=selection, calibration_steps=args.calibration_steps, target_loss=.5))
    frozen = json.loads(frozen_path.read_text())
    for gamma in gammas:
        for seed in range(args.seeds):
            c = dict(gamma=gamma, lr=frozen["selection"][str(gamma)], seed=seed,
                     max_steps=args.max_steps, target_loss=.5, width=16)
            result = image_train(c, root/"runs", data, device=args.device)
            if result["status"] != "diverged":
                analyze(c, root/"runs", data)
            print(json.dumps(dict(gamma=gamma, seed=seed, status=result["status"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/rotated_digits")
    p.add_argument("--calibration-steps", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    main(p.parse_args())
