"""Calibrated small-ReLU FLS experiment on a polygonal circle.

The image-graph result is exact for the sampled polygon and extracted affine
network maps up to the stated floating-point/LP tolerance. It is not a proof
about the continuum between polygon vertices or the full torus experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from src.data.circle import circle_dataset
from src.metrics.quotient import image_graph
from src.models.mlp import MLP, ScaledModel
from src.training.checkpoints import atomic_json, fingerprint, save_checkpoint
from src.training.fls import reference_lr


@torch.no_grad()
def evaluate(model, x, y):
    logits = model(x)
    return float(F.cross_entropy(logits, y)), float((logits.argmax(1) == y).float().mean())


def config_id(config):
    return fingerprint({k: config[k] for k in sorted(config)})


def train(config, root, save=True):
    directory = Path(root) / config_id(config)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "config.json", config)
    torch.manual_seed(config["seed"])
    model = ScaledModel(
        MLP(dimension=2, width=config["width"], depth=config["depth"], classes=2),
        gamma=config["gamma"], centered=True,
    )
    optimizer = torch.optim.SGD(model.network.parameters(), lr=config["lr"])
    x, y, _ = circle_dataset(config["n_train"], seed=4001)
    validation_x, validation_y, _ = circle_dataset(config["n_validation"], seed=4002)
    generator = torch.Generator().manual_seed(config["seed"] + 707)
    history = []
    status = "budget_exhausted"
    for step in range(config["max_steps"] + 1):
        if step == 0 or step % config["eval_every"] == 0 or step == config["max_steps"]:
            loss, accuracy = evaluate(model, x, y)
            validation_loss, validation_accuracy = evaluate(model, validation_x, validation_y)
            if not np.isfinite(loss) or loss > 1e6:
                status = "diverged"
                break
            history.append(dict(step=step, training_loss=loss, training_accuracy=accuracy,
                                validation_loss=validation_loss,
                                validation_accuracy=validation_accuracy))
            if loss <= config["target_loss"]:
                status = "converged"
                break
        if step == config["max_steps"]:
            break
        ids = torch.randint(len(x), (config["batch_size"],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[ids]), y[ids])
        if not torch.isfinite(loss):
            status = "diverged"
            break
        loss.backward()
        optimizer.step()
    result = dict(run_id=directory.name, config=config, status=status, history=history)
    if save and status != "diverged":
        save_checkpoint(directory / "final.pt", dict(model=model.state_dict(), config=config, step=step))
    atomic_json(summary_path, result)
    return result


def affine_layers(network, through_layer):
    layers = []
    for layer in network.hidden[: through_layer + 1]:
        layers.append((layer.weight.detach().numpy(), layer.bias.detach().numpy(), True))
    return layers


def analyze(config, root, polygon_points=720, tolerance=1e-7):
    directory = Path(root) / config_id(config)
    target = directory / "quotient.json"
    if target.exists():
        return json.loads(target.read_text())
    torch.manual_seed(config["seed"])
    model = ScaledModel(
        MLP(dimension=2, width=config["width"], depth=config["depth"], classes=2),
        gamma=config["gamma"], centered=True,
    )
    initial = copy.deepcopy(model.network)
    model.load_state_dict(torch.load(directory / "final.pt", weights_only=False)["model"])
    theta = np.arange(polygon_points) * (2 * np.pi / polygon_points)
    polygon = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    result = dict(
        polygon_points=polygon_points,
        tolerance=tolerance,
        initial=[image_graph(polygon, affine_layers(initial, layer), tolerance)
                 for layer in range(config["depth"])],
        final=[image_graph(polygon, affine_layers(model.network, layer), tolerance)
               for layer in range(config["depth"])],
        qualification=("Betti numbers are for the numerically resolved image graph of a "
                       "720-edge polygonal circle. Pairwise image intersections use LP; "
                       "results are tolerance-dependent and do not certify a smooth continuum."),
    )
    atomic_json(target, result)
    return result


def main(args):
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    gammas = [0.125, 1.0, 16.0, 128.0]
    frozen_path = root / "gamma_to_lr.json"
    common = dict(width=16, depth=2, n_train=4000, n_validation=1000,
                  max_steps=2048, eval_every=64, batch_size=128, target_loss=0.08)
    if not frozen_path.exists():
        selection = {}
        for gamma in gammas:
            candidates = []
            for multiplier in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
                config = dict(**common, gamma=gamma,
                              lr=reference_lr(gamma, common["depth"], 0.05) * multiplier,
                              seed=900)
                trial = train(config, root / "calibration", save=False)
                last = trial["history"][-1]
                candidates.append(dict(lr=config["lr"], status=trial["status"],
                                       loss=last["training_loss"], step=last["step"]))
                print(json.dumps(dict(stage="calibration", gamma=gamma,
                                      lr=config["lr"], status=trial["status"])), flush=True)
            stable = [c for c in candidates if c["status"] != "diverged"]
            if not stable:
                raise RuntimeError(f"No stable circle learning rate for gamma={gamma}")
            winner = min(stable, key=lambda c: (c["status"] != "converged",
                                                c["step"] if c["status"] == "converged" else c["loss"],
                                                c["loss"]))
            selection[str(gamma)] = dict(**winner, candidates=candidates)
            atomic_json(root / "calibration_progress.json", selection)
        atomic_json(frozen_path, dict(selection=selection, common=common,
                                      policy="stable target-reaching rate, then fewest evaluated steps"))
    frozen = json.loads(frozen_path.read_text())
    manifest = []
    for gamma in gammas:
        for seed in range(5):
            config = dict(**common, gamma=gamma, seed=seed,
                          lr=frozen["selection"][str(gamma)]["lr"])
            result = train(config, root / "runs")
            if result["status"] != "diverged":
                quotient = analyze(config, root / "runs", args.polygon_points, args.tolerance)
            else:
                quotient = None
            manifest.append(dict(gamma=gamma, seed=seed, run_id=result["run_id"],
                                 status=result["status"], quotient=quotient))
            atomic_json(root / "manifest.json", manifest)
            print(json.dumps(dict(stage="main", gamma=gamma, seed=seed,
                                  status=result["status"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/circle_quotient")
    parser.add_argument("--polygon-points", type=int, default=720)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    main(parser.parse_args())
