"""Validate the complete CIFAR-10/CIFAR-100 study against saved artifacts.

This is deliberately a post-training verifier: it reloads each trained network on
CPU and reproduces its held-out test evaluation, in addition to checking the
frozen calibration and every required gamma/seed cell.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import inspect_run
from scripts.run_cifar import CIFARResNet, ScaledModel, evaluate, load_data
from src.training.checkpoints import atomic_json, fingerprint


GAMMAS = [0.125, 0.5, 1.0, 4.0, 16.0, 64.0, 128.0]
SEEDS = range(5)


def finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return value is None or finite_number(value) or isinstance(value, str)


def validate_dataset(repo, name):
    root = repo / "results" / name.lower()
    frozen_path = root / "gamma_to_lr.json"
    if not frozen_path.is_file():
        raise ValueError(f"Missing frozen calibration: {frozen_path}")
    frozen = json.loads(frozen_path.read_text())
    selection = frozen.get("selection", {})
    expected_keys = {str(gamma) for gamma in GAMMAS}
    if set(selection) != expected_keys:
        raise ValueError(f"Unexpected calibration cells for {name}: {sorted(selection)}")
    if frozen.get("calibration_steps") != 2000:
        raise ValueError(f"Unexpected calibration budget for {name}")
    classes = 10 if name == "CIFAR10" else 100
    data = load_data(name, str(repo / "data" / "cifar"), download=False)
    if {part: len(values[0]) for part, values in data.items()} != {
        "train": 45000, "validation": 5000, "test": 10000
    }:
        raise ValueError(f"Unexpected {name} data split")
    records = []
    for gamma in GAMMAS:
        lr = selection[str(gamma)].get("lr")
        if not finite_number(lr) or lr <= 0:
            raise ValueError(f"Invalid frozen rate for {name}, gamma={gamma}")
        for seed in SEEDS:
            config = dict(classes=classes, dataset=name, batch_size=128,
                          target_loss=0.2, eval_every=500, gamma=gamma,
                          seed=seed, lr=lr, max_steps=35200)
            run = root / "runs" / fingerprint(config)
            record = inspect_run(run, check_tensors=True)
            metric_path = run / "metrics.json"
            reps_path = run / "representations.npz"
            if not metric_path.is_file() or not reps_path.is_file():
                raise ValueError(f"Missing analysis artifact: {run}")
            metric = json.loads(metric_path.read_text())
            with np.load(reps_path) as arrays:
                expected_arrays = {f"{stage}_h{layer}" for stage in ("initial", "final")
                                   for layer in range(1, 5)} | {"input_logit_jacobian_singular_values"}
                if set(arrays.files) != expected_arrays or any(
                    not np.isfinite(arrays[key]).all() for key in arrays.files
                ):
                    raise ValueError(f"Invalid representation artifact: {run}")
            layers = metric.get("layers")
            saved = metric.get("test", {})
            if not isinstance(layers, list) or len(layers) != 4:
                raise ValueError(f"Unexpected layer metric shape: {run}")
            if not finite_tree(metric) or not all(finite_number(saved.get(key)) for key in ("loss", "accuracy")):
                raise ValueError(f"Non-finite saved test metric: {run}")
            state = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
            torch.manual_seed(seed)
            model = ScaledModel(CIFARResNet(classes), gamma).cpu()
            model.network.load_state_dict(state["network"])
            replayed = evaluate(model, data["test"], "cpu")
            if abs(replayed["loss"] - saved["loss"]) > 1e-4 or abs(replayed["accuracy"] - saved["accuracy"]) > 1e-8:
                raise ValueError(f"Held-out replay mismatch: {run}")
            records.append(dict(run_id=record["run_id"], gamma=gamma, seed=seed,
                                status=record["status"], checkpoint_sha256=record["final_sha256"],
                                saved_test=saved, replayed_test=replayed))
    return dict(dataset=name, expected_runs=35, validated_runs=len(records), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    datasets = [validate_dataset(repo, name) for name in ("CIFAR10", "CIFAR100")]
    result = dict(schema="feature-topology.cifar-validation.v1", complete=True,
                  datasets=datasets,
                  scope="CIFAR robustness study only; it makes no latent-manifold, injectivity, or topology claim.")
    atomic_json(Path(args.output), result)
    print(json.dumps({"complete": True, "validated_runs": sum(x["validated_runs"] for x in datasets)}))


if __name__ == "__main__":
    main()
