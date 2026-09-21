"""Validate the complete CIFAR-10/CIFAR-100 study against saved artifacts.

This is deliberately a post-training verifier: it reloads each trained network on
CPU and reproduces its held-out test evaluation, in addition to checking the
frozen calibration and every required gamma/seed cell.
"""
import argparse
import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return value is None or finite_number(value) or isinstance(value, str)


def cached_replay(cache, key, binding):
    row = cache.get("records", {}).get(key)
    if row is None or any(row.get(name) != value for name, value in binding.items()):
        return None
    replayed = row.get("replayed_test", {})
    if not all(finite_number(replayed.get(name)) for name in ("loss", "accuracy")):
        raise ValueError(f"Non-finite cached CIFAR replay: {key}")
    return replayed


def validate_dataset(repo, name, cache, cache_path, allow_missing=False):
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
    missing = []
    for gamma in GAMMAS:
        lr = selection[str(gamma)].get("lr")
        if not finite_number(lr) or lr <= 0:
            raise ValueError(f"Invalid frozen rate for {name}, gamma={gamma}")
        for seed in SEEDS:
            config = dict(classes=classes, dataset=name, batch_size=128,
                          target_loss=0.2, eval_every=500, gamma=gamma,
                          seed=seed, lr=lr, max_steps=35200)
            run = root / "runs" / fingerprint(config)
            if not (run / "summary.json").is_file():
                if allow_missing:
                    missing.append({"gamma": gamma, "seed": seed,
                                    "run_id": fingerprint(config)})
                    continue
                raise ValueError(f"Missing CIFAR run: {run}")
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
            binding = {
                "dataset": name, "run_id": record["run_id"],
                "checkpoint_sha256": record["final_sha256"],
                "metrics_sha256": sha256(metric_path),
                "representations_sha256": sha256(reps_path),
                "saved_test": saved,
            }
            key = f"{name}/{record['run_id']}"
            replayed = cached_replay(cache, key, binding)
            if replayed is None:
                state = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
                torch.manual_seed(seed)
                model = ScaledModel(CIFARResNet(classes), gamma).cpu()
                model.network.load_state_dict(state["network"])
                replayed = evaluate(model, data["test"], "cpu")
                cache["records"][key] = {**binding, "replayed_test": replayed}
                atomic_json(cache_path, cache)
            if abs(replayed["loss"] - saved["loss"]) > 1e-4 or abs(replayed["accuracy"] - saved["accuracy"]) > 1e-8:
                raise ValueError(f"Held-out replay mismatch: {run}")
            records.append(dict(run_id=record["run_id"], gamma=gamma, seed=seed,
                                status=record["status"], checkpoint_sha256=record["final_sha256"],
                                metrics_sha256=binding["metrics_sha256"],
                                representations_sha256=binding["representations_sha256"],
                                saved_test=saved, replayed_test=replayed))
    return dict(dataset=name, expected_runs=35, validated_runs=len(records),
                missing=missing, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="results/completion/cifar_validation_progress.json")
    parser.add_argument("--incremental", action="store_true",
                        help="Replay available runs without weakening final completeness")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = repo / cache_path
    verifier_sha256 = sha256(Path(__file__))
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if (cache.get("schema") != "feature-topology.cifar-validation-progress.v1"
                or cache.get("verifier_sha256") != verifier_sha256
                or not isinstance(cache.get("records"), dict)):
            raise ValueError("CIFAR validation cache does not match this verifier")
    else:
        cache = {"schema": "feature-topology.cifar-validation-progress.v1",
                 "verifier_sha256": verifier_sha256, "complete": False, "records": {}}
    # A restarted audit is incomplete until every current artifact binding has
    # been revisited, even when all expensive replay values remain reusable.
    cache["complete"] = False
    cache.pop("validated_runs", None)
    atomic_json(cache_path, cache)
    datasets = [validate_dataset(repo, name, cache, cache_path,
                                 allow_missing=args.incremental)
                for name in ("CIFAR10", "CIFAR100")]
    validated_runs=sum(x["validated_runs"] for x in datasets)
    complete=validated_runs==70 and all(not x["missing"] for x in datasets)
    if not args.incremental and not complete:
        raise ValueError(f"CIFAR replay coverage {validated_runs}/70")
    cache.update(complete=complete, validated_runs=validated_runs)
    atomic_json(cache_path, cache)
    result = dict(schema="feature-topology.cifar-validation.v1", complete=complete,
                  datasets=datasets,
                  replay_cache=cache_path.relative_to(repo).as_posix(),
                  replay_cache_sha256=sha256(cache_path),
                  scope="CIFAR robustness study only; it makes no latent-manifold, injectivity, or topology claim.")
    atomic_json(Path(args.output), result)
    print(json.dumps({"complete": complete, "validated_runs": validated_runs}))


if __name__ == "__main__":
    main()
