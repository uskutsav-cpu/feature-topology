"""Repair dSprites CKA baselines from the seeded initial network.

Persistent-homology and probe measurements depend only on saved final features.
This script preserves those measurements and recomputes the only field that was
previously derived from an ambient RNG-dependent initial model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.run_dsprites import build, prepare
from src.metrics.geometry import cka
from src.training.checkpoints import atomic_json


def feature_layers(model, x, device):
    with torch.no_grad():
        batches = [model.network.representations(batch.to(device)) for batch in x.split(128)]
    return [torch.cat([batch[layer].cpu() for batch in batches]).numpy() for layer in range(3)]


def initial_model(config, device):
    state = torch.get_rng_state()
    torch.manual_seed(config["seed"])
    model = build(config).to(device).eval()
    torch.set_rng_state(state)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return model, digest.hexdigest()


def repair(repo, device):
    repo = Path(repo)
    root = repo / "results/dsprites"
    data, dataset_id = prepare(repo / "data/dsprites/dsprites.npz", root)
    repaired = []
    for metric_path in sorted(root.glob("runs/*/metrics.json")):
        run = metric_path.parent
        metric = json.loads(metric_path.read_text())
        config = json.loads((run / "config.json").read_text())
        if metric.get("schema") != "feature-topology.dsprites-metrics.v1":
            raise ValueError(f"Unexpected metric schema: {run}")
        if config["dataset_id"] != dataset_id or metric.get("dataset_id") != dataset_id:
            raise ValueError(f"Dataset identity mismatch: {run}")
        backup = run / "metrics.pre_seeded_initial.json"
        if not backup.exists():
            shutil.copyfile(metric_path, backup)
        initial, initial_sha = initial_model(config, device)
        final = build(config).to(device).eval()
        state = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
        final.load_state_dict(state["model"])
        before = feature_layers(initial, data["test"]["x"], device)
        after = feature_layers(final, data["test"]["x"], device)
        drifts = []
        for layer, (a, b) in enumerate(zip(before, after), start=1):
            score = cka(a, b)
            drift = None if not np.isfinite(score) else float(1 - score)
            metric["layers"][layer - 1]["cka_drift"] = drift
            drifts.append(drift)
        metric["initial_model_sha256"] = initial_sha
        metric["initial_model_reconstructed_from_seed"] = True
        metric["cka_baseline_repaired"] = True
        metric.pop("nonfinite_metrics", None)
        atomic_json(metric_path, metric)
        repaired.append(dict(run_id=run.name, initial_model_sha256=initial_sha, cka_drift=drifts))
    return dict(schema="feature-topology.dsprites-initial-metric-repair.v1",
                dataset_id=dataset_id, repaired_runs=repaired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = repair(args.repo, args.device)
    atomic_json(Path(args.output), result)
    print(json.dumps(dict(repaired=len(result["repaired_runs"]))) )
