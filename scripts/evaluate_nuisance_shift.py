"""Evaluate matched-risk torus checkpoints in frozen nuisance environments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.torus import dataset
from src.training.checkpoints import atomic_json
from src.training.train import build, evaluate


EVALUATION_ENVIRONMENTS = ("iid", "concentrated", "spurious", "unseen")


def matched_checkpoint(run, target):
    summary = json.loads((run/"summary.json").read_text())
    eligible = [row for row in summary.get("history", []) if row["training_loss"] <= target]
    if not eligible:
        return None, None
    selected = min(eligible, key=lambda row: row["step"])
    path = run/f"step_{selected['step']:07d}.pt"
    if not path.exists() and (run/"final.pt").exists():
        final = torch.load(run/"final.pt", weights_only=True, map_location="cpu")
        if final.get("step") == selected["step"]:
            path = run/"final.pt"
    if not path.exists():
        raise FileNotFoundError(f"Matched-risk checkpoint is missing: {path}")
    return path, selected


def evaluate_run(run, target=.1, samples=5000):
    run = Path(run)
    config = json.loads((run/"config.json").read_text())
    checkpoint, training = matched_checkpoint(run, target)
    if checkpoint is None:
        return {"run_id": run.name, "status": "target_not_reached", "target_loss": target}
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = build(config, config["seed"])
    model.load_state_dict(state["model"])
    model.eval()
    common = {k: config[k] for k in
              ["dimension", "manifold", "swap", "relevance", "relevance_mode"] if k in config}
    environments = {}
    for index, name in enumerate(EVALUATION_ENVIRONMENTS):
        x, y, _, _ = dataset(
            samples, seed=2026092000+index, nuisance_condition=name,
            distribution_split="test", **common,
        )
        loss, accuracy = evaluate(model, x, y)
        environments[name] = {"loss": loss, "accuracy": accuracy, "samples": samples}
    with checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "schema": "feature-topology.nuisance-shift-evaluation.v1",
        "run_id": run.name,
        "status": "evaluated",
        "training_nuisance_condition": config.get("nuisance_condition", "iid"),
        "gamma": config["gamma"], "seed": config["seed"],
        "target_loss": target, "selected_step": training["step"],
        "actual_training_loss": training["training_loss"],
        "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha256,
        "environments": environments,
        "scope": "Predictive OOD evaluation; no injectivity or quotient claim.",
    }


def run(runs, output, target=.1, samples=5000):
    runs, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = list(runs.glob("*/summary.json"))
    if not summaries:
        summaries = list(runs.rglob("runs/*/summary.json"))
    for summary in sorted(summaries):
        row = evaluate_run(summary.parent, target, samples)
        rows.append(row)
        atomic_json(output/f"{summary.parent.name}.json", row)
    atomic_json(output/"manifest.json", {
        "schema": "feature-topology.nuisance-shift-manifest.v1",
        "target_loss": target, "samples_per_environment": samples,
        "evaluation_environments": list(EVALUATION_ENVIRONMENTS), "runs": rows,
    })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", default="results/ood_evaluation")
    parser.add_argument("--target-loss", type=float, default=.1)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    run(args.runs, args.output, args.target_loss, args.samples)
