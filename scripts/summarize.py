import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.analysis.statistics import bootstrap_mean, segmented_fit, bootstrap_transition
from src.training.checkpoints import atomic_json


def rows_from(root):
    rows = []
    for run in Path(root).glob("runs/*"):
        summary_path = run/"summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for metric in run.glob("metrics/*/*.json"):
            if metric.name == "options.json":
                continue
            data = json.loads(metric.read_text())
            history = summary["history"]
            matched = min(history, key=lambda h:abs(h["step"]-data["step"]))
            for layer in data["layers"]:
                rows.append(dict(run_id=run.name, profile=metric.parent.name, checkpoint=metric.stem,
                    gamma=data["gamma"], seed=data["seed"], step=data["step"], layer=layer["layer"],
                    status=summary["status"], training_loss=matched["training_loss"],
                    training_accuracy=matched["training_accuracy"], test_accuracy=data["test_accuracy"],
                    ntk_drift=data["ntk_drift"], weight_displacement=summary["weight_displacement"] if metric.stem == "final" else None,
                    cka_drift=layer["cka_drift"], effective_rank=layer["effective_rank"],
                    local_q01=layer["jacobian"]["q01"], local_normalized_q01=layer["jacobian"]["normalized_q01"],
                    task_jacobian=layer["jacobian"]["task_norm"], nuisance_jacobian=layer["jacobian"]["nuisance_norm"],
                    global_q01=layer["global_margin"]["q01"], global_normalized_q01=layer["global_margin"]["normalized_q01"],
                    collision_score=layer["collisions"]["mean"],
                    linear_nuisance_cosine=layer["probes"]["linear"].get("angular_cosine"),
                    mlp_nuisance_cosine=layer["probes"]["mlp"].get("angular_cosine"),
                    linear_task_accuracy=layer["probes"]["linear"]["task_accuracy"],
                    mlp_task_accuracy=layer["probes"]["mlp"]["task_accuracy"],
                    ph_h1_top1=np.mean([r["H1"]["top1"] for r in layer["persistence"]]),
                    ph_h1_top2=np.mean([r["H1"]["top2"] for r in layer["persistence"]])))
    return pd.DataFrame(rows)


def summarize(root):
    root = Path(root); frame = rows_from(root)
    if frame.empty:
        raise RuntimeError("No computed metrics yet")
    frame.to_csv(root/"metrics_long.csv", index=False)
    final = frame[frame.checkpoint == "final"]
    metrics = ["test_accuracy", "ntk_drift", "cka_drift", "effective_rank", "local_normalized_q01",
               "global_normalized_q01", "nuisance_jacobian", "mlp_nuisance_cosine", "ph_h1_top2"]
    aggregates, transitions = [], []
    for (profile, layer), part in final.groupby(["profile", "layer"]):
        for metric in metrics:
            for gamma, group in part.groupby("gamma"):
                aggregates.append(dict(profile=profile, layer=int(layer), gamma=float(gamma), metric=metric,
                                       **bootstrap_mean(group[metric])))
            matrix = part.pivot(index="seed", columns="gamma", values=metric).dropna()
            if len(matrix):
                fit = segmented_fit(matrix.columns, matrix.mean(0))
                transitions.append(dict(profile=profile, layer=int(layer), metric=metric, **fit,
                                        bootstrap=bootstrap_transition(matrix.columns, matrix.to_numpy(), repeats=500)))
    pd.DataFrame(aggregates).to_csv(root/"seed_bootstrap_ci.csv", index=False)
    atomic_json(root/"transitions.json", transitions)
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    a = p.parse_args(); summarize(a.root)
