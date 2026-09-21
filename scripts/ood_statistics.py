"""Frozen seed-level summaries for the nuisance-intervention evaluation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research_ext.io import atomic_json, atomic_text
from research_ext.statistics import bootstrap_mean, paired_contrast


ENVIRONMENTS = ("iid", "concentrated", "spurious", "unseen")
METRICS = ("accuracy", "loss")


def nuisance_rows(manifest):
    if manifest.get("schema") != "feature-topology.nuisance-shift-manifest.v1":
        raise ValueError("Unexpected nuisance-shift manifest schema")
    if tuple(manifest.get("evaluation_environments", ())) != ENVIRONMENTS:
        raise ValueError("Frozen nuisance environments changed")
    rows = []
    for run in manifest.get("runs", []):
        if run.get("status") != "evaluated" or set(run.get("environments", {})) != set(ENVIRONMENTS):
            raise ValueError(f"Incomplete nuisance-shift run: {run.get('run_id')}")
        for environment in ENVIRONMENTS:
            values = run["environments"][environment]
            for metric in METRICS:
                value = values.get(metric)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not np.isfinite(value)):
                    raise ValueError(f"Nonfinite nuisance-shift value: {run['run_id']}")
                rows.append({
                    "run_id": run["run_id"],
                    "training_condition": run["training_nuisance_condition"],
                    "gamma": float(run["gamma"]), "seed": int(run["seed"]),
                    "evaluation_environment": environment, "metric": metric,
                    "value": float(value), "selected_step": int(run["selected_step"]),
                    "actual_training_loss": float(run["actual_training_loss"]),
                    "checkpoint_sha256": run["checkpoint_sha256"],
                })
    frame = pd.DataFrame(rows)
    keys = ["training_condition", "gamma", "seed", "evaluation_environment", "metric"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Missing or duplicate nuisance-shift cells")
    return frame


def summarize_nuisance_shift(manifest_path, output, repeats=2000):
    manifest = json.loads(Path(manifest_path).read_text())
    frame = nuisance_rows(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output/"ood_metrics_long.csv", frame.to_csv(index=False))
    summaries = []
    for keys, part in frame.groupby(["training_condition", "gamma", "evaluation_environment", "metric"]):
        if part.seed.duplicated().any():
            raise ValueError("Training seed is not a unique replicate")
        summaries.append(dict(training_condition=keys[0], gamma=float(keys[1]),
                              evaluation_environment=keys[2], metric=keys[3],
                              **bootstrap_mean(part.value.tolist(), repeats=repeats)))
    atomic_text(output/"ood_seed_confidence_intervals.csv", pd.DataFrame(summaries).to_csv(index=False))
    reference = frame[frame.evaluation_environment == "iid"][[
        "run_id", "training_condition", "gamma", "seed", "metric", "value"
    ]].rename(columns={"value": "iid_value"})
    shifted = frame[frame.evaluation_environment != "iid"].merge(
        reference, on=["run_id", "training_condition", "gamma", "seed", "metric"],
        validate="many_to_one")
    shifted["shift_minus_iid"] = shifted.value-shifted.iid_value
    atomic_text(output/"ood_environment_shifts.csv", shifted.to_csv(index=False))
    shift_summaries = []
    for keys, part in shifted.groupby(["training_condition", "gamma", "evaluation_environment", "metric"]):
        shift_summaries.append(dict(training_condition=keys[0], gamma=float(keys[1]),
                                    evaluation_environment=keys[2], metric=keys[3],
                                    contrast="environment minus iid",
                                    **bootstrap_mean(part.shift_minus_iid.tolist(), repeats=repeats)))
    shift_summary = pd.DataFrame(shift_summaries)
    atomic_text(output/"ood_environment_shift_confidence_intervals.csv",
                shift_summary.to_csv(index=False))
    contrasts = []
    for keys, part in frame.groupby(["training_condition", "evaluation_environment", "metric"]):
        gammas = sorted(part.gamma.unique())
        matrix = part.pivot(index="seed", columns="gamma", values="value")
        for left, right in zip(gammas, gammas[1:]):
            contrasts.append(dict(training_condition=keys[0], evaluation_environment=keys[1],
                                  metric=keys[2], gamma_left=float(left), gamma_right=float(right),
                                  **paired_contrast(matrix[left].dropna().to_dict(),
                                                    matrix[right].dropna().to_dict(), repeats=repeats)))
    atomic_json(output/"ood_adjacent_gamma_contrasts.json", contrasts)
    shifted_environments=[environment for environment in ENVIRONMENTS if environment!='iid']
    fig,axes=plt.subplots(len(METRICS),len(shifted_environments),figsize=(12,6),
                          sharex=True,layout='constrained')
    for row,metric in enumerate(METRICS):
        for column,environment in enumerate(shifted_environments):
            ax=axes[row,column]
            part=shift_summary[(shift_summary.metric==metric)&
                               (shift_summary.evaluation_environment==environment)]
            for condition,group in part.groupby('training_condition'):
                group=group.sort_values('gamma')
                ax.plot(group.gamma,group['mean'],marker='o',label=condition)
                band=group[group.lower.notna()&group.upper.notna()]
                if not band.empty:
                    ax.fill_between(band.gamma.to_numpy(float),band.lower.to_numpy(float),
                                    band.upper.to_numpy(float),alpha=.12)
            ax.axhline(0,color='black',linewidth=.8,alpha=.5)
            ax.set_xscale('log',base=2)
            ax.set_title(f'{environment}: {metric} shift')
            ax.set_xlabel('Output scale γ')
            if column==0:
                ax.set_ylabel('Intervention minus IID')
            ax.grid(alpha=.2)
    axes[0,-1].legend(frameon=False,fontsize=8)
    fig.suptitle('Nuisance intervention effects with seed-bootstrap intervals')
    for suffix in ['png','pdf','svg']:
        fig.savefig(output/f'ood_intervention_shifts.{suffix}',dpi=220)
    plt.close(fig)
    result = {
        "schema": "feature-topology.ood-statistics.v1",
        "runs": int(frame.run_id.nunique()), "long_rows": len(frame),
        "shift_rows": len(shifted), "bootstrap_repeats": repeats,
        "replicate_unit": "training seed",
        "scope": "Matched-risk predictive nuisance interventions; association does not establish representation causality.",
    }
    atomic_json(output/"ood_analysis_scope.json", result)
    return result
