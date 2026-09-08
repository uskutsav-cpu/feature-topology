"""Generate auditable result tables; keep measured results and missingness apart."""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import math
import numpy as np
import pandas as pd
from .catalog import load_catalog, matched_risk, coverage
from .io import atomic_json, atomic_text, digest, environment, file_digest
from .statistics import bootstrap_mean, paired_contrast, model_comparison, crossing_interval, event_order

METRICS = ["test_accuracy", "ntk_drift", "cka_drift", "effective_rank", "local_q01",
           "local_normalized_q01", "global_normalized_q01", "nuisance_jacobian",
           "mlp_nuisance_cosine", "ph_h1_top2", "ph_h2_top1"]


def analyze(roots, output, *, target=.1, profile=None, gammas=None, seeds=None,
            repeats=2000, transition_repeats=300, rules=None, make_plots=False) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gammas = list(gammas if gammas is not None else [2.**i for i in range(-5, 8)])
    seeds = list(seeds if seeds is not None else range(10))
    catalog = load_catalog(roots, profile=profile)
    atomic_json(output/"audit.json", coverage(catalog, gammas, seeds))
    catalog.require_clean()
    frame = catalog.frame()
    if frame.empty:
        raise ValueError("No valid metric rows. audit.json records the missing evidence.")
    atomic_text(output/"metrics_long.csv", frame.to_csv(index=False))
    matched, exclusions = matched_risk(frame, target)
    atomic_text(output/"matched_risk.csv", matched.to_csv(index=False))
    atomic_json(output/"exclusions.json", exclusions)
    aggregates, contrasts, transitions, missing = [], [], [], []
    if not matched.empty:
        for (condition, pid, layer), part in matched.groupby(["condition", "profile", "layer"]):
            shared = dict(condition=condition, profile=pid, layer=int(layer))
            for metric in METRICS:
                if metric not in part:
                    continue
                for gamma, group in part.groupby("gamma"):
                    complete = group[group[metric].notna()]
                    aggregates.append({**shared, "gamma": float(gamma), "metric": metric,
                                       "missing_seeds": len(group)-len(complete),
                                       **bootstrap_mean(complete[metric].tolist(), repeats=repeats)})
                matrix = part.pivot(index="seed", columns="gamma", values=metric).reindex(columns=gammas)
                valid = matrix.dropna()
                missing.append({**shared, "metric": metric, "complete_case_seeds": [int(s) for s in valid.index],
                                "excluded_seeds": [int(s) for s in matrix.index.difference(valid.index)]})
                fit = model_comparison(gammas, valid.to_numpy(), repeats=transition_repeats)
                transitions.append({**shared, "metric": metric, **fit})
                for left, right in zip(gammas, gammas[1:]):
                    a, b = matrix[left].dropna().to_dict(), matrix[right].dropna().to_dict()
                    contrasts.append({**shared, "metric": metric, "gamma_left": left,
                                      "gamma_right": right, **paired_contrast(a, b, repeats=repeats)})
    agg = pd.DataFrame(aggregates)
    atomic_text(output/"seed_bootstrap_ci.csv", agg.to_csv(index=False))
    atomic_json(output/"paired_contrasts.json", contrasts)
    atomic_json(output/"model_comparisons.json", transitions)
    atomic_json(output/"complete_case_accounting.json", missing)
    events, orders = [], []
    rules = list(rules or [])
    names = [r["name"] for r in rules]
    if len(set(names)) != len(names):
        raise ValueError("Event-rule names must be unique")
    for keys, part in frame.groupby(["condition", "profile", "run_id", "layer"]):
        part = part.sort_values("step")
        shared = dict(condition=keys[0], profile=keys[1], run_id=keys[2], layer=int(keys[3]),
                      gamma=float(part.gamma.iloc[0]), seed=int(part.seed.iloc[0]))
        current = {}
        for rule in rules:
            metric = rule["metric"]
            if metric not in part:
                raise ValueError(f"Unknown rule metric: {metric}")
            result = crossing_interval(part.step.tolist(), part[metric].tolist(),
                                       threshold=rule["threshold"], direction=rule["direction"],
                                       consecutive=rule.get("consecutive", 1))
            events.append({**shared, "rule": rule["name"], "metric": metric, **result})
            current[rule["name"]] = result
        for a, b in combinations(current, 2):
            orders.append({**shared, "event_a": a, "event_b": b, "order": event_order(current[a], current[b])})
    atomic_json(output/"event_intervals.json", events)
    atomic_json(output/"event_order.json", orders)
    settings = {"target_loss": target, "profile_filter": profile, "gammas": gammas, "seeds": seeds,
                "bootstrap_repeats": repeats, "transition_repeats": transition_repeats, "rules": rules}
    atomic_json(output/"analysis_settings.json", settings)
    atomic_json(output/"environment.json", environment())
    if make_plots and not agg.empty:
        plot_aggregates(agg, output/"figures")
    result = {"status": "analysis_completed_not_study_completion", "runs_read": len(catalog.runs),
              "unique_metric_rows": len(frame), "matched_risk_rows": len(matched),
              "excluded_run_layer_profiles": len(exclusions), "event_rules": len(rules),
              "analysis_id": digest(settings), "formal_proof_status": "not_asserted"}
    text = f"""# Feature-topology analysis snapshot

Status: **analysis completed, not a declaration that the research is finished**.

- Runs inspected: {len(catalog.runs)} (aliases are removed from statistical rows).
- Unique run/profile/checkpoint/layer measurements: {len(frame)}.
- Selected common-threshold measurements: {len(matched)}.
- Excluded run/profile/layer comparisons: {len(exclusions)}.
- Training-loss threshold: {target}; actual losses are retained, not forced equal.

## Interpretation boundaries

Bootstrap units are independent training seeds. Gamma contrasts resample matched
seed pairs. Model fits are exploratory descriptive comparisons; a hinge winner
is not proof of a phase transition. Missing gamma levels are not interpolated.
Event rules are {'provided and recorded' if rules else 'absent; no threshold events were invented'}.
Observed checkpoint ordering is not causal ordering and can miss between-checkpoint
events. High probe performance supports decodability; low probe performance does
not prove information-theoretic destruction. Rational small-network certificates
have a separate scope and are not substituted for empirical measurements.

See `audit.json`, `exclusions.json`, and `complete_case_accounting.json` before
interpreting `seed_bootstrap_ci.csv` or `model_comparisons.json`.
"""
    atomic_text(output/"REPORT.md", text)
    atomic_json(output/"analysis_result.json", result)
    inputs = {}
    for run in catalog.runs:
        directory = Path(run["directory"])
        paths = [directory/"config.json", directory/"summary.json", *directory.glob("metrics/*/*.json")]
        for path in paths:
            if path.is_file():
                inputs[str(path)] = file_digest(path)
    atomic_json(output/"input_hashes.json", inputs)
    return result


def plot_aggregates(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    for keys, group in frame.groupby(["condition", "profile", "layer", "metric"]):
        group = group.sort_values("gamma")
        valid = group[group["mean"].notna()]
        if valid.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(valid.gamma, valid["mean"], marker="o")
        band = valid[valid.lower.notna() & valid.upper.notna()]
        if not band.empty:
            ax.fill_between(band.gamma.to_numpy(float), band.lower.to_numpy(float), band.upper.to_numpy(float), alpha=.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Output-scale parameter gamma")
        ax.set_ylabel(keys[3])
        ax.set_title(f"Layer {keys[2]} | seed-level 95% intervals")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output/f"{keys[0]}_{keys[1][:8]}_L{keys[2]}_{keys[3]}.png", dpi=180)
        plt.close(fig)
