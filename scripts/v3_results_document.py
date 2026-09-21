"""Create the concise v3 scientific result record from derived frozen outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_ext.io import atomic_json, atomic_text
from research_ext.report import METRICS
from research_ext.statistics import bootstrap_mean, paired_contrast


PRIMARY_LABELS = {
    "cka_drift": "geometry deformation (1 - CKA)",
    "fiber_local_normalized_minimum": "local fiber margin L",
    "fiber_global_normalized_minimum": "global fiber margin M",
    "fiber_collision_pairs": "sampled collision-pair count",
    "mlp_nuisance_cosine": "nonlinear nuisance decodability D_F",
    "test_accuracy": "test accuracy R",
    "ph_h1_top2": "supporting H1 persistence",
    "ph_h2_top1": "supporting H2 persistence",
}


def _finite(value):
    return value is not None and np.isfinite(value)


def _interval(row, prefix=""):
    mean = row.get(prefix + "mean")
    lower = row.get(prefix + "lower")
    upper = row.get(prefix + "upper")
    if not all(_finite(value) for value in (mean, lower, upper)):
        return "not estimable"
    return f"{mean:.4g} [{lower:.4g}, {upper:.4g}]"


def endpoint_effects(matched_path: Path, repeats: int = 2000) -> pd.DataFrame:
    """Report high-minus-low gamma for every metric, paired by training seed."""
    frame = pd.read_csv(matched_path)
    required = {"seed", "gamma", "layer", *METRICS}
    if not required <= set(frame):
        raise ValueError(f"Primary matched-risk columns missing: {sorted(required-set(frame))}")
    last_layer = int(frame.layer.max())
    frame = frame[frame.layer == last_layer]
    low, high = float(frame.gamma.min()), float(frame.gamma.max())
    rows = []
    for metric in METRICS:
        low_values = frame[frame.gamma == low].set_index("seed")[metric].dropna().to_dict()
        high_values = frame[frame.gamma == high].set_index("seed")[metric].dropna().to_dict()
        low_summary = bootstrap_mean(list(low_values.values()), repeats=repeats)
        high_summary = bootstrap_mean(list(high_values.values()), repeats=repeats)
        contrast = paired_contrast(low_values, high_values, repeats=repeats)
        rows.append({
            "layer": last_layer, "metric": metric,
            "gamma_low": low, "gamma_high": high,
            **{f"low_{key}": value for key, value in low_summary.items()},
            **{f"high_{key}": value for key, value in high_summary.items()},
            **{f"delta_{key}": value for key, value in contrast.items()},
        })
    return pd.DataFrame(rows)


def threshold_audit(primary_ci: pd.DataFrame, exact_certificates: pd.DataFrame,
                    deformation_threshold: float, margin_threshold: float) -> dict:
    """Apply only the thresholds frozen before v3 production metrics."""
    last_layer = int(primary_ci.layer.max())
    current = primary_ci[primary_ci.layer == last_layer]
    geometry = current[(current.metric == "cka_drift")
                       & (current.lower > deformation_threshold)]
    separated = current[(current.metric == "fiber_global_normalized_minimum")
                        & (current.lower > margin_threshold)]
    both = geometry[["gamma"]].merge(separated[["gamma"]], on="gamma")
    margins = current[current.metric.isin([
        "fiber_local_normalized_minimum", "fiber_global_normalized_minimum"])]
    near = margins[margins.upper < margin_threshold][["gamma", "metric"]]
    collisions = exact_certificates[exact_certificates.exact_collision_present.astype(bool)]
    return {
        "schema": "feature-topology.v3-claim-audit.v1",
        "threshold_source": "configs/analysis_plan_v2.json",
        "primary_layer": last_layer,
        "deformation_threshold": deformation_threshold,
        "near_singular_normalized_margin": margin_threshold,
        "deformation_with_positive_sampled_global_margin_gammas":
            sorted(float(value) for value in both.gamma.unique()),
        "near_singular_cells": [
            {"gamma": float(row.gamma), "metric": row.metric}
            for row in near.itertuples(index=False)
        ],
        "rationally_certified_trained_polygon_collisions": int(len(collisions)),
        "exact_continuum_task_quotient_established": False,
        "qualification": (
            "Threshold classifications are empirical diagnostics with pointwise seed "
            "intervals. Positive sampled margins do not prove injectivity, and a restricted "
            "polygon collision would not by itself prove equality with task equivalence."
        ),
    }


def render(repo: Path, output: Path, frozen_manifest_sha256: str, reports: list[dict],
           relevance_statistics: dict, ood_statistics: dict,
           certification: dict, width_scaling: dict) -> dict:
    repo = Path(repo).resolve()
    output = Path(output)
    primary = output / "main/main"
    effects = endpoint_effects(primary / "matched_risk.csv")
    effects_path = output / "headline_endpoint_effects.csv"
    atomic_text(effects_path, effects.to_csv(index=False))
    primary_ci = pd.read_csv(primary / "seed_bootstrap_ci.csv")
    exact = pd.read_csv(output / "certification/trained_circle_exact_certificates.csv")
    plan = json.loads((repo / "configs/analysis_plan_v2.json").read_text())
    thresholds = plan["descriptive_thresholds"]
    audit = threshold_audit(primary_ci, exact,
                            thresholds["geometry_deformed"],
                            thresholds["near_singular_normalized_margin"])
    atomic_json(output / "claim_audit.json", audit)

    effect_rows = effects[effects.metric.isin(PRIMARY_LABELS)].copy()
    effect_rows["order"] = effect_rows.metric.map(
        {name: index for index, name in enumerate(PRIMARY_LABELS)})
    effect_rows = effect_rows.sort_values("order")
    effect_table = [
        "| Frozen metric | low γ mean [95% CI] | high γ mean [95% CI] | "
        "paired high-minus-low [95% CI] | paired seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in effect_rows.to_dict("records"):
        effect_table.append(
            f"| {PRIMARY_LABELS[row['metric']]} | {_interval(row, 'low_')} | "
            f"{_interval(row, 'high_')} | {_interval(row, 'delta_')} | "
            f"{int(row['delta_n_seeds'])} |"
        )

    supported = []
    if audit["deformation_with_positive_sampled_global_margin_gammas"]:
        values = ", ".join(f"{value:g}" for value in
                           audit["deformation_with_positive_sampled_global_margin_gammas"])
        supported.append(
            "At γ=" + values + ", the last-layer pointwise seed interval is above the "
            "frozen deformation threshold while the sampled global fiber-margin interval "
            "remains above the frozen near-singularity threshold. This supports empirical "
            "deformation without sampled degeneration, not a proof of injectivity."
        )
    if audit["near_singular_cells"]:
        supported.append(
            f"The frozen near-singularity criterion is met in {len(audit['near_singular_cells'])} "
            "last-layer gamma/metric cells; `claim_audit.json` lists them without relabeling "
            "them as quotient formation."
        )
    else:
        supported.append(
            "No last-layer gamma/metric cell has its entire pointwise seed interval below "
            "the frozen near-singularity threshold."
        )
    supported.append(
        f"The evidence ladder replays {certification['analytic_exact_controls']} analytic "
        f"controls and {certification['rational_trained_layer_certificates']} rational trained-"
        "layer certificates on their declared finite domains."
    )

    unsupported = [
        ("Phase-transition terminology is not supported by the prespecified finite-width gate."
         if not width_scaling["phase_transition_language_allowed"] else
         "The finite-width gate permits transition terminology only for its stated empirical order parameter; it is not a thermodynamic-limit proof."),
        "Exact continuum task-quotient formation is not established. PH loss, small margins, "
        "probe failure, and sampled collisions are not substituted for an equivalence-relation proof.",
        "Geometry alone does not establish information destruction, and local Jacobian rank "
        "does not establish global injectivity.",
        "No additional monotonicity or architecture-universality decision rule is introduced "
        "after observing v3; full curves and paired contrasts remain the evidence.",
    ]
    text = """# Frozen v3 scientific results

This document is generated deterministically from the immutable v3 result manifest and
the final analysis outputs. The statistical replicate is the training seed. Intervals are
pointwise 95% percentile seed bootstraps with 2,000 resamples unless an artifact states
otherwise.

## Conclusions supported at the stated evidence level

""" + "\n".join(f"- {item}" for item in supported) + """

## Unsupported or deliberately unresolved claims

""" + "\n".join(f"- {item}" for item in unsupported) + f"""

## Finite-width decision

The frozen gate returns **{width_scaling['terminology']}**. Phase-transition language is
`{str(width_scaling['phase_transition_language_allowed']).lower()}`. This decision is
mechanical and was not changed in response to the result. The complete clause values and
scaling estimates are recorded in `analysis_manifest.json` and visualized in
[`width_scaling/finite_width_gate.png`](width_scaling/finite_width_gate.png).

## Primary endpoint effect sizes

These are descriptive endpoint contrasts across the complete frozen gamma range at the
last hidden layer. Every frozen metric is written to `headline_endpoint_effects.csv`; the
predeclared geometry, margin, collision, decoding, predictive, and supporting-PH measures
are shown here.

""" + "\n".join(effect_table) + f"""

## Relevance and nuisance interventions

All {relevance_statistics['relevance_levels']} relevance levels are reported with training-
seed intervals and fixed-gamma paired contrasts in `relevance_dependence/`. The OOD analysis
contains {ood_statistics['runs']} checkpoint-hash-bound runs and reports intervention-minus-
IID effects in `ood_evaluation/`. These predictive contrasts do not establish representation
causality. Major figures are
[`relevance_regime_map.png`](relevance_dependence/relevance_regime_map.png) and
[`ood_intervention_shifts.png`](ood_evaluation/ood_intervention_shifts.png).

## Exact, certified, and empirical scopes

- **Analytic/exact controls:** exact only on their explicitly defined domains.
- **Trained rational certificates:** {certification['exact_scope']}
- **Numerical polygon diagnostics:** {certification['numerical_scope']}
- **Formal audit:** {certification['formal_scope']}
- **Neural-network sweeps:** sampled empirical evidence with seed uncertainty; never a proof
  of continuum injectivity or quotienting.

## Limitations

- The v3 analysis plan was prospective for production metrics but exploratory relative to
  earlier pilot and recovered H1-only artifacts.
- Persistent-homology subsamples are supporting computations, not statistical replicates.
- CIFAR calibration did not reach the target loss in the hosted budget; its frozen rule chose
  the lowest stable optimization loss. Image comparisons therefore retain actual losses and
  are final stopping-threshold comparisons, not identical-risk matches.
- Image schemas are kept separate; CIFAR makes no latent-manifold topology claim.
- Pointwise intervals are not simultaneous multiple-comparison guarantees.

## Provenance

- Frozen input-manifest SHA-256: `{frozen_manifest_sha256}`
- Final condition-level reports: {len(reports)}
- Primary claim classification: `claim_audit.json`
- Complete derived-file hashes: `analysis_manifest.json`
"""
    atomic_text(output / "V3_RESULTS.md", text)
    return audit
