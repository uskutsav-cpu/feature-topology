"""Prospective finite-width scaling gate for transition terminology."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .catalog import ABLATION_PRODUCTION, PRODUCTION, load_catalog, matched_risk
from .io import atomic_json, file_digest, read_json
from src.training.checkpoints import fingerprint


def _aicc(residuals: np.ndarray, parameter_count: int) -> float | None:
    n = len(residuals)
    # parameter_count includes the residual variance.
    if n <= parameter_count + 1:
        return None
    rss = float(np.square(residuals).sum())
    floor = max(float(np.mean(np.square(residuals))), 1.) * (100*n*np.finfo(float).eps)**2
    return float(n*math.log(max(rss/n, floor)) + 2*parameter_count
                 + 2*parameter_count*(parameter_count+1)/(n-parameter_count-1))


def _logistic(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    lower, amplitude, center, log_scale = parameters
    scale = np.exp(log_scale)
    z = np.clip((x-center)/scale, -50., 50.)
    return lower + amplitude/(1.+np.exp(-z))


def fit_width_curve(gammas, values, expected_direction="decreasing") -> dict:
    """Compare a quadratic smooth null with a monotone logistic alternative."""
    gamma, y = np.asarray(gammas, float), np.asarray(values, float)
    if (gamma.ndim != 1 or y.shape != gamma.shape or len(gamma) < 7
            or np.any(gamma <= 0) or not np.isfinite(gamma).all()
            or not np.isfinite(y).all() or len(np.unique(gamma)) != len(gamma)):
        return {"status": "insufficient_data"}
    if expected_direction not in {"increasing", "decreasing"}:
        raise ValueError("expected_direction must be increasing or decreasing")
    order = np.argsort(gamma)
    x, y = np.log2(gamma[order]), y[order]
    design = np.column_stack((np.ones_like(x), x, x*x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 3:
        return {"status": "rank_deficient"}
    smooth_residuals = y-design@coefficients
    span = max(float(np.ptp(y)), np.finfo(float).eps)
    sign = 1. if expected_direction == "increasing" else -1.
    bounds = (
        [float(y.min()-span), 0. if sign > 0 else -3*span, float(x[1]), math.log(.05)],
        [float(y.max()+span), 3*span if sign > 0 else 0., float(x[-2]), math.log(20.)],
    )
    starts = []
    for center in x[1:-1]:
        starts.append(np.array([y[0], sign*span, center, math.log(1.)]))
    choices = []
    for start in starts:
        fit = least_squares(lambda p: _logistic(p, x)-y, start, bounds=bounds)
        choices.append(fit)
    best = min(choices, key=lambda result: float(np.square(result.fun).sum()))
    lower, amplitude, center, log_scale = best.x
    scale = math.exp(float(log_scale))
    smooth_aicc = _aicc(smooth_residuals, 4)
    transition_aicc = _aicc(best.fun, 5)
    delta = None if smooth_aicc is None or transition_aicc is None else smooth_aicc-transition_aicc
    return {
        "status": "ok",
        "n_gamma_levels": len(x),
        "smooth": {"coefficients": coefficients.tolist(), "aicc": smooth_aicc,
                   "rss": float(np.square(smooth_residuals).sum())},
        "transition": {
            "lower": float(lower), "amplitude": float(amplitude),
            "center_log2_gamma": float(center), "center_gamma": float(2**center),
            "scale_log2_gamma": scale,
            "width_10_90_log2_gamma": float(2*math.log(9)*scale),
            "aicc": transition_aicc, "rss": float(np.square(best.fun).sum()),
        },
        "delta_aicc_smooth_minus_transition": delta,
        "center_is_interior": bool(x[1] < center < x[-2]),
    }


def evaluate_gate(frame: pd.DataFrame, specification: dict) -> dict:
    """Evaluate all frozen clauses on a matched-risk, last-layer data frame."""
    metric = specification["primary_metric"]
    widths = [int(v) for v in specification["widths"]]
    gammas = [float(v) for v in specification["gammas"]]
    seeds = [int(v) for v in specification["seeds"]]
    repeats = int(specification["bootstrap_repeats"])
    expected = {(w, g, s) for w in widths for g in gammas for s in seeds}
    observed_all = {(int(r.width), float(r.gamma), int(r.seed)) for r in frame.itertuples()
                    if pd.notna(getattr(r, metric))}
    observed = observed_all & expected
    if observed != expected:
        return {
            "status": "incomplete",
            "expected_cells": len(expected), "observed_cells": len(observed & expected),
            "missing_cells": [list(v) for v in sorted(expected-observed)],
            "ignored_out_of_design_cells": [list(v) for v in sorted(observed_all-expected)],
            "phase_transition_language_allowed": False, "terminology": "crossover",
        }
    frame = frame[frame.apply(
        lambda row: (int(row.width), float(row.gamma), int(row.seed)) in expected, axis=1)]
    matrices, fits = {}, []
    for width in widths:
        part = frame[frame.width == width]
        matrix = part.pivot(index="seed", columns="gamma", values=metric).reindex(index=seeds, columns=gammas)
        matrices[width] = matrix.to_numpy(float)
        fit = fit_width_curve(gammas, matrices[width].mean(0), specification["expected_direction"])
        fits.append({"width": width, **fit})
    transitions = [row["transition"] for row in fits]
    fitted_widths = np.array([row["width_10_90_log2_gamma"] for row in transitions])
    centers = np.array([row["center_log2_gamma"] for row in transitions])
    rho = float(pd.Series(widths).corr(pd.Series(fitted_widths), method="spearman"))
    ratio = float(fitted_widths[-1]/fitted_widths[0])
    drift = float(abs(centers[-1]-centers[-2]))
    rng = np.random.default_rng(20260919)
    shrink_hits = stable_hits = 0
    successful = 0
    for _ in range(repeats):
        boot_widths, boot_centers = [], []
        # Width is a paired experimental axis: resample the same seed identities
        # jointly at every width rather than manufacturing independent replicates.
        indices = rng.integers(len(seeds), size=len(seeds))
        for width in widths:
            matrix = matrices[width]
            draw = matrix[indices].mean(0)
            fit = fit_width_curve(gammas, draw, specification["expected_direction"])
            if fit.get("status") != "ok":
                break
            boot_widths.append(fit["transition"]["width_10_90_log2_gamma"])
            boot_centers.append(fit["transition"]["center_log2_gamma"])
        if len(boot_widths) != len(widths):
            continue
        successful += 1
        boot_rho = pd.Series(widths).corr(pd.Series(boot_widths), method="spearman")
        if (boot_rho <= specification["gate"]["maximum_width_spearman_rho"]
                and boot_widths[-1]/boot_widths[0]
                <= specification["gate"]["maximum_largest_to_smallest_transition_width_ratio"]):
            shrink_hits += 1
        if abs(boot_centers[-1]-boot_centers[-2]) <= specification["gate"]["maximum_largest_width_center_drift_log2"]:
            stable_hits += 1
    shrink_support = shrink_hits/successful if successful else 0.
    stable_support = stable_hits/successful if successful else 0.
    clauses = {
        "transition_preferred_each_width": all(
            row["delta_aicc_smooth_minus_transition"] >= specification["gate"]["minimum_delta_aicc_each_width"]
            for row in fits),
        "widths_shrink_systematically": rho <= specification["gate"]["maximum_width_spearman_rho"],
        "largest_to_smallest_width_ratio": ratio <= specification["gate"]["maximum_largest_to_smallest_transition_width_ratio"],
        "bootstrap_shrink_support": shrink_support >= specification["gate"]["minimum_bootstrap_shrink_support"],
        "largest_width_center_stable": drift <= specification["gate"]["maximum_largest_width_center_drift_log2"],
        "bootstrap_stable_center_support": stable_support >= specification["gate"]["minimum_bootstrap_stable_center_support"],
        "all_centers_interior": all(row["center_is_interior"] for row in fits),
    }
    allowed = all(clauses.values())
    return {
        "status": "complete", "expected_cells": len(expected), "observed_cells": len(observed),
        "ignored_out_of_design_cells": [list(v) for v in sorted(observed_all-expected)],
        "metric": metric, "fits": fits,
        "scaling": {"transition_width_spearman_rho": rho,
                    "largest_to_smallest_transition_width_ratio": ratio,
                    "largest_width_center_drift_log2": drift,
                    "bootstrap_successful": successful,
                    "bootstrap_shrink_support": shrink_support,
                    "bootstrap_stable_center_support": stable_support},
        "clauses": clauses, "phase_transition_language_allowed": allowed,
        "terminology": "phase transition" if allowed else "crossover",
        "scope": specification["scope"],
    }


def analyze_width_scaling(roots, output, specification_path, profile=None) -> dict:
    repository = Path(__file__).resolve().parents[1]
    expanded_roots = []
    for value in roots:
        root = Path(value)
        if not root.exists():
            continue
        if (root/"runs").is_dir() or root.name == "runs":
            expanded_roots.append(root)
        else:
            expanded_roots.extend(sorted(root.glob("*/runs")))
    if not expanded_roots:
        raise FileNotFoundError("No supplied width-scaling result root exists")
    primary_profile = fingerprint(PRODUCTION)
    ablation_profile = fingerprint(ABLATION_PRODUCTION)
    specification_path = Path(specification_path)
    specification = read_json(specification_path)
    catalog = load_catalog(expanded_roots, profile=profile)
    catalog.require_clean()
    frame, exclusions = matched_risk(catalog.frame(), specification["target_training_loss"])
    if not frame.empty:
        if profile is None:
            frame = frame[((frame.width == 256) & (frame.profile == primary_profile))
                          | ((frame.width != 256) & (frame.profile == ablation_profile))]
        frame = frame[(frame.manifold == "torus") & (~frame.swap)
                      & (frame.relevance == 0.) & (frame.nuisance_condition == "iid")
                      & (frame.layer == frame.depth-1)]
    result = evaluate_gate(frame, specification)
    result.update({
        "schema": "feature-topology.width-scaling-result.v1",
        "specification": str(specification_path.resolve().relative_to(repository)),
        "specification_sha256": file_digest(specification_path),
        "profiles": ({"explicit": profile} if profile else
                     {"primary_width_256": primary_profile, "width_ablation": ablation_profile}),
        "input_hashes": {
            str(path.resolve().relative_to(repository)): file_digest(path)
            for root in expanded_roots for path in Path(root).rglob("*.json")
        },
    })
    atomic_json(output, result)
    return result
