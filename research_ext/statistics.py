"""Seed-level inference and explicitly descriptive transition diagnostics.

PH subsamples and checkpoints are NOT independent training replicates.
Change-point fits are exploratory; no automatic 'phase transition' claim is made.
"""
from __future__ import annotations
from typing import Sequence
import math
import numpy as np


def bootstrap_mean(values: Sequence[float], *, repeats: int = 2000,
                   confidence: float = .95, seed: int = 1729) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Supply a finite, one-dimensional vector, one value per seed")
    if repeats < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    result = {"n_seeds": len(x), "mean": float(x.mean()) if len(x) else None,
              "lower": None, "upper": None, "method": "seed percentile bootstrap",
              "confidence": confidence}
    if len(x) < 2:
        result["status"] = "insufficient_seeds"
        return result
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for start in range(0, repeats, 256):
        n = min(256, repeats-start)
        draws[start:start+n] = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    lo, hi = np.quantile(draws, [(1-confidence)/2, (1+confidence)/2])
    result.update(lower=float(lo), upper=float(hi), status="ok")
    return result


def paired_contrast(left: dict[int, float], right: dict[int, float], **kwargs) -> dict:
    """Return right-minus-left, paired by training seed; expose incomplete pairs."""
    common = sorted(left.keys() & right.keys())
    result = bootstrap_mean([right[k]-left[k] for k in common], **kwargs)
    result.update(paired_seeds=common, left_only=sorted(left.keys()-right.keys()),
                  right_only=sorted(right.keys()-left.keys()), contrast="right minus left")
    return result


def _design(x: np.ndarray, model: str, knot: float | None = None) -> np.ndarray:
    if model == "constant":
        return np.ones((len(x), 1))
    if model == "linear":
        return np.column_stack([np.ones_like(x), x])
    if model == "quadratic":
        return np.column_stack([np.ones_like(x), x, x*x])
    if model == "hinge" and knot is not None:
        return np.column_stack([np.ones_like(x), x, np.maximum(0, x-knot)])
    raise ValueError("Unknown model")


def _fit(x: np.ndarray, y: np.ndarray, model: str) -> dict:
    knots = [None] if model != "hinge" else list(x[2:-2])
    if not knots:
        return {"model": model, "aicc": None, "status": "insufficient_levels"}
    choices = []
    for knot in knots:
        X = _design(x, model, knot)
        coeff, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < X.shape[1]:
            continue
        rss = float(np.sum((y-X@coeff)**2))
        # Include residual variance, and the searched hinge location.
        k = X.shape[1] + 1 + (model == "hinge")
        n = len(x)
        scale_floor = max(float(np.mean(y*y)), 1.) * (100*n*np.finfo(float).eps)**2
        aicc = None if n <= k+1 else (n*math.log(max(rss/n, scale_floor)) +
                                            2*k + 2*k*(k+1)/(n-k-1))
        choices.append({"model": model, "rss": rss, "aicc": aicc,
                        "coefficients": coeff.tolist(), "n_levels": n,
                        "knot_log2_gamma": None if knot is None else float(knot),
                        "k_parameters_including_variance": int(k), "mse_roundoff_floor": scale_floor, "status": "ok"})
    if not choices:
        return {"model": model, "aicc": None, "status": "rank_deficient"}
    return min(choices, key=lambda c: c["rss"])


def model_comparison(gammas: Sequence[float], seed_by_gamma: np.ndarray,
                     *, repeats: int = 300, seed: int = 2718) -> dict:
    """Paired bootstrap of entire seed trajectories, not individual gamma cells.

    Complete cases must be supplied. Caller reports excluded seeds. Hinge location
    intervals are conditional on the hinge winning AICc; support is reported too.
    AICc treats gamma-mean residuals heuristically, not as a confirmatory test.
    """
    g = np.asarray(gammas, dtype=float)
    matrix = np.asarray(seed_by_gamma, dtype=float)
    if (g.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != len(g)
            or np.any(g <= 0) or not np.isfinite(g).all()
            or not np.isfinite(matrix).all() or len(set(g)) != len(g)):
        raise ValueError("Invalid paired seed × gamma matrix")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    order = np.argsort(g)
    g, matrix = g[order], matrix[:, order]
    if len(g) < 7 or len(matrix) < 2:
        return {"status": "insufficient_data", "minimum_gamma_levels": 7,
                "minimum_seeds": 2, "n_levels": len(g), "n_seeds": len(matrix)}
    x, y = np.log2(g), matrix.mean(0)
    models = ["constant", "linear", "quadratic", "hinge"]
    fits = [_fit(x, y, name) for name in models]
    for fit in fits:
        errors = []
        for i in range(len(g)):
            use = np.arange(len(g)) != i
            held = _fit(x[use], y[use], fit["model"])
            if held.get("coefficients") is None:
                continue
            pred = _design(x[i:i+1], held["model"], held["knot_log2_gamma"]) @ held["coefficients"]
            errors.append(float((pred[0]-y[i])**2))
        fit["leave_one_gamma_out_mse"] = float(np.mean(errors)) if len(errors) == len(g) else None
    valid = [f for f in fits if f.get("aicc") is not None]
    winner = min(valid, key=lambda f: f["aicc"])["model"]
    rng, knots, wins = np.random.default_rng(seed), [], {name: 0 for name in models}
    for _ in range(repeats):
        draw = matrix[rng.integers(len(matrix), size=len(matrix))].mean(0)
        candidates = [_fit(x, draw, name) for name in models]
        selected = min([f for f in candidates if f.get("aicc") is not None], key=lambda f: f["aicc"])
        wins[selected["model"]] += 1
        if selected["model"] == "hinge":
            knots.append(selected["knot_log2_gamma"])
    interval = None
    if knots:
        interval = (2**np.quantile(knots, [.025, .975])).tolist()
    return {"status": "exploratory", "fits": fits, "aicc_winner": winner,
            "bootstrap_winner_fractions": {k: v/repeats for k, v in wins.items()},
            "conditional_hinge_gamma_interval": interval, "n_seeds": len(matrix),
            "n_levels": len(g), "bootstrap_repeats": repeats,
            "warning": "Descriptive model comparison, not proof of a transition or causal ordering."}


def crossing_interval(steps: Sequence[int], values: Sequence[float | None],
                      *, threshold: float, direction: str, consecutive: int = 1) -> dict:
    """Bracket the FIRST OBSERVED crossing, not the first unobserved crossing.

    With consecutive > 1, onset means the first confirmed run of observed hits;
    earlier isolated hits remain possible. Missing metrics break confirmation.
    The lower bracket is always an actual non-hit, never an earlier hit. An unobserved
    crossing between evaluation times can never be ruled out by this routine.
    """
    if direction not in {"above", "below"} or consecutive < 1 or not math.isfinite(threshold):
        raise ValueError("Invalid crossing rule")
    if len(steps) != len(values) or not steps or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Steps must be nonempty and strictly increasing")
    last_nonhit = None
    any_hit = False
    start = None
    lower = None
    count = 0
    observed = 0
    for i, (step, value) in enumerate(zip(steps, values)):
        if value is None or not math.isfinite(float(value)):
            count, start = 0, None
            continue
        observed += 1
        hit = value >= threshold if direction == "above" else value <= threshold
        if hit:
            any_hit = True
            if count == 0:
                # A missing sample may interrupt hits without demonstrating a
                # return below the threshold. Keep the last actual non-hit.
                start, lower = step, last_nonhit
            count += 1
            if count >= consecutive:
                return {"status": "left_censored" if lower is None else "observed_bracket",
                        "lower_step": lower, "upper_step": start, "confirmed_at_step": step,
                        "threshold": threshold, "direction": direction, "consecutive": consecutive}
        else:
            last_nonhit = step
            count, start = 0, None
    return {"status": ("unconfirmed" if any_hit else "right_censored") if observed else "unobserved",
            "lower_step": last_nonhit, "upper_step": None,
            "threshold": threshold, "direction": direction, "consecutive": consecutive}


def event_order(a: dict, b: dict) -> str:
    """Only order disjoint observed/censoring intervals; otherwise unresolved."""
    if a["status"] in {"unobserved", "unconfirmed"} or b["status"] in {"unobserved", "unconfirmed"}:
        return "unresolved"
    if a.get("upper_step") is not None and b.get("lower_step") is not None:
        if a["upper_step"] < b["lower_step"]:
            return "a_before_b"
    if b.get("upper_step") is not None and a.get("lower_step") is not None:
        if b["upper_step"] < a["lower_step"]:
            return "b_before_a"
    return "unresolved"
