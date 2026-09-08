import numpy as np
from scipy.stats import spearmanr


def bootstrap_mean(values, repeats=5000, seed=0):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, lower=None, upper=None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return dict(n=len(values), mean=float(values.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)))


def segmented_fit(gamma, values):
    x = np.log(np.asarray(gamma, float)); y = np.asarray(values, float)
    order = np.argsort(x); x = x[order]; y = y[order]
    if len(x) < 6 or not np.all(np.isfinite(y)):
        return {"estimable": False, "reason": "At least six finite distinct gamma levels required"}
    if len(np.unique(x)) != len(x):
        raise ValueError("Aggregate within gamma before fitting")
    def fit(design):
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        rss = float(np.sum((y-design@coef)**2))
        return rss, coef
    smooth_rss, _ = fit(np.stack((np.ones_like(x), x, x*x), 1))
    candidates = []
    for knot in x[2:-2]:
        rss, coefficients = fit(np.stack((np.ones_like(x), x, np.maximum(x-knot, 0)), 1))
        candidates.append((rss, knot, coefficients))
    rss, knot, coefficients = min(candidates, key=lambda v:v[0])
    n = len(x)
    # Knot selection contributes one parameter; quadratic null has three.
    bic_segment = n*np.log(max(rss/n, 1e-20))+4*np.log(n)
    bic_smooth = n*np.log(max(smooth_rss/n, 1e-20))+3*np.log(n)
    rho = spearmanr(x, y)
    return dict(estimable=True, gamma_change=float(np.exp(knot)), rss=rss,
                delta_bic=float(bic_segment-bic_smooth), coefficients=coefficients.tolist(),
                spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
                interpretation="Negative delta BIC favors a continuous hinge over a quadratic smooth null")


def bootstrap_transition(gammas, seed_by_gamma, repeats=1000, seed=0):
    array = np.asarray(seed_by_gamma, float)
    if array.ndim != 2 or array.shape[1] != len(gammas):
        raise ValueError("Expected complete seed x gamma matrix")
    if len(gammas) < 6:
        return dict(n_seeds=len(array), repeats=0, lower=None, upper=None,
                    reason="At least six gamma levels required")
    rng = np.random.default_rng(seed)
    transitions = []
    for _ in range(repeats):
        means = array[rng.integers(len(array), size=len(array))].mean(0)
        fit = segmented_fit(gammas, means)
        if fit.get("estimable"):
            transitions.append(fit["gamma_change"])
    return dict(n_seeds=len(array), repeats=repeats,
                lower=float(np.quantile(transitions, .025)) if transitions else None,
                upper=float(np.quantile(transitions, .975)) if transitions else None)
