import numpy as np


def cka(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a-a.mean(0); b = b-b.mean(0)
    denom = np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
    return float(np.linalg.norm(a.T@b)**2/denom) if denom > 0 else float("nan")


def effective_rank(h):
    h = np.asarray(h, dtype=np.float64); h = h-h.mean(0)
    e = np.maximum(np.linalg.eigvalsh(h.T@h), 0)
    if e.sum() == 0:
        return 0.
    p = e[e > 0]/e.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


def pair_indices(n, count=50000, seed=2718):
    rng = np.random.default_rng(seed)
    i = rng.integers(n, size=count)
    j = (i+rng.integers(1, n, size=count)) % n
    return i, j


def distortion(initial, current, count=50000):
    i, j = pair_indices(len(initial), count)
    den = np.linalg.norm(initial[i]-initial[j], axis=1)
    valid = den > 1e-12
    ratios = np.linalg.norm(current[i]-current[j], axis=1)[valid]/den[valid]
    if not len(ratios):
        return {"valid_pairs": 0}
    return {"valid_pairs": int(valid.sum()), "median": float(np.median(ratios)),
            "q05": float(np.quantile(ratios, .05)), "q95": float(np.quantile(ratios, .95)),
            "variance": float(np.var(ratios))}


def rms_scale(h):
    return float(np.sqrt(np.mean(np.sum((h-h.mean(0))**2, axis=1))))
