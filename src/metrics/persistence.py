import numpy as np
import hashlib
import json
from pathlib import Path
try:
    from ripser import ripser
except ModuleNotFoundError as exc:
    if exc.name != 'ripser':
        raise
    ripser = None
from src.metrics.geometry import rms_scale
from src.training.checkpoints import atomic_write


def statistics(diagrams):
    out = {}
    for degree, diagram in enumerate(diagrams):
        finite = diagram[np.isfinite(diagram[:, 1])]
        lifetime = np.sort(finite[:, 1]-finite[:, 0])[::-1]
        positive = lifetime[lifetime > 0]
        p = positive/positive.sum() if len(positive) else np.array([])
        out[f"H{degree}"] = {"total_persistence": float(lifetime.sum()),
                              "top1": float(lifetime[0]) if len(lifetime) else 0.,
                              "top2": float(lifetime[1]) if len(lifetime) > 1 else 0.,
                              "entropy": float(-np.sum(p*np.log(p))),
                              "essential_bars": int(np.isinf(diagram[:, 1]).sum())}
    return out


def persistence(h, size=500, repeats=20, maxdim=2, seed=2026, normalize=True, cache_dir=None):
    h = np.asarray(h)
    if h.ndim != 2 or min(h.shape) < 1 or not np.isfinite(h).all():
        raise ValueError("PH requires a finite, nonempty N x D point cloud")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 1000:
        raise ValueError("Rips size must be an integer between 1 and 1000")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("PH repeats must be a positive integer")
    if not isinstance(maxdim, int) or isinstance(maxdim, bool) or maxdim < 0:
        raise ValueError("PH maxdim must be a nonnegative integer")
    cache = None
    if cache_dir is not None:
        digest = hashlib.sha256(np.ascontiguousarray(h).tobytes())
        digest.update(json.dumps(dict(shape=h.shape, dtype=str(h.dtype), size=size, maxdim=maxdim,
                                     seed=seed, normalize=normalize), sort_keys=True).encode())
        cache = Path(cache_dir)/digest.hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
    scale = rms_scale(h) if normalize else 1.
    h = (h-h.mean(0))/(scale if scale > 0 else 1.)
    rng = np.random.default_rng(seed)
    rows, diagrams = [], []
    for repeat in range(repeats):
        ids = rng.choice(len(h), min(size, len(h)), replace=False)
        cached = cache/f"repeat_{repeat:03d}.npz" if cache is not None else None
        if cached is not None and cached.exists():
            with np.load(cached, allow_pickle=False) as saved:
                d = [saved[f"H{i}"] for i in range(maxdim+1)]
                rows.append(json.loads(str(saved["statistics"])))
            diagrams.append(d)
            continue
        if ripser is None:
            raise RuntimeError("Ripser is required for uncached PH; install ripser. No substitute results were generated.")
        d = ripser(h[ids], maxdim=maxdim, coeff=2)["dgms"]
        rows.append({"repeat": repeat, "size": len(ids), "normalized": normalize,
                     "scale": scale, **statistics(d)})
        diagrams.append(d)
        if cached is not None:
            atomic_write(cached, lambda stream: np.savez_compressed(
                stream, indices=ids, statistics=json.dumps(rows[-1]),
                **{f"H{i}":v for i,v in enumerate(d)}))
    return rows, diagrams
