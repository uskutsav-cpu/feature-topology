import numpy as np
import hashlib
import json
import os
from pathlib import Path
from ripser import ripser
from src.metrics.geometry import rms_scale


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
    if size > 1000:
        raise ValueError("Rips is capped at 1000 points; use subsampling")
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
        d = ripser(h[ids], maxdim=maxdim, coeff=2)["dgms"]
        rows.append({"repeat": repeat, "size": len(ids), "normalized": normalize,
                     "scale": scale, **statistics(d)})
        diagrams.append(d)
        if cached is not None:
            temp = cached.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, indices=ids, statistics=json.dumps(rows[-1]),
                                    **{f"H{i}":v for i,v in enumerate(d)})
            os.replace(temp, cached)
    return rows, diagrams
