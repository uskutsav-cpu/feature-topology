import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.data.torus import geodesic
from src.metrics.geometry import pair_indices


def global_margin(h, latent, delta=.5, count=100000, manifold="torus"):
    i, j = pair_indices(len(h), count)
    d = geodesic(latent[i], latent[j], manifold)
    valid = d > delta
    ratio = np.linalg.norm(h[i[valid]]-h[j[valid]], axis=1)/d[valid]
    if not len(ratio):
        return {"eligible_pairs": 0}
    return {"sampled_min": float(ratio.min()), "q001": float(np.quantile(ratio, .001)),
            "q01": float(np.quantile(ratio, .01)), "q05": float(np.quantile(ratio, .05)),
            "eligible_pairs": int(valid.sum()), "delta": delta}


def collisions(h, latent, k=20, manifold="torus"):
    k = min(k, len(h)-1)
    raw = NearestNeighbors(n_neighbors=k+1, algorithm="brute").fit(h).kneighbors(h, return_distance=False)
    # Explicitly remove self: duplicate representations can move self away
    # from the first position, or outside the returned ties entirely.
    ids = np.array([row[row != i][:k] for i, row in enumerate(raw)])
    d = geodesic(latent[:, None, :], latent[ids], manifold)
    c = d.max(1)
    return {"mean": float(c.mean()), "q95": float(np.quantile(c, .95)), "k": k}
