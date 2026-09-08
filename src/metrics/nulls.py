import numpy as np


def controls(h, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(h.shape[1], h.shape[1])))
    # Permuting whole rows preserves every geometric statistic exactly while
    # destroying correspondence with the latent variables. It is not a new
    # geometry and should leave PH unchanged: a stronger alignment null.
    shuffled = h[rng.permutation(len(h))]
    return {"scale_0.1": .1*h, "rotation": h@q,
            "latent_alignment_permutation": shuffled}


def covariance_cloud(h, seed=0):
    rng = np.random.default_rng(seed)
    centered = h-h.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    cloud = rng.normal(size=(len(h), len(s))) @ (s[:, None]*vt)/np.sqrt(max(len(h)-1, 1))
    return cloud+h.mean(0)
