import numpy as np
import torch
from torch.func import jacfwd, vmap
from src.data.torus import coordinates


def tangent_jacobians(network, latent, q, layer, manifold="torus", batch_size=128):
    # Forward AD is efficient here: the domain has only two coordinates.
    def f(z):
        return network.representations(coordinates(z, manifold) @ q.T)[layer]
    chunks = []
    for z in latent.split(batch_size):
        chunks.append(vmap(jacfwd(f))(z).detach().cpu().numpy())
    return np.concatenate(chunks)


def summarize(j, scale=1., swap=False):
    s = np.linalg.svd(j, compute_uv=False)[..., -1]
    out = {"min": float(s.min()), "q001": float(np.quantile(s, .001)),
           "q01": float(np.quantile(s, .01)), "q05": float(np.quantile(s, .05)),
           "median": float(np.median(s)),
           "task_norm": float(np.median(np.linalg.norm(j[..., int(swap)], axis=-1))),
           "nuisance_norm": float(np.median(np.linalg.norm(j[..., 1-int(swap)], axis=-1)))}
    out.update({f"fraction_below_{t:g}": float(np.mean(s < t)) for t in [1e-4, 1e-3, 1e-2]})
    out["normalized_q01"] = float(np.quantile(s, .01)/scale) if scale > 0 else 0.
    return out, s
