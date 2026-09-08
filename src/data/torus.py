import numpy as np
import torch


def embedding(seed=31415, dimension=16):
    if dimension < 4:
        raise ValueError("Embedding dimension must be at least four")
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, 4)))
    return q.astype(np.float32)


def coordinates(latent, manifold="torus"):
    t, p = latent.unbind(-1)
    if manifold == "torus":
        return torch.stack((t.cos(), t.sin(), p.cos(), p.sin()), -1)
    if manifold == "cylinder":
        return torch.stack((t.cos(), t.sin(), p, torch.zeros_like(p)), -1)
    raise ValueError(manifold)


def labels(latent, swap=False, relevance=0., relevance_mode="periodic"):
    t, p = latent[:, int(swap)], latent[:, 1-int(swap)]
    # A fractional multiple of an angle is not well-defined on S1. The
    # periodic default is continuous across the nuisance seam; literal mode
    # reproduces the brief's coordinate-dependent theta + lambda*phi.
    angle = t + relevance * (np.sin(p) if relevance_mode == "periodic" else p)
    return (np.mod(angle, 2*np.pi) / (np.pi/2)).astype(np.int64)


def dataset(n=20000, seed=0, dimension=16, manifold="torus", grid=False,
            swap=False, relevance=0., relevance_mode="periodic"):
    rng = np.random.default_rng(seed)
    if grid:
        side = int(np.sqrt(n))
        if side*side != n:
            raise ValueError("Grid size must be a perfect square")
        t = np.arange(side)*2*np.pi/side
        p = t if manifold == "torus" else np.linspace(0, 1, side)
        latent = np.stack(np.meshgrid(t, p, indexing="ij"), -1).reshape(-1, 2)
    else:
        latent = rng.uniform(0, 2*np.pi, (n, 2))
        if manifold == "cylinder":
            latent[:, 1] /= 2*np.pi
    q = embedding(dimension=dimension)
    z = torch.tensor(latent, dtype=torch.float32)
    x = coordinates(z, manifold) @ torch.from_numpy(q).T
    y = torch.from_numpy(labels(latent, swap, relevance, relevance_mode))
    return x, y, z, torch.from_numpy(q)


def geodesic(a, b, manifold="torus"):
    d = np.abs(np.asarray(a)-np.asarray(b))
    d[..., 0] = np.minimum(d[..., 0] % (2*np.pi), 2*np.pi-d[..., 0] % (2*np.pi))
    if manifold == "torus":
        d[..., 1] = np.minimum(d[..., 1] % (2*np.pi), 2*np.pi-d[..., 1] % (2*np.pi))
    return np.linalg.norm(d, axis=-1)
