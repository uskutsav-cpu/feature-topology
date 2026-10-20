"""Numerical witness construction for a separate continuous-fiber feasibility study.

These routines are approximate. Neither sampled derivatives nor a fitted
readout constitute a continuum certificate.
"""
import numpy as np


def fit_circular_readout(features, phi, relative_ridge=1e-10):
    features, phi = np.asarray(features, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    target = np.column_stack([np.cos(phi), np.sin(phi)])
    mean, target_mean = features.mean(axis=0), target.mean(axis=0)
    centered = features-mean
    gram = centered.T @ centered
    ridge = relative_ridge*np.trace(gram)/features.shape[1]
    if not np.isfinite(gram).all() or ridge <= 0:
        raise ValueError('Circular witness needs finite, nonconstant representations')
    readout = np.linalg.solve(gram+ridge*np.eye(gram.shape[0]), centered.T @ (target-target_mean)).T
    intercept = target_mean-readout @ mean
    error = features @ readout.T+intercept-target
    return readout, intercept, dict(ridge=float(ridge),
                                    fit_rmse=float(np.sqrt(np.mean(error**2))))


def features_and_phi_derivative(network, latent, embedding):
    """Exact chain-rule structure evaluated in ordinary floating-point arithmetic."""
    import torch
    latent = torch.as_tensor(latent, dtype=torch.float64)
    embedding = torch.as_tensor(embedding, dtype=torch.float64)
    theta, phi = latent.unbind(-1)
    coordinates = torch.stack([theta.cos(), theta.sin(), phi.cos(), phi.sin()], dim=1)
    x = coordinates @ embedding.T
    derivative = torch.stack([torch.zeros_like(phi), torch.zeros_like(phi),
                              -phi.sin(), phi.cos()], dim=1) @ embedding.T
    for layer in network.hidden:
        before = x @ layer.weight.T+layer.bias
        derivative = (derivative @ layer.weight.T)*(before > 0)
        x = torch.relu(before)
    return x, derivative
