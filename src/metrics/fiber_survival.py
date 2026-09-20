"""Metrics for distinguishing deformation, degeneration, and identification.

The dense statistics are finite-grid evidence.  Only controls with analytic
definitions may use the accompanying exact expectations as certificates.
"""
import numpy as np


def local_fiber_margin(jacobian, nuisance_index=1, representation_scale=1.0):
    """Summarize ||Dh|TF|| for a one-dimensional nuisance fiber."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim < 3 or jacobian.shape[-1] <= nuisance_index:
        raise ValueError("jacobian must have shape (..., output, latent)")
    if not np.isfinite(jacobian).all():
        raise ValueError("nonfinite jacobian")
    if not np.isfinite(representation_scale) or representation_scale < 0:
        raise ValueError("invalid representation scale")
    values = np.linalg.norm(jacobian[..., :, nuisance_index], axis=-1).reshape(-1)
    return {
        "minimum": float(values.min()),
        "q01": float(np.quantile(values, .01)),
        "median": float(np.median(values)),
        "normalized_minimum": float(values.min()/representation_scale)
            if representation_scale > 0 else None,
        "samples": int(values.size),
        "qualification": "Dense-sample differential evidence; not a continuum lower bound.",
    }


def dense_fiber_separation(representations, fiber_values, collision_tolerance=1e-10,
                           periodic=True):
    """Evaluate all distinct within-fiber pairs on a B x F x D grid."""
    h = np.asarray(representations, dtype=np.float64)
    phi = np.asarray(fiber_values, dtype=np.float64)
    if h.ndim != 3 or h.shape[1] != len(phi) or h.shape[1] < 2:
        raise ValueError("representations must be B x F x D and match fiber_values")
    if not np.isfinite(h).all() or not np.isfinite(phi).all():
        raise ValueError("nonfinite input")
    if collision_tolerance < 0 or not np.isfinite(collision_tolerance):
        raise ValueError("invalid collision tolerance")
    delta = np.abs(phi[:, None]-phi[None, :])
    if periodic:
        delta %= 2*np.pi
        latent_distance = np.minimum(delta, 2*np.pi-delta)
    else:
        latent_distance = delta
    upper = np.triu(np.ones_like(latent_distance, dtype=bool), 1)
    ratios, distances = [], []
    exact, tolerant = 0, 0
    for fiber in h:
        distance = np.linalg.norm(fiber[:, None, :]-fiber[None, :, :], axis=-1)
        selected = distance[upper]
        distances.append(selected)
        ratios.append(selected/latent_distance[upper])
        exact += int(np.sum(np.all(fiber[:, None, :] == fiber[None, :, :], axis=-1)[upper]))
        tolerant += int(np.sum(selected <= collision_tolerance))
    ratios = np.concatenate(ratios)
    distances = np.concatenate(distances)
    return {
        "sampled_minimum": float(ratios.min()),
        "q001": float(np.quantile(ratios, .001)),
        "q01": float(np.quantile(ratios, .01)),
        "minimum_representation_distance": float(distances.min()),
        "exact_float_collision_pairs": exact,
        "tolerance_collision_pairs": tolerant,
        "collision_tolerance": float(collision_tolerance),
        "pairs": int(ratios.size),
        "periodic_fiber": bool(periodic),
        "qualification": "Exhaustive only on the declared grid; not a continuum certificate.",
    }


def empirical_regime(cka_drift, local_margin, global_margin, collision_pairs,
                     deformation_threshold=.05, singular_threshold=1e-3):
    """Apply frozen descriptive thresholds; the label is not a proof."""
    if collision_pairs > 0:
        return "sampled-collision"
    if min(local_margin, global_margin) < singular_threshold:
        return "near-singular"
    if cka_drift >= deformation_threshold:
        return "deformed"
    return "preserved"
