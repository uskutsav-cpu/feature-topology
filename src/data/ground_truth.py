"""Analytically controlled maps on the product torus.

These maps separate deformation, near-singularity, and actual identification.
They are controls for diagnostics, not learned representations.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundTruthMap:
    name: str
    regime: str
    description: str
    expected_injective: bool
    expected_local_margin: float
    expected_global_margin: float

    def evaluate(self, latent):
        latent = np.asarray(latent, dtype=np.float64)
        theta, phi = latent[..., 0], latent[..., 1]
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        if self.name == "isometry":
            return np.stack((ct, st, cp, sp), axis=-1)
        if self.name == "deformed":
            radius = np.exp(2.0 * ct)
            angle = phi + 1.5 * st
            return np.stack((ct, st, radius*np.cos(angle), radius*np.sin(angle)), axis=-1)
        if self.name == "near_singular":
            epsilon = 1e-4
            return np.stack((ct, st, epsilon*cp, epsilon*sp), axis=-1)
        if self.name == "double_cover":
            return np.stack((ct, st, np.cos(2*phi), np.sin(2*phi)), axis=-1)
        if self.name == "quotient":
            return np.stack((ct, st), axis=-1)
        raise ValueError(self.name)

    def jacobian(self, latent):
        latent = np.asarray(latent, dtype=np.float64)
        theta, phi = latent[..., 0], latent[..., 1]
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        shape = latent.shape[:-1]
        if self.name == "isometry":
            out = np.zeros(shape+(4, 2))
            out[..., :, 0] = np.stack((-st, ct, np.zeros(shape), np.zeros(shape)), axis=-1)
            out[..., :, 1] = np.stack((np.zeros(shape), np.zeros(shape), -sp, cp), axis=-1)
            return out
        if self.name == "deformed":
            radius = np.exp(2.0*ct)
            angle = phi+1.5*st
            ca, sa = np.cos(angle), np.sin(angle)
            radius_theta = -2.0*st*radius
            angle_theta = 1.5*ct
            out = np.zeros(shape+(4, 2))
            out[..., :, 0] = np.stack((
                -st, ct,
                radius_theta*ca-radius*sa*angle_theta,
                radius_theta*sa+radius*ca*angle_theta,
            ), axis=-1)
            out[..., :, 1] = np.stack((
                np.zeros(shape), np.zeros(shape), -radius*sa, radius*ca,
            ), axis=-1)
            return out
        if self.name == "near_singular":
            epsilon = 1e-4
            out = np.zeros(shape+(4, 2))
            out[..., :, 0] = np.stack((-st, ct, np.zeros(shape), np.zeros(shape)), axis=-1)
            out[..., :, 1] = np.stack((np.zeros(shape), np.zeros(shape), -epsilon*sp, epsilon*cp), axis=-1)
            return out
        if self.name == "double_cover":
            out = np.zeros(shape+(4, 2))
            out[..., :, 0] = np.stack((-st, ct, np.zeros(shape), np.zeros(shape)), axis=-1)
            out[..., :, 1] = np.stack((
                np.zeros(shape), np.zeros(shape), -2*np.sin(2*phi), 2*np.cos(2*phi),
            ), axis=-1)
            return out
        if self.name == "quotient":
            out = np.zeros(shape+(2, 2))
            out[..., :, 0] = np.stack((-st, ct), axis=-1)
            return out
        raise ValueError(self.name)


def ground_truth_maps():
    """Return the frozen diagnostic suite with analytic margin values."""
    return (
        GroundTruthMap("isometry", "preserved", "Product-circle isometry.", True, 1.0, 2/np.pi),
        GroundTruthMap(
            "deformed", "deformed",
            "Injective radius-and-twist deformation with recoverable theta and phi.",
            True, np.exp(-2.0), 2*np.exp(-2.0)/np.pi,
        ),
        GroundTruthMap(
            "near_singular", "near-singular",
            "Injective nuisance circle scaled by epsilon=1e-4.",
            True, 1e-4, 2e-4/np.pi,
        ),
        GroundTruthMap(
            "double_cover", "quotiented",
            "Everywhere-regular nuisance double cover: positive local margin but global collisions.",
            False, 2.0, 0.0,
        ),
        GroundTruthMap(
            "quotient", "quotiented",
            "Task quotient that removes the nuisance circle.",
            False, 0.0, 0.0,
        ),
    )
