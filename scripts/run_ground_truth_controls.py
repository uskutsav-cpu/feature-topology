"""Evaluate the fiber-survival hierarchy on maps with known ground truth."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.ground_truth import ground_truth_maps
from src.data.torus import labels
from src.metrics.fiber_survival import (
    dense_fiber_separation,
    empirical_regime,
    local_fiber_margin,
)
from src.metrics.geometry import cka, effective_rank, rms_scale
from src.metrics.persistence import persistence
from src.metrics.probes import evaluate_probes
from src.metrics.quotient import image_graph
from src.training.checkpoints import atomic_json


def latent_grid(base_points, fiber_points):
    theta = np.arange(base_points)*2*np.pi/base_points
    phi = np.arange(fiber_points)*2*np.pi/fiber_points
    return np.stack(np.meshgrid(theta, phi, indexing="ij"), axis=-1), phi


def random_latent(count, seed):
    return np.random.default_rng(seed).uniform(0, 2*np.pi, (count, 2))


def run(output, base_points=16, fiber_points=32, ph_size=256, ph_repeats=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid, phi = latent_grid(base_points, fiber_points)
    controls = ground_truth_maps()
    identity = controls[0].evaluate(grid).reshape(-1, 4)
    train_latent = random_latent(2000, 2026091901)
    test_latent = random_latent(1000, 2026091902)
    train_y, test_y = labels(train_latent), labels(test_latent)
    rows = {}
    for control in controls:
        values = control.evaluate(grid)
        flat = values.reshape(-1, values.shape[-1])
        scale = rms_scale(flat)
        local = local_fiber_margin(control.jacobian(grid), representation_scale=scale)
        global_ = dense_fiber_separation(values, phi)
        ph, _ = persistence(flat, size=min(ph_size, len(flat)), repeats=ph_repeats,
                            maxdim=2, seed=2026091903)
        train_h, test_h = control.evaluate(train_latent), control.evaluate(test_latent)
        probes = evaluate_probes(
            train_h, test_h, train_y, test_y,
            train_latent[:, 1], test_latent[:, 1], seed=2026091904,
        )
        fiber_vertices = values[0]
        graph = image_graph(
            fiber_vertices,
            [(np.eye(values.shape[-1]), np.zeros(values.shape[-1]), False)],
        )
        drift = 1-cka(identity, flat)
        rows[control.name] = {
            "truth": {
                "regime": control.regime,
                "description": control.description,
                "injective": control.expected_injective,
                "local_margin": control.expected_local_margin,
                "global_margin": control.expected_global_margin,
            },
            "geometry": {
                "cka_drift_from_isometry": drift,
                "effective_rank": effective_rank(flat),
                "rms_scale": scale,
            },
            "local_fiber_margin": local,
            "global_fiber_separation": global_,
            "empirical_regime": empirical_regime(
                drift, local["minimum"], global_["sampled_minimum"],
                global_["tolerance_collision_pairs"],
            ),
            "nuisance_fiber_image_graph": graph,
            "persistent_homology": ph,
            "held_out_probes": probes,
        }
        atomic_json(output/"progress.json", rows)
    result = {
        "schema": "feature-topology.ground-truth-controls.v1",
        "grid": {"base_points": base_points, "fiber_points": fiber_points},
        "persistent_homology": {
            "subsample_size": min(ph_size, len(identity)), "repeats": ph_repeats,
            "max_dimension": 2,
        },
        "thresholds": {"deformation": .05, "near_singular": 1e-3,
                       "collision_tolerance": 1e-10},
        "qualification": (
            "Analytic truth labels certify these constructed maps. Numerical metrics remain "
            "finite-grid diagnostics and do not become continuum certificates."
        ),
        "controls": rows,
    }
    atomic_json(output/"controls.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ground_truth_controls")
    parser.add_argument("--base-points", type=int, default=16)
    parser.add_argument("--fiber-points", type=int, default=32)
    parser.add_argument("--ph-size", type=int, default=256)
    parser.add_argument("--ph-repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.base_points, args.fiber_points, args.ph_size, args.ph_repeats)
