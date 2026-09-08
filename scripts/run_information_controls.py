"""Known invertible amplification versus exact projection controls.

For every finite A>0, (A cos(theta), A sin(theta), cos(phi), sin(phi)) is
injective on S1 x S1. Decreasing normalized margins is therefore not sufficient
for loss of information. Projection removes the phi coordinates by construction.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.checkpoints import atomic_json
from src.metrics.fibers import paired_separation, noise_decodability


def map_latent(latent, amplitude, projection=False):
    theta, phi = latent.T
    keep = 0. if projection else 1.
    return np.column_stack((amplitude*np.cos(theta), amplitude*np.sin(theta),
                            keep*np.cos(phi), keep*np.sin(phi)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='results/continuation/information_controls.json')
    args = p.parse_args(); rows = []
    for seed in (200, 201, 202):
        rng = np.random.default_rng(seed)
        train = rng.uniform(0, 2*np.pi, (2000, 2))
        test = rng.uniform(0, 2*np.pi, (1000, 2))
        paired = test.copy(); paired[:, 1] = (paired[:, 1]+np.pi) % (2*np.pi)
        for amplitude in (1., 10., 100., 1000.):
            for projection in (False, True):
                a = map_latent(train, amplitude, projection)
                b = map_latent(test, amplitude, projection)
                c = map_latent(paired, amplitude, projection)
                scale = float(np.sqrt(np.mean(np.sum((a-a.mean(0))**2, axis=1))))
                rows.append(dict(seed=seed, amplitude=amplitude, projection=projection,
                    known_map='noninjective_phi_projection' if projection else 'injective_torus_embedding',
                    tangent_sigma_min=0. if projection else min(amplitude, 1.),
                    normalized_tangent_sigma_min=(0. if projection else min(amplitude, 1.)/scale),
                    fibers=paired_separation(b, c, scale),
                    noise=noise_decodability(a, b, train[:, 1], test[:, 1], seed=seed)))
    atomic_json(args.output, rows)
    print(json.dumps(dict(completed_controls=len(rows), output=args.output)))


if __name__ == '__main__': main()
