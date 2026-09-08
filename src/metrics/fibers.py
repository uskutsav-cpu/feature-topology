"""Factor-controlled separation and finite-noise decodability diagnostics.

Neither a positive finite-sample margin nor probe failure certifies injectivity
or information destruction. Noise is an explicit intervention, not training.
"""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.torus import coordinates


def torus_fiber_pairs(samples=512, seed=20260908, nuisance=1):
    """Paired latent points: hold task angle fixed, shift nuisance by pi."""
    if not isinstance(samples, int) or samples < 1 or nuisance not in (0, 1):
        raise ValueError('samples must be positive and nuisance must be 0 or 1')
    a = np.random.default_rng(seed).uniform(0, 2*np.pi, (samples, 2))
    b = a.copy(); b[:, nuisance] = (b[:, nuisance]+np.pi) % (2*np.pi)
    return torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def paired_separation(a, b, representation_scale=1.0):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or len(a) == 0:
        raise ValueError('paired representations must have equal, nonempty N x D shapes')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite representation')
    if not np.isfinite(representation_scale) or representation_scale < 0:
        raise ValueError('representation_scale must be finite and nonnegative')
    distance = np.linalg.norm(a-b, axis=1)
    ratio = distance/np.pi
    return dict(pairs=len(a), latent_separation=float(np.pi),
                raw_min=float(ratio.min()), raw_q01=float(np.quantile(ratio, .01)),
                raw_median=float(np.median(ratio)),
                normalized_q01=float(np.quantile(ratio, .01)/representation_scale)
                    if representation_scale > 0 else None,
                exact_float_equal_pairs=int(np.sum(np.all(a == b, axis=1))),
                qualification='Sampled pi-separated nuisance fibers; floating-point equality is not a continuum certificate.')


def noise_decodability(train_h, test_h, train_phi, test_phi,
                       levels=(0., .001, .01, .1), seed=0):
    """Held-out circular ridge probes with noise added before fitted scaling.

    Noise SD per coordinate = level * training RMS / sqrt(width), so expected
    noise norm = level * training RMS. One paired noise draw is scaled across
    levels; train/test draws are independent. No test-derived normalization.
    """
    a = np.asarray(train_h, dtype=np.float64); b = np.asarray(test_h, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(a.shape) < 1 or len(b) < 1:
        raise ValueError('invalid train/test representation shapes')
    if len(a) != len(train_phi) or len(b) != len(test_phi):
        raise ValueError('targets must match representations')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite representation')
    levels = tuple(float(v) for v in levels)
    if any(not np.isfinite(v) or v < 0 for v in levels):
        raise ValueError('noise levels must be finite and nonnegative')
    target = np.column_stack((np.cos(train_phi), np.sin(train_phi)))
    rms = float(np.sqrt(np.mean(np.sum((a-a.mean(0))**2, axis=1))))
    rng = np.random.default_rng(seed)
    an = rng.normal(size=a.shape); bn = rng.normal(size=b.shape)
    rows = []
    for level in levels:
        sd = level*rms/np.sqrt(a.shape[1])
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.))
        probe.fit(a+sd*an, target)
        pred = probe.predict(b+sd*bn)
        error = np.angle(np.exp(1j*(np.arctan2(pred[:, 1], pred[:, 0])-test_phi)))
        rows.append(dict(noise_level=level, training_rms=rms, coordinate_noise_sd=float(sd),
                         angular_cosine=float(np.cos(error).mean()),
                         angular_mae=float(np.abs(error).mean())))
    return rows
