import numpy as np
import pytest
from src.data.torus import coordinates, geodesic
from src.metrics.fibers import torus_fiber_pairs, paired_separation, noise_decodability


def test_controlled_pairs_hold_task_fixed():
    a, b = torus_fiber_pairs(100)
    np.testing.assert_array_equal(a[:, 0], b[:, 0])
    np.testing.assert_allclose(geodesic(a, b), np.pi, rtol=1e-6)


def test_true_projection_vs_invertible_compression():
    a, b = torus_fiber_pairs(100)
    x, y = coordinates(a).numpy(), coordinates(b).numpy()
    projected = paired_separation(x[:, :2], y[:, :2])
    assert projected['exact_float_equal_pairs'] == 100
    assert projected['raw_q01'] == 0
    x[:, 2:] *= 1e-4; y[:, 2:] *= 1e-4
    retained = paired_separation(x, y)
    assert retained['raw_q01'] > 0
    assert retained['exact_float_equal_pairs'] == 0


def test_normalized_fiber_scale_invariance():
    a, b = torus_fiber_pairs(100)
    x, y = coordinates(a).numpy(), coordinates(b).numpy()
    first = paired_separation(x, y, 2.)
    other = paired_separation(7*x, 7*y, 14.)
    assert other['normalized_q01'] == pytest.approx(first['normalized_q01'], rel=1e-6)


def test_noise_probe_retention_and_projection_control():
    rng = np.random.default_rng(9)
    train_phi = rng.uniform(0, 2*np.pi, 1000); test_phi = rng.uniform(0, 2*np.pi, 500)
    a = np.column_stack((np.cos(train_phi), np.sin(train_phi)))
    b = np.column_stack((np.cos(test_phi), np.sin(test_phi)))
    clean = noise_decodability(a, b, train_phi, test_phi, levels=[0, .1])
    gone = noise_decodability(np.zeros_like(a), np.zeros_like(b), train_phi, test_phi, levels=[0])
    assert clean[0]['angular_cosine'] > .999
    assert clean[1]['angular_cosine'] > .98
    assert abs(gone[0]['angular_cosine']) < .15
    assert clean == noise_decodability(a, b, train_phi, test_phi, levels=[0, .1])


def test_noise_scale_uses_training_only():
    a = np.arange(80.).reshape(40, 2); b = np.ones((20, 2)); p = np.zeros(40); q = np.zeros(20)
    x = noise_decodability(a, b, p, q, levels=[.1])[0]
    y = noise_decodability(a, b*1e6, p, q, levels=[.1])[0]
    assert x['coordinate_noise_sd'] == y['coordinate_noise_sd']


@pytest.mark.parametrize('levels', [[-1], [float('nan')], [float('inf')]])
def test_invalid_noise_levels(levels):
    with pytest.raises(ValueError):
        noise_decodability(np.ones((4, 2)), np.ones((2, 2)), np.zeros(4), np.zeros(2), levels)
