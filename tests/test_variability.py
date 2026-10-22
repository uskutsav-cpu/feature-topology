import numpy as np
import pytest

from src.metrics.variability import class_covariances, variability_baselines


def test_vci_is_invariant_to_well_conditioned_invertible_linear_transform():
    rng = np.random.default_rng(73)
    labels = np.repeat(np.arange(4), 300)
    values = rng.normal(size=(1200, 8))+rng.normal(size=(4, 8))[labels]
    left, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    right, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    transform = left @ np.diag(np.linspace(.3, 3, 8)) @ right
    original = variability_baselines(values, labels)
    changed = variability_baselines(values @ transform+3, labels)
    assert changed['vci'] == pytest.approx(original['vci'], abs=1e-11)
    assert changed['rank_between'] == original['rank_between'] == 3
    assert changed['within_to_between_covariance_trace_ratio'] != pytest.approx(
        original['within_to_between_covariance_trace_ratio'])


def test_exact_class_collapse_is_zero_but_constant_representation_is_undefined():
    labels = np.repeat(np.arange(4), 20)
    collapsed = variability_baselines(np.eye(4)[labels], labels)
    assert collapsed['vci'] == pytest.approx(0, abs=1e-12)
    assert collapsed['within_to_between_covariance_trace_ratio'] == 0
    constant = variability_baselines(np.ones((80, 4)), labels)
    assert constant['status'] == 'undefined_between_covariance'
    assert constant['vci'] is None
    assert constant['within_to_between_covariance_trace_ratio'] is None


def test_class_balanced_covariance_decomposition_with_unequal_class_counts():
    values = np.array([[1., 2.], [3, 4], [10, -2], [10, 0], [10, 2]])
    labels = np.array([0, 0, 1, 1, 1])
    within, between, total = class_covariances(values, labels, classes=2)
    weights = np.where(labels == 0, 1/4, 1/6)
    mean = np.sum(values*weights[:, None], axis=0)
    direct = (values-mean).T @ ((values-mean)*weights[:, None])
    assert np.allclose(total, direct)
    assert np.allclose(total, within+between)
    assert np.linalg.eigvalsh(within).min() >= -1e-12
    assert np.linalg.eigvalsh(between).min() >= -1e-12


def test_missing_classes_and_nonfinite_features_are_rejected():
    with pytest.raises(ValueError, match='Every declared class'):
        variability_baselines(np.ones((2, 4)), [0, 1])
    with pytest.raises(ValueError, match='finite'):
        variability_baselines(np.full((4, 2), np.nan), np.arange(4))
