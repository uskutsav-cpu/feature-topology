import numpy as np
import pytest

from src.data.ground_truth import ground_truth_maps
from src.metrics.fiber_survival import (
    dense_fiber_separation,
    empirical_regime,
    local_fiber_margin,
)


def grid(base_points=12, fiber_points=24):
    theta = np.arange(base_points)*2*np.pi/base_points
    phi = np.arange(fiber_points)*2*np.pi/fiber_points
    return np.stack(np.meshgrid(theta, phi, indexing="ij"), axis=-1), phi


@pytest.mark.parametrize("control", ground_truth_maps(), ids=lambda control: control.name)
def test_ground_truth_local_margins(control):
    latent, _ = grid()
    measured = local_fiber_margin(control.jacobian(latent))["minimum"]
    assert measured == pytest.approx(control.expected_local_margin, rel=1e-10, abs=1e-12)


def test_global_margin_separates_near_singularity_from_quotient():
    latent, phi = grid()
    controls = {control.name: control for control in ground_truth_maps()}
    near = dense_fiber_separation(controls["near_singular"].evaluate(latent), phi)
    quotient = dense_fiber_separation(controls["quotient"].evaluate(latent), phi)
    assert near["sampled_minimum"] > 0
    assert near["tolerance_collision_pairs"] == 0
    assert quotient["sampled_minimum"] == 0
    assert quotient["tolerance_collision_pairs"] > 0


def test_positive_local_margin_does_not_imply_global_injectivity():
    latent, phi = grid(fiber_points=24)
    control = next(x for x in ground_truth_maps() if x.name == "double_cover")
    local = local_fiber_margin(control.jacobian(latent))
    global_ = dense_fiber_separation(control.evaluate(latent), phi)
    assert local["minimum"] == pytest.approx(2.0)
    assert global_["tolerance_collision_pairs"] > 0
    assert global_["sampled_minimum"] < 1e-12


def test_empirical_regime_precedence():
    assert empirical_regime(.9, 2., 0., 1) == "sampled-collision"
    assert empirical_regime(.9, 1e-4, 1e-4, 0) == "near-singular"
    assert empirical_regime(.2, 1., 1., 0) == "deformed"
    assert empirical_regime(.0, 1., 1., 0) == "preserved"


def test_nonperiodic_fiber_uses_interval_distance():
    fiber = np.linspace(0, 1, 5)
    values = fiber[None, :, None]
    result = dense_fiber_separation(values, fiber, periodic=False)
    assert result["sampled_minimum"] == pytest.approx(1.0)
    assert not result["periodic_fiber"]
