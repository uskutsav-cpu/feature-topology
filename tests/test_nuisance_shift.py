import json

import numpy as np
import pytest

from scripts.evaluate_nuisance_shift import evaluate_run
from src.data.torus import dataset, labels, sample_latent
from src.training.train import train


def circular_distance(a, b):
    delta = np.abs(a-b) % (2*np.pi)
    return np.minimum(delta, 2*np.pi-delta)


def test_iid_sampling_is_backward_compatible():
    old = dataset(100, seed=7)
    explicit = dataset(100, seed=7, nuisance_condition="iid", distribution_split="train")
    for left, right in zip(old, explicit):
        np.testing.assert_array_equal(left, right)


def test_concentrated_shift_moves_nuisance_mode():
    train = sample_latent(5000, 1, nuisance_condition="concentrated", distribution_split="train")
    test = sample_latent(5000, 2, nuisance_condition="concentrated", distribution_split="test")
    assert np.mean(np.cos(train[:, 1])) > .8
    assert np.mean(np.cos(test[:, 1])) < -.8


def test_spurious_association_reverses():
    train = sample_latent(4000, 3, nuisance_condition="spurious", distribution_split="train")
    test = sample_latent(4000, 4, nuisance_condition="spurious", distribution_split="test")
    for latent, offset in [(train, 0), (test, 2)]:
        target = (labels(latent)*0.5*np.pi+np.pi/4+offset*np.pi/2) % (2*np.pi)
        assert circular_distance(latent[:, 1], target).mean() < .3


def test_unseen_arcs_are_disjoint():
    train = sample_latent(1000, 5, nuisance_condition="unseen", distribution_split="train")
    test = sample_latent(1000, 6, nuisance_condition="unseen", distribution_split="test")
    assert train[:, 1].max() < 1.5*np.pi
    assert test[:, 1].min() >= 1.5*np.pi


def test_invalid_shift_requests_rejected():
    with pytest.raises(ValueError):
        sample_latent(10, 0, nuisance_condition="bad")
    with pytest.raises(ValueError):
        sample_latent(10, 0, manifold="cylinder", nuisance_condition="unseen")


def test_matched_risk_ood_evaluation(tmp_path):
    config = dict(gamma=1., lr=.1, seed=0, width=8, depth=1, n_train=64,
                  n_validation=32, n_grid=25, max_steps=1, target_loss=10.)
    result = train(config, tmp_path/"runs")
    row = evaluate_run(tmp_path/"runs"/result["run_id"], target=10., samples=32)
    assert row["status"] == "evaluated"
    assert row["selected_step"] == 0
    assert set(row["environments"]) == {"iid", "concentrated", "spurious", "unseen"}
    assert json.loads(json.dumps(row)) == row
