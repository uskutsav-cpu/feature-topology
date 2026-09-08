import numpy as np
import torch
from src.data.images import rotated_digits
from src.models.cnn import CNN
from src.analysis.statistics import bootstrap_mean, segmented_fit


def test_image_identity_split():
    data = rotated_digits(rotations=1)
    identities = [set(data[k][3]) for k in ["train", "validation", "test"]]
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]
    assert CNN()(data["train"][0][:3]).shape == (3, 10)


def test_seed_bootstrap_and_segment():
    ci = bootstrap_mean([1., 2., 3.], repeats=500)
    assert ci["lower"] <= 2 <= ci["upper"]
    gammas = 2.**np.arange(-5, 8)
    x = np.log(gammas)
    fit = segmented_fit(gammas, 1+x+3*np.maximum(x, 0))
    assert np.isclose(fit["gamma_change"], 1)
    assert fit["delta_bic"] < 0
