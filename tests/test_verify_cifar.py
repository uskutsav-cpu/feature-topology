import pytest

from scripts.verify_cifar import cached_replay


def test_cached_replay_requires_every_artifact_binding():
    binding = {
        "dataset": "CIFAR10", "run_id": "a" * 16,
        "checkpoint_sha256": "b" * 64, "metrics_sha256": "c" * 64,
        "representations_sha256": "d" * 64,
        "saved_test": {"loss": 1.0, "accuracy": .5},
    }
    key = "CIFAR10/" + "a" * 16
    cache = {"records": {key: {**binding,
                                "replayed_test": {"loss": 1.0, "accuracy": .5}}}}
    assert cached_replay(cache, key, binding) == {"loss": 1.0, "accuracy": .5}
    changed = {**binding, "metrics_sha256": "e" * 64}
    assert cached_replay(cache, key, changed) is None


def test_cached_replay_rejects_nonfinite_measurement():
    key = "CIFAR100/run"
    binding = {"dataset": "CIFAR100"}
    cache = {"records": {key: {**binding,
                                "replayed_test": {"loss": None, "accuracy": .5}}}}
    with pytest.raises(ValueError, match="Non-finite cached"):
        cached_replay(cache, key, binding)
