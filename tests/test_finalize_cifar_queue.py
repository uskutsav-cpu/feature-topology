import hashlib
import json

import pytest

from scripts.finalize_cifar_queue import promote
from src.training.checkpoints import atomic_json


def fixture(tmp_path, *, production_complete=35):
    config = tmp_path / "configs/completion"
    collections = tmp_path / "results/completion/remote_collections"
    config.mkdir(parents=True)
    collections.mkdir(parents=True)
    specification = config / "cifar_sources_v3.json"
    atomic_json(specification, {"schema": "feature-topology.cifar-sources.v1"})
    specification_hash = hashlib.sha256(specification.read_bytes()).hexdigest()
    commit = "a" * 40
    state = {
        "schema": "feature-topology.remote-cifar-queue.v1",
        "source_commit": commit,
        "attempts": [{"dataset": "CIFAR10", "stage": "production",
                      "workflow_run": "123", "url": "https://github.com/o/r/actions/runs/123",
                      "cell_ids": ["b" * 16], "started": 1.0, "finished": 2.0,
                      "conclusion": "success"}],
        "CIFAR10": {
            "calibration": {"expected": 49, "complete": 49, "missing": 0, "invalid": []},
            "production": {"expected": 35, "complete": production_complete,
                           "missing": 35-production_complete, "invalid": []},
        },
    }
    source = tmp_path / "results/completion/cifar10_remote_queue_source.json"
    atomic_json(source, state)
    for stage, expected in (("calibration", 49), ("production", 35)):
        complete = expected if stage == "calibration" else production_complete
        atomic_json(collections / f"cifar10_{stage}.json", {
            "schema": "feature-topology.remote-cifar-collection.v1",
            "source_commit": commit,
            "source_specification": "configs/completion/cifar_sources_v3.json",
            "source_specification_sha256": specification_hash,
            "expected": expected,
            "complete": {str(index): "asset" for index in range(complete)},
            "missing": [] if complete == expected else ["missing"], "invalid": [],
        })
    return source, state


def test_complete_ledger_is_promoted_byte_canonically(tmp_path):
    source, state = fixture(tmp_path)
    output = tmp_path / "results/completion/cifar10_remote_queue_v3.json"
    assert promote(tmp_path, "CIFAR10", source, output) == state
    assert json.loads(output.read_text()) == state


def test_partial_ledger_is_never_promoted(tmp_path):
    source, _ = fixture(tmp_path, production_complete=34)
    output = tmp_path / "results/completion/cifar10_remote_queue_v3.json"
    with pytest.raises(ValueError, match="stage incomplete"):
        promote(tmp_path, "CIFAR10", source, output)
    assert not output.exists()


def test_conflicting_canonical_ledger_is_not_replaced(tmp_path):
    source, _ = fixture(tmp_path)
    output = tmp_path / "results/completion/cifar10_remote_queue_v3.json"
    atomic_json(output, {"different": True})
    with pytest.raises(ValueError, match="conflicting canonical"):
        promote(tmp_path, "CIFAR10", source, output)
