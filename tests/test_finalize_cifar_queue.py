import hashlib
import json

import pytest

from scripts.cifar_provenance import expected_cell_ids, merge_collection_reports
from scripts.finalize_cifar_queue import promote, promote_multisource
from scripts.remote_cifar import GAMMAS
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


def test_multisource_promotion_preserves_each_corrective_cohort(tmp_path):
    config = tmp_path / "configs/completion"; config.mkdir(parents=True)
    completion = tmp_path / "results/completion"; collections = completion / "remote_collections"
    collections.mkdir(parents=True)
    specification = config / "cifar_sources_v3.json"
    atomic_json(specification, {"schema": "feature-topology.cifar-sources.v1"})
    frozen_dir = tmp_path / "results/cifar10"; frozen_dir.mkdir(parents=True)
    atomic_json(frozen_dir / "gamma_to_lr.json", {
        "selection": {str(gamma): {"lr": .01} for gamma in GAMMAS}})
    calibration_ids = sorted(expected_cell_ids(tmp_path, "CIFAR10", "calibration"))
    production_ids = sorted(expected_cell_ids(tmp_path, "CIFAR10", "production"))
    original_ids, correction_ids = production_ids[:-5], production_ids[-5:]
    specification_hash = hashlib.sha256(specification.read_bytes()).hexdigest()
    original_commit, correction_commit = "a" * 40, "c" * 40

    def report(path, commit, tag, identifiers):
        atomic_json(path, {
            "schema": "feature-topology.remote-cifar-collection.v1",
            "result_tag": tag, "source_commit": commit,
            "source_specification": "configs/completion/cifar_sources_v3.json",
            "source_specification_sha256": specification_hash,
            "expected": len(identifiers),
            "complete": {identifier: f"cifar-{identifier}.tar.gz"
                         for identifier in identifiers},
            "partial": {}, "missing": [], "invalid": [], "installed_files": {},
        })

    calibration = collections / "cifar10_calibration.json"
    original = collections / "cifar10_production_original.json"
    correction = collections / "cifar10_production_correction.json"
    merged = collections / "cifar10_production.json"
    report(calibration, original_commit, "original-tag", calibration_ids)
    report(original, original_commit, "original-tag", original_ids)
    report(correction, correction_commit, "correction-tag", correction_ids)
    merged_value = merge_collection_reports(
        tmp_path, "CIFAR10", "production", [original, correction], merged)
    assert set(merged_value["complete"]) == set(production_ids)
    assert {row["source_commit"] for row in merged_value["components"]} == {
        original_commit, correction_commit}

    attempt = {"dataset": "CIFAR10", "stage": "production", "workflow_run": "123",
               "url": "https://github.com/o/r/actions/runs/123", "cell_ids": original_ids,
               "started": 1.0, "finished": 2.0, "conclusion": "success"}
    original_ledger = completion / "original.json"
    atomic_json(original_ledger, {
        "schema": "feature-topology.remote-cifar-queue.v1",
        "source_commit": original_commit, "attempts": [attempt]})
    correction_ledger = completion / "correction.json"
    atomic_json(correction_ledger, {
        "schema": "feature-topology.remote-cifar-correction.v1",
        "dataset": "CIFAR10", "stage": "production", "reason": "exact transport",
        "source_commit": correction_commit, "result_tag": "correction-tag",
        "expected_cells": 5, "workflow_run": "456",
        "url": "https://github.com/o/r/actions/runs/456", "started": 3.0,
        "finished": 4.0, "conclusion": "success", "cell_ids": correction_ids})
    canonical = completion / "cifar10_remote_queue_v3.json"
    state = promote_multisource(
        tmp_path, "CIFAR10", [original_ledger, correction_ledger],
        {"calibration": calibration, "production": merged}, canonical)
    assert state["schema"] == "feature-topology.remote-cifar-queue.v2"
    assert {row["source_commit"] for row in state["sources"]} == {
        original_commit, correction_commit}


def test_multisource_merge_rejects_overlapping_cells(tmp_path):
    config = tmp_path / "configs/completion"; config.mkdir(parents=True)
    specification = config / "cifar_sources_v3.json"
    atomic_json(specification, {"schema": "feature-topology.cifar-sources.v1"})
    frozen_dir = tmp_path / "results/cifar10"; frozen_dir.mkdir(parents=True)
    atomic_json(frozen_dir / "gamma_to_lr.json", {
        "selection": {str(gamma): {"lr": .01} for gamma in GAMMAS}})
    identifier = next(iter(expected_cell_ids(tmp_path, "CIFAR10", "production")))
    reports = tmp_path / "results/completion/remote_collections"; reports.mkdir(parents=True)
    specification_hash = hashlib.sha256(specification.read_bytes()).hexdigest()
    for index in range(2):
        atomic_json(reports / f"part{index}.json", {
            "schema": "feature-topology.remote-cifar-collection.v1",
            "result_tag": f"tag-{index}", "source_commit": str(index + 1) * 40,
            "source_specification": "configs/completion/cifar_sources_v3.json",
            "source_specification_sha256": specification_hash, "expected": 1,
            "complete": {identifier: f"cifar-{identifier}.tar.gz"},
            "missing": [], "invalid": []})
    with pytest.raises(ValueError, match="Overlapping"):
        merge_collection_reports(tmp_path, "CIFAR10", "production",
                                 [reports / "part0.json", reports / "part1.json"],
                                 reports / "merged.json")
