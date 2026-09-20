import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.collect_remote_metrics import install_archive, validate_archive
from src.training.checkpoints import fingerprint


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def make_archive(path, index, provenance, payloads):
    run_id = index["runs"][0]["run_id"]
    profile = fingerprint(index["metric_options"])
    files = {f"main/runs/{run_id}/metrics/{profile}/{name}": digest_bytes(value)
             for name, value in payloads.items()}
    report = {
        "schema": "feature-topology.remote-metrics.v1",
        "run_id": run_id,
        "job_id": "job-1",
        "complete": True,
        "completed": ["final", "step_0000000"],
        "expected": ["final", "step_0000000"],
        "input_archive_sha256": index["runs"][0]["archive_sha256"],
        "analysis_data_sha256": index["analysis_data"]["sha256"],
        "metric_provenance": provenance,
        "host": {},
        "files": files,
    }
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for name, value in payloads.items():
                archive_name = f"main/runs/{run_id}/metrics/{profile}/{name}"
                info = tarfile.TarInfo(archive_name)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
            encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
            info = tarfile.TarInfo("job_manifest.json")
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    return {**report, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_complete_archive_validation_and_canonical_install(tmp_path):
    run_id = "a" * 16
    training = {"config.json": b"config", "summary.json": b"summary",
                "step_0000000.pt": b"initial", "final.pt": b"final"}
    index = {
        "runs": [{"run_id": run_id, "archive_sha256": "input-hash",
                  "files": {name: digest_bytes(value) for name, value in training.items()}}],
        "analysis_data": {"sha256": "data-hash"},
        "metric_options": {"all_checkpoints": False},
    }
    target = tmp_path / "results" / "width" / "condition" / "runs" / run_id
    target.mkdir(parents=True)
    for name, value in training.items():
        (target / name).write_bytes(value)
    provenance = {"analysis_data_sha256": "data-hash", "sources": {}}
    archive = tmp_path / "metrics.tar.gz"
    status = make_archive(archive, index, provenance, {"options.json": b"{}", "final.json": b"metric"})
    report = validate_archive(archive, status, index, provenance)
    written = install_archive(archive, report, index, tmp_path, {run_id: target})
    installed = target / "metrics" / fingerprint(index["metric_options"]) / "final.json"
    assert written == 2 and installed.read_bytes() == b"metric"
    assert install_archive(archive, report, index, tmp_path, {run_id: target}) == 0


def test_archive_status_mismatch_is_rejected(tmp_path):
    run_id = "b" * 16
    index = {
        "runs": [{"run_id": run_id, "archive_sha256": "input-hash",
                  "files": {"step_0000000.pt": "x", "final.pt": "y"}}],
        "analysis_data": {"sha256": "data-hash"},
        "metric_options": {"all_checkpoints": False},
    }
    provenance = {"analysis_data_sha256": "data-hash"}
    archive = tmp_path / "metrics.tar.gz"
    status = make_archive(archive, index, provenance, {"final.json": b"metric"})
    status["analysis_data_sha256"] = "wrong"
    import pytest
    with pytest.raises(ValueError, match="disagreement"):
        validate_archive(archive, status, index, provenance)
