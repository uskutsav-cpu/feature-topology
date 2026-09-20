"""Download, verify, and install checksum-bound hosted metric results.

The hosted worker uses a normalized ``main/runs/<id>`` workspace.  This
collector maps verified outputs back to the one canonical local run selected by
the completion inventory.  Existing scientific files are immutable: identical
files are reused and conflicts fail closed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import inventory
from scripts.compute_metrics import metric_provenance
from scripts.remote_metrics import required_checkpoints
from src.training.checkpoints import atomic_json, fingerprint


TAG = re.compile(r"[0-9A-Za-z._-]+")
STATUS = re.compile(r"status-([0-9a-f]{16})-(.+)\.json")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(tag, cache, patterns, allow_empty=False):
    if not TAG.fullmatch(tag):
        raise ValueError(f"Unsafe release tag: {tag}")
    cache.mkdir(parents=True, exist_ok=True)
    command = ["gh", "release", "download", tag, "--dir", str(cache), "--skip-existing"]
    for pattern in patterns:
        command.extend(["--pattern", pattern])
    result = subprocess.run(command, text=True, capture_output=True)
    empty_message = result.stderr.lower()+result.stdout.lower()
    expected_empty = "no assets match" in empty_message or "no assets to download" in empty_message
    if result.returncode and not (allow_empty and expected_empty):
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def _record(index, run_id):
    matches = [row for row in index["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one input record for {run_id}, found {len(matches)}")
    return matches[0]


def _protocol_complete(report, index, record):
    expected_checkpoints = sorted(required_checkpoints(index, record))
    completed = sorted(report.get("completed", []))
    all_training_checkpoints = sorted(
        name[:-3] for name in record["files"] if name.endswith(".pt")
    )
    current_complete = (
        report.get("complete") is True
        and completed == expected_checkpoints
        and sorted(report.get("expected", [])) == expected_checkpoints
    )
    legacy_endpoint_complete = (
        report.get("complete") is False
        and not index["metric_options"]["all_checkpoints"]
        and completed == expected_checkpoints
        and sorted(report.get("expected", [])) == all_training_checkpoints
    )
    return current_complete or legacy_endpoint_complete


def validate_archive(archive, status, index, expected_provenance):
    """Validate a complete result archive without changing local results."""
    archive = Path(archive)
    run_id = status["run_id"]
    record = _record(index, run_id)
    if sha256(archive) != status.get("archive_sha256"):
        raise ValueError(f"Result archive checksum mismatch: {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = {member.name for member in members}
        if len(names) != len(members) or "job_manifest.json" not in names:
            raise ValueError(f"Malformed result archive: {archive.name}")
        report = json.load(tar.extractfile("job_manifest.json"))
        for field in ["run_id", "job_id", "complete", "completed", "expected",
                      "input_archive_sha256", "analysis_data_sha256",
                      "metric_provenance", "files"]:
            if report.get(field) != status.get(field):
                raise ValueError(f"Status/archive disagreement for {field}: {archive.name}")
        if (report["run_id"] != run_id
                or report["input_archive_sha256"] != record["archive_sha256"]
                or report["analysis_data_sha256"] != index["analysis_data"]["sha256"]
                or report["metric_provenance"] != expected_provenance):
            raise ValueError(f"Result provenance mismatch: {archive.name}")
        # Releases produced before the endpoint-schedule bookkeeping fix
        # recorded every training checkpoint in ``expected`` and set
        # ``complete=false``, despite having both frozen endpoint measurements.
        # Accept exactly that auditable legacy shape; any genuinely missing
        # protocol checkpoint still fails closed.
        if not _protocol_complete(report, index, record):
            raise ValueError(f"Incomplete checkpoint coverage: {archive.name}")
        if names != {*report["files"], "job_manifest.json"}:
            raise ValueError(f"Result archive membership mismatch: {archive.name}")
        for member in members:
            if member.name == "job_manifest.json":
                continue
            relative = Path(member.name)
            allowed = (relative.parts[:3] == ("main", "runs", run_id)
                       or relative.parts[:1] == ("ph_cache",))
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts or not allowed):
                raise ValueError(f"Unsafe result member: {member.name}")
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != report["files"][member.name]:
                raise ValueError(f"Result member checksum mismatch: {member.name}")
    return report


def _canonical_runs(repo):
    report = inventory(repo)
    if report["failures"]:
        raise ValueError(f"Training inventory is invalid: {report['failures']}")
    result = {}
    for row in report["validated_runs"]:
        if row["run_id"] in result:
            raise ValueError(f"Multiple canonical training runs found for {row['run_id']}")
        result[row["run_id"]] = Path(row["path"])
    return result


def _verify_training_target(target, record):
    for name, expected in record["files"].items():
        path = target / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Canonical training input changed or is missing: {path}")


def _write_immutable(target, payload, expected):
    if target.exists():
        if sha256(target) == expected or target.name == "host.json":
            return False
        raise ValueError(f"Conflicting existing metric artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Refusing corrupt metric artifact: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def install_archive(archive, report, index, repo, canonical):
    """Install a previously validated archive into immutable local locations."""
    run_id = report["run_id"]
    record = _record(index, run_id)
    target_run = canonical.get(run_id)
    if target_run is None:
        raise ValueError(f"No canonical local training run for {run_id}")
    _verify_training_target(target_run, record)
    profile = fingerprint(index["metric_options"])
    prefix = Path("main") / "runs" / run_id / "metrics" / profile
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for name, expected in report["files"].items():
            source = Path(name)
            if source.parts[:len(prefix.parts)] == prefix.parts:
                target = target_run / "metrics" / profile / Path(*source.parts[len(prefix.parts):])
            elif source.parts[:1] == ("ph_cache",):
                target = Path(repo) / "results" / source
            else:
                raise ValueError(f"Result member belongs to a different profile: {name}")
            payload = tar.extractfile(name).read()
            written += int(_write_immutable(target, payload, expected))
    return written


def collect(repo, index_path, input_tag, result_tag, cache, install=True, excluded=()):
    repo = Path(repo).resolve()
    index = json.loads(Path(index_path).read_text())
    cache = Path(cache)
    input_cache = cache / input_tag
    result_cache = cache / result_tag
    _download(input_tag, input_cache, [index["analysis_data"]["asset"]])
    analysis_data = input_cache / index["analysis_data"]["asset"]
    if sha256(analysis_data) != index["analysis_data"]["sha256"]:
        raise ValueError("Downloaded analysis archive checksum mismatch")
    _download(result_tag, result_cache, ["status-*.json", "metrics-*.tar.gz"], allow_empty=True)
    provenance = metric_provenance(analysis_data)
    excluded = set(excluded)
    expected = {row["run_id"] for row in index["runs"]} - excluded
    statuses = {}
    invalid = []
    for path in sorted(result_cache.glob("status-*.json")):
        match = STATUS.fullmatch(path.name)
        if not match or match.group(1) not in expected:
            continue
        try:
            status = json.loads(path.read_text())
            if status.get("run_id") != match.group(1):
                raise ValueError("status filename/run ID mismatch")
            statuses.setdefault(match.group(1), []).append((path, status))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            invalid.append({"status": path.name, "error": str(exc)})
    complete, partial, installed = {}, {}, {}
    canonical = _canonical_runs(repo) if install else {}
    for run_id in sorted(expected):
        candidates = statuses.get(run_id, [])
        record = _record(index, run_id)
        completed = [
            (path, row) for path, row in candidates
            if _protocol_complete(row, index, record)
        ]
        if not completed:
            if candidates:
                partial[run_id] = [path.name for path, _ in candidates]
            continue
        path, status = sorted(completed, key=lambda item: item[0].name)[-1]
        archive = result_cache / f"metrics-{run_id}-{status['job_id']}.tar.gz"
        try:
            report = validate_archive(archive, status, index, provenance)
            complete[run_id] = archive.name
            if install:
                installed[run_id] = install_archive(archive, report, index, repo, canonical)
        except (ValueError, KeyError, OSError, tarfile.TarError) as exc:
            invalid.append({"status": path.name, "archive": archive.name, "error": str(exc)})
    missing = sorted(expected - set(complete))
    return {
        "schema": "feature-topology.remote-collection.v1",
        "index": str(Path(index_path)),
        "input_tag": input_tag,
        "result_tag": result_tag,
        "expected": len(expected),
        "excluded_shared_profiles": sorted(excluded),
        "complete": complete,
        "installed_files": installed,
        "partial": partial,
        "missing": missing,
        "invalid": invalid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--index", required=True)
    parser.add_argument("--input-tag", required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--cache", default="results/completion/remote_metric_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-index", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    excluded = set()
    for path in args.exclude_index:
        excluded.update(row["run_id"] for row in json.loads(Path(path).read_text())["runs"])
    result = collect(args.repo, args.index, args.input_tag, args.result_tag, args.cache,
                     install=not args.check_only, excluded=excluded)
    atomic_json(Path(args.output), result)
    print(json.dumps({key: (len(value) if isinstance(value, (dict, list)) else value)
                      for key, value in result.items() if key not in {"schema", "index"}}, indent=2))
    raise SystemExit(bool(result["invalid"]))
