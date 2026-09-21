"""Validate and additively install hosted frozen-v3 CIFAR cell archives."""
from __future__ import annotations

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
from scripts.remote_cifar import cell_id, config_for, parse_cell, sha256, sources
from src.training.checkpoints import atomic_json


STATUS = re.compile(r"status-([0-9a-f]{16})-([0-9A-Za-z_-]+)\.json")
TAG = re.compile(r"[0-9A-Za-z._-]+")


def download(tag: str, cache: str | Path) -> None:
    if not TAG.fullmatch(tag):
        raise ValueError("Unsafe CIFAR result tag")
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    command = ["gh", "release", "download", tag, "--dir", str(cache), "--skip-existing",
               "--pattern", "status-*.json", "--pattern", "cifar-*.tar.gz"]
    result = subprocess.run(command, text=True, capture_output=True)
    message = (result.stdout+result.stderr).lower()
    if result.returncode and "no assets" not in message:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def download_statuses(tag: str, cache: str | Path) -> None:
    """Retain small status manifests without bulk-downloading every archive."""
    if not TAG.fullmatch(tag):
        raise ValueError("Unsafe CIFAR result tag")
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    command = ["gh", "release", "download", tag, "--dir", str(cache),
               "--skip-existing", "--pattern", "status-*.json"]
    result = subprocess.run(command, text=True, capture_output=True)
    message = (result.stdout+result.stderr).lower()
    if result.returncode and "no assets" not in message:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def download_asset(tag: str, name: str, destination: str | Path) -> Path:
    """Download one exact release asset into a temporary transport directory."""
    if not TAG.fullmatch(tag) or Path(name).name != name or not name.endswith(".tar.gz"):
        raise ValueError("Unsafe CIFAR release asset")
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    command = ["gh", "release", "download", tag, "--dir", str(destination),
               "--pattern", name]
    subprocess.run(command, check=True, text=True, capture_output=True)
    target = destination/name
    if not target.is_file():
        raise FileNotFoundError(f"Downloaded CIFAR asset is missing: {name}")
    return target


def _equal_manifest(status: dict, report: dict) -> bool:
    return all(status.get(key) == report.get(key) for key in [
        "schema", "cell", "cell_id", "config", "run_id", "complete",
        "dataset_sha256", "source_specification_sha256", "host", "files"])


def validate_archive(archive: str | Path, status: dict, cell: dict,
                     source_specification: str | Path,
                     expected_commit: str | None = None) -> dict:
    archive = Path(archive); cell = parse_cell(cell)
    specification = sources(source_specification)
    expected_source = specification["datasets"][cell["dataset"]]["sha256"]
    if sha256(archive) != status.get("archive_sha256"):
        raise ValueError("Hosted CIFAR archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers(); names = {member.name for member in members}
        if len(names) != len(members) or "job_manifest.json" not in names:
            raise ValueError("Malformed hosted CIFAR archive")
        report = json.load(tar.extractfile("job_manifest.json"))
        if not _equal_manifest(status, report):
            raise ValueError("Hosted CIFAR status/archive disagreement")
        if (report.get("schema") != "feature-topology.remote-cifar.v1"
                or report.get("cell") != cell or report.get("cell_id") != cell_id(cell)
                or report.get("config") != config_for(cell)
                or report.get("dataset_sha256") != expected_source
                or report.get("source_specification_sha256") != sha256(source_specification)
                or (expected_commit is not None
                    and report.get("host", {}).get("commit") != expected_commit)):
            raise ValueError("Hosted CIFAR provenance mismatch")
        if names != {*report.get("files", {}), "job_manifest.json"}:
            raise ValueError("Hosted CIFAR archive membership mismatch")
        run_prefix = Path(cell["dataset"].lower())/(
            "calibration" if cell["stage"] == "calibration" else "runs")/report["run_id"]
        for name, expected in report["files"].items():
            relative = Path(name)
            member = tar.getmember(name)
            if (relative.is_absolute() or ".." in relative.parts or not member.isfile()
                    or relative.parts[:len(run_prefix.parts)] != run_prefix.parts):
                raise ValueError(f"Unsafe hosted CIFAR member: {name}")
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Hosted CIFAR member checksum mismatch: {name}")
        if report.get("complete"):
            required = {run_prefix/"summary.json"}
            if cell["stage"] == "calibration":
                required.add(run_prefix/"final.pt")
            else:
                required.update({run_prefix/"final.pt", run_prefix/"metrics.json",
                                 run_prefix/"representations.npz"})
            summary_name = (run_prefix/"summary.json").as_posix()
            summary = json.load(tar.extractfile(summary_name))
            if (summary.get("status") == "diverged"):
                required = {run_prefix/"summary.json"}
            if (summary.get("config") != config_for(cell)
                    or summary.get("run_id") != report["run_id"]
                    or not {path.as_posix() for path in required} <= set(report["files"])):
                raise ValueError("Hosted CIFAR completion claim is invalid")
    return report


def _write_immutable(target: Path, payload: bytes, expected: str) -> bool:
    if target.exists():
        if target.is_file() and sha256(target) == expected:
            return False
        raise ValueError(f"Conflicting existing CIFAR artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(payload)
    try:
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Refusing corrupt CIFAR artifact: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def install_archive(repo: str | Path, archive: str | Path, report: dict) -> int:
    repo, archive = Path(repo).resolve(), Path(archive)
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for name, expected in report["files"].items():
            # A terminal archive is retained byte-for-byte on its immutable
            # GitHub release.  The optimizer resume is execution state, not a
            # frozen scientific output, and can be very large.  Validate it
            # above but do not duplicate it in the local results tree.
            if report.get("complete") and Path(name).name == "resume.pt":
                continue
            target = repo/"results"/name
            if not target.resolve().is_relative_to((repo/"results").resolve()):
                raise ValueError("CIFAR install path escapes results")
            written += int(_write_immutable(target, tar.extractfile(name).read(), expected))
    return written


def _installed_complete(repo: Path, report: dict) -> bool:
    """Check that a streamed terminal archive was already installed exactly."""
    if not report.get("complete"):
        return False
    for name, expected in report.get("files", {}).items():
        if Path(name).name == "resume.pt":
            continue
        target = repo/"results"/name
        if not target.is_file() or sha256(target) != expected:
            return False
    return True


def _load_transport_ledger(path: Path) -> dict:
    if not path.exists():
        return {"schema": "feature-topology.remote-cifar-transport.v1", "records": {}}
    value = json.loads(path.read_text())
    if value.get("schema") != "feature-topology.remote-cifar-transport.v1" \
            or not isinstance(value.get("records"), dict):
        raise ValueError(f"Invalid CIFAR transport validation ledger: {path}")
    return value


def _validated_transport(record: dict | None, status_path: Path, status: dict,
                         specification_sha256: str, expected_commit: str | None) -> bool:
    return bool(record
                and record.get("status_sha256") == sha256(status_path)
                and record.get("archive_sha256") == status.get("archive_sha256")
                and record.get("source_specification_sha256") == specification_sha256
                and record.get("source_commit") == expected_commit
                and record.get("cell_id") == status.get("cell_id")
                and record.get("complete") == status.get("complete"))


def collect(repo: str | Path, tag: str, cells: list[dict], cache: str | Path,
            source_specification: str | Path, install: bool = True,
            expected_commit: str | None = None, bounded_cache: bool = False) -> dict:
    repo = Path(repo).resolve(); cache = Path(cache)
    source_specification = Path(source_specification)
    if not source_specification.is_absolute():
        source_specification = repo/source_specification
    specification_sha256 = sha256(source_specification)
    normalized = [parse_cell(cell) for cell in cells]
    expected = {cell_id(cell): cell for cell in normalized}
    if len(expected) != len(normalized):
        raise ValueError("Duplicate requested CIFAR cells")
    if bounded_cache:
        download_statuses(tag, cache)
    else:
        download(tag, cache)
    transport_path = cache/"validated_transport.json"
    transport = _load_transport_ledger(transport_path) if bounded_cache else None
    statuses: dict[str, list[tuple[Path, dict]]] = {}
    invalid = []
    for path in sorted(cache.glob("status-*.json")):
        match = STATUS.fullmatch(path.name)
        if not match or match.group(1) not in expected:
            continue
        try:
            row = json.loads(path.read_text())
            if row.get("cell_id") != match.group(1):
                raise ValueError("CIFAR status filename/cell mismatch")
            statuses.setdefault(match.group(1), []).append((path, row))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"status": path.name, "error": str(exc)})
    complete, partial, installed = {}, {}, {}
    for identifier, cell in expected.items():
        candidates = statuses.get(identifier, [])
        accepted = []
        for path, status in candidates:
            archive_name = f"cifar-{identifier}-{path.name[len('status-'+identifier+'-'):-5]}.tar.gz"
            archive = cache/archive_name
            try:
                record = transport["records"].get(path.name) if transport is not None else None
                reused = (bounded_cache and not archive.exists()
                          and _validated_transport(record, path, status,
                                                   specification_sha256, expected_commit)
                          and (not status.get("complete") or not install
                               or _installed_complete(repo, status)))
                written = 0
                if reused:
                    report = status
                elif bounded_cache and not archive.exists():
                    with tempfile.TemporaryDirectory(prefix="cifar-transfer-",
                                                     dir=cache.parent) as temporary:
                        transfer = download_asset(tag, archive_name, temporary)
                        report = validate_archive(transfer, status, cell, source_specification,
                                                  expected_commit=expected_commit)
                        if report.get("complete") and install:
                            written = install_archive(repo, transfer, report)
                else:
                    report = validate_archive(archive, status, cell, source_specification,
                                              expected_commit=expected_commit)
                    if bounded_cache and report.get("complete") and install:
                        written = install_archive(repo, archive, report)
                if transport is not None and not reused:
                    transport["records"][path.name] = {
                        "status_sha256": sha256(path),
                        "archive_sha256": status.get("archive_sha256"),
                        "source_specification_sha256": specification_sha256,
                        "source_commit": expected_commit,
                        "cell_id": identifier,
                        "complete": bool(report.get("complete")),
                    }
                    atomic_json(transport_path, transport)
                if report["complete"]:
                    accepted.append((path, archive_name, report, written))
                else:
                    partial.setdefault(identifier, []).append(path.name)
            except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
                invalid.append({"status": path.name, "archive": archive.name, "error": str(exc)})
        if accepted:
            _, archive_name, report, streamed_writes = sorted(
                accepted, key=lambda item: item[0].name)[-1]
            complete[identifier] = archive_name
            if install:
                installed[identifier] = (sum(item[3] for item in accepted) if bounded_cache
                                         else install_archive(repo, cache/archive_name, report))
    return {"schema": "feature-topology.remote-cifar-collection.v1", "result_tag": tag,
            "source_commit": expected_commit,
            "source_specification": source_specification.relative_to(repo).as_posix(),
            "source_specification_sha256": specification_sha256,
            "expected": len(expected), "complete": complete, "partial": partial,
            "missing": sorted(set(expected)-set(complete)), "invalid": invalid,
            "installed_files": installed}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    parser.add_argument("--tag",required=True)
    parser.add_argument("--cells",required=True,help="JSON cell list")
    parser.add_argument("--cache",required=True)
    parser.add_argument("--source-specification",default="configs/completion/cifar_sources_v3.json")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output",required=True)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    result=collect(args.repo,args.tag,json.loads(args.cells),args.cache,
                   args.source_specification,install=not args.check_only,
                   expected_commit=args.expected_commit)
    atomic_json(Path(args.output),result)
    print(json.dumps({key:(len(value) if isinstance(value,(list,dict)) else value)
                      for key,value in result.items() if key!="schema"},indent=2))
    return int(bool(result["invalid"]))


if __name__=="__main__":
    raise SystemExit(main())
