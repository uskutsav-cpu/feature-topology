"""Checksum-bound, resumable hosted workers for the frozen CIFAR-v3 study.

This module changes execution placement only.  It imports the original training
and analysis functions and reconstructs their exact frozen configurations.
Completed archives are additive and immutable; incompatible resume artifacts
are ignored, while malformed compatible artifacts fail closed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


GAMMAS = (0.125, 0.5, 1.0, 4.0, 16.0, 64.0, 128.0)
MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BASE = {"batch_size": 128, "target_loss": 0.2, "eval_every": 500}


def fingerprint(value: dict) -> str:
    """Match the historical run identifier without importing PyTorch."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(path: str | Path = "configs/completion/cifar_sources_v3.json") -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "feature-topology.cifar-sources.v1":
        raise ValueError("Unexpected CIFAR source specification")
    return value


def parse_cell(raw: str | dict) -> dict:
    cell = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if cell.get("stage") not in {"calibration", "production"}:
        raise ValueError("CIFAR cell stage must be calibration or production")
    if cell.get("dataset") not in {"CIFAR10", "CIFAR100"}:
        raise ValueError("Unknown CIFAR dataset")
    gamma = cell.get("gamma")
    if (not isinstance(gamma, (int, float)) or isinstance(gamma, bool)
            or float(gamma) not in GAMMAS):
        raise ValueError("CIFAR cell has an out-of-design gamma")
    normalized = {"stage": cell["stage"], "dataset": cell["dataset"],
                  "gamma": float(gamma)}
    if cell["stage"] == "calibration":
        multiplier = cell.get("multiplier")
        if (not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool)
                or float(multiplier) not in MULTIPLIERS or set(cell) != {
                    "stage", "dataset", "gamma", "multiplier"}):
            raise ValueError("Malformed CIFAR calibration cell")
        normalized["multiplier"] = float(multiplier)
    else:
        seed, lr = cell.get("seed"), cell.get("lr")
        if (not isinstance(seed, int) or isinstance(seed, bool) or seed not in range(5)
                or not isinstance(lr, (int, float)) or isinstance(lr, bool)
                or not math.isfinite(lr) or lr <= 0 or set(cell) != {
                    "stage", "dataset", "gamma", "seed", "lr"}):
            raise ValueError("Malformed CIFAR production cell")
        normalized.update(seed=seed, lr=float(lr))
    return normalized


def cell_id(cell: dict) -> str:
    return fingerprint(parse_cell(cell))


def config_for(cell: dict) -> dict:
    cell = parse_cell(cell)
    classes = 10 if cell["dataset"] == "CIFAR10" else 100
    shared = dict(**BASE, classes=classes, dataset=cell["dataset"], gamma=cell["gamma"])
    if cell["stage"] == "calibration":
        gamma = cell["gamma"]
        center = 0.1 * (gamma**2 if gamma <= 1 else gamma**0.5)
        return dict(**shared, lr=center*cell["multiplier"], seed=900, max_steps=2000)
    return dict(**shared, seed=cell["seed"], lr=cell["lr"], max_steps=35200)


def run_directory(work: str | Path, cell: dict) -> Path:
    cell = parse_cell(cell)
    stage = "calibration" if cell["stage"] == "calibration" else "runs"
    return Path(work)/cell["dataset"].lower()/stage/fingerprint(config_for(cell))


def verify_dataset(data_root: str | Path, dataset: str, specification: dict) -> tuple[Path, str]:
    row = specification["datasets"][dataset]
    archive = Path(data_root)/row["archive"]
    if (not archive.is_file() or archive.stat().st_size != row["bytes"]
            or sha256(archive) != row["sha256"]):
        raise ValueError(f"CIFAR source archive mismatch: {archive}")
    return archive, row["sha256"]


def _safe_member(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or ".." in value.parts
            or "." in value.parts or "\\" in name):
        raise ValueError(f"Unsafe CIFAR archive member: {name}")
    return value


def restore_archives(resume: str | Path, work: str | Path, cell: dict,
                     dataset_sha256: str) -> int:
    """Restore the newest compatible cumulative archive for this cell."""
    resume, work = Path(resume), Path(work)
    expected_config = config_for(cell)
    candidates = []
    for archive in sorted(resume.glob(f"cifar-{cell_id(cell)}-*.tar.gz"), reverse=True):
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = {member.name for member in members}
            if len(names) != len(members) or "job_manifest.json" not in names:
                raise ValueError(f"Malformed CIFAR resume archive: {archive}")
            report = json.load(tar.extractfile("job_manifest.json"))
            if (report.get("schema") != "feature-topology.remote-cifar.v1"
                    or report.get("cell") != parse_cell(cell)
                    or report.get("config") != expected_config
                    or report.get("dataset_sha256") != dataset_sha256):
                continue
            if names != {*report.get("files", {}), "job_manifest.json"}:
                raise ValueError(f"CIFAR resume membership mismatch: {archive}")
            candidates.append((archive, report))
            break
    if not candidates:
        return 0
    archive, report = candidates[0]
    restored = 0
    with tarfile.open(archive, "r:gz") as tar:
        for name, expected in report["files"].items():
            relative = _safe_member(name)
            member = tar.getmember(name)
            if not member.isfile():
                raise ValueError("Only regular CIFAR resume files are allowed")
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"CIFAR resume hash mismatch: {name}")
            target = work.joinpath(*relative.parts)
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"Conflicting CIFAR resume artifact: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            restored += 1
    return restored


def compute(work: str | Path, data_root: str | Path, resume: str | Path,
            cell: dict, device: str, source_specification: str | Path) -> dict:
    # The workflow plan intentionally uses only the standard library.  Import
    # scientific dependencies after the compute job has installed the lockfile.
    from scripts.run_cifar import analyze, load_data, train
    cell = parse_cell(cell)
    specification = sources(source_specification)
    data = load_data(cell["dataset"], str(data_root), download=True)
    _, dataset_sha = verify_dataset(data_root, cell["dataset"], specification)
    restored = restore_archives(resume, work, cell, dataset_sha)
    config = config_for(cell)
    root = Path(work)/cell["dataset"].lower()
    stage_root = root/("calibration" if cell["stage"] == "calibration" else "runs")
    result = train(config, stage_root, data, device)
    if cell["stage"] == "production" and result["status"] != "diverged":
        analyze(config, stage_root, data, device)
    return {"cell_id": cell_id(cell), "run_id": fingerprint(config),
            "status": result["status"], "restored_files": restored}


def _complete(directory: Path, cell: dict) -> bool:
    summary_path = directory/"summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    config = config_for(cell)
    if (summary.get("config") != config or summary.get("run_id") != fingerprint(config)
            or summary.get("status") not in {"converged", "budget_exhausted", "diverged"}):
        raise ValueError("Invalid hosted CIFAR summary")
    if summary["status"] == "diverged":
        return True
    required = {"final.pt"} if cell["stage"] == "calibration" else {
        "final.pt", "metrics.json", "representations.npz"}
    return all((directory/name).is_file() for name in required)


def pack(work: str | Path, data_root: str | Path, output: str | Path, cell: dict,
         job_id: str, source_specification: str | Path) -> dict:
    cell = parse_cell(cell)
    if not re.fullmatch(r"[0-9A-Za-z_-]+", job_id):
        raise ValueError("Unsafe hosted CIFAR job ID")
    specification = sources(source_specification)
    _, dataset_sha = verify_dataset(data_root, cell["dataset"], specification)
    directory = run_directory(work, cell)
    paths = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.exists() else []
    files = {path.relative_to(Path(work)).as_posix(): sha256(path) for path in paths}
    report = {
        "schema": "feature-topology.remote-cifar.v1", "cell": cell,
        "cell_id": cell_id(cell), "config": config_for(cell),
        "run_id": fingerprint(config_for(cell)), "complete": _complete(directory, cell),
        "dataset_sha256": dataset_sha,
        "source_specification_sha256": sha256(source_specification),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version(), "commit": os.environ.get("GITHUB_SHA"),
                 "workflow_run": os.environ.get("GITHUB_RUN_ID")},
        "files": files,
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    archive = output/f"cifar-{cell_id(cell)}-{job_id}.tar.gz"
    with archive.open("wb") as stream, gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for path in paths:
                relative = path.relative_to(Path(work)).as_posix()
                info = tarfile.TarInfo(relative); info.size = path.stat().st_size; info.mode = 0o644
                with path.open("rb") as source:
                    tar.addfile(info, source)
            payload = (json.dumps(report, sort_keys=True, indent=2)+"\n").encode()
            info = tarfile.TarInfo("job_manifest.json"); info.size = len(payload); info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
    report["archive_sha256"] = sha256(archive)
    status = output/f"status-{cell_id(cell)}-{job_id}.json"
    status.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    return report


def plan(raw_cells: str) -> list[dict]:
    values = json.loads(raw_cells)
    if not isinstance(values, list) or not values or len(values) > 64:
        raise ValueError("Hosted CIFAR batches require 1--64 cells")
    cells = [parse_cell(value) for value in values]
    ids = [cell_id(value) for value in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Hosted CIFAR cells must be unique")
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "compute", "pack"])
    parser.add_argument("--cells")
    parser.add_argument("--cell")
    parser.add_argument("--work", default="cifar-work")
    parser.add_argument("--data", default="data/cifar")
    parser.add_argument("--resume", default="resume")
    parser.add_argument("--output", default="outgoing")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--job-id", default="local")
    parser.add_argument("--source-specification", default="configs/completion/cifar_sources_v3.json")
    args = parser.parse_args()
    if args.command == "plan":
        cells = plan(args.cells)
        line = "cells="+json.dumps(cells, separators=(",", ":"))+"\n"
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
                stream.write(line)
        print(line, end="")
    else:
        cell = parse_cell(args.cell)
        result = (compute(args.work, args.data, args.resume, cell, args.device,
                          args.source_specification) if args.command == "compute" else
                  pack(args.work, args.data, args.output, cell, args.job_id,
                       args.source_specification))
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
