"""Source-exact provenance helpers for frozen hosted CIFAR collections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.remote_cifar import cell_id
from scripts.run_cifar_remote_queue import calibration_cells, production_cells
from src.training.checkpoints import atomic_json


COMMIT = re.compile(r"[0-9a-f]{40}")
CELL = re.compile(r"[0-9a-f]{16}")
TAG = re.compile(r"[0-9A-Za-z._-]+")
EXPECTED = {"calibration": 49, "production": 35}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_cell_ids(repo: str | Path, dataset: str, stage: str) -> set[str]:
    repo = Path(repo)
    if stage == "calibration":
        cells = calibration_cells(dataset)
    elif stage == "production":
        frozen = json.loads((repo / "results" / dataset.lower() / "gamma_to_lr.json").read_text())
        cells = production_cells(dataset, frozen)
    else:
        raise ValueError(f"Unknown CIFAR stage: {stage}")
    return {cell_id(cell) for cell in cells}


def _inside(repo: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(repo.resolve()):
        raise ValueError(f"{label} must be inside the repository")
    return resolved


def validate_component_report(repo: str | Path, dataset: str, stage: str,
                              report_path: str | Path) -> dict:
    """Validate one complete, immutable-source collection over a cell subset."""
    repo = Path(repo).resolve()
    report_path = _inside(repo, Path(report_path), "CIFAR collection report")
    report = json.loads(report_path.read_text())
    specification = repo / "configs/completion/cifar_sources_v3.json"
    complete = report.get("complete", {})
    identifiers = set(complete) if isinstance(complete, dict) else set()
    if (report.get("schema") != "feature-topology.remote-cifar-collection.v1"
            or not COMMIT.fullmatch(str(report.get("source_commit", "")))
            or not TAG.fullmatch(str(report.get("result_tag", "")))
            or report.get("source_specification") != "configs/completion/cifar_sources_v3.json"
            or report.get("source_specification_sha256") != sha256(specification)
            or report.get("expected") != len(identifiers)
            or report.get("missing") != [] or report.get("invalid") != []
            or not identifiers or any(not CELL.fullmatch(value) for value in identifiers)
            or any(not isinstance(value, str) or not value.endswith(".tar.gz")
                   for value in complete.values())):
        raise ValueError(f"Incomplete hosted CIFAR component: {dataset}/{stage}")
    if not identifiers <= expected_cell_ids(repo, dataset, stage):
        raise ValueError(f"Unexpected hosted CIFAR cells: {dataset}/{stage}")
    return report


def merge_collection_reports(repo: str | Path, dataset: str, stage: str,
                             sources: list[str | Path], output: str | Path) -> dict:
    """Merge disjoint source-bound reports without erasing cohort provenance."""
    repo = Path(repo).resolve()
    output = _inside(repo, Path(output), "Merged CIFAR collection report")
    expected = expected_cell_ids(repo, dataset, stage)
    complete: dict[str, dict] = {}
    components = []
    for source in sources:
        path = _inside(repo, Path(source), "CIFAR component report")
        report = validate_component_report(repo, dataset, stage, path)
        overlap = set(complete) & set(report["complete"])
        if overlap:
            raise ValueError(f"Overlapping hosted CIFAR component cells: {sorted(overlap)}")
        relative = path.relative_to(repo).as_posix()
        component_hash = sha256(path)
        identifiers = sorted(report["complete"])
        components.append({
            "path": relative,
            "sha256": component_hash,
            "result_tag": report["result_tag"],
            "source_commit": report["source_commit"],
            "cell_ids": identifiers,
        })
        for identifier, archive in report["complete"].items():
            complete[identifier] = {
                "archive": archive,
                "component_report": relative,
                "component_report_sha256": component_hash,
                "result_tag": report["result_tag"],
                "source_commit": report["source_commit"],
            }
    if set(complete) != expected:
        raise ValueError(
            f"Merged hosted CIFAR coverage mismatch: {dataset}/{stage} "
            f"{len(complete)}/{len(expected)}"
        )
    specification = repo / "configs/completion/cifar_sources_v3.json"
    merged = {
        "schema": "feature-topology.remote-cifar-collection.v2",
        "dataset": dataset,
        "stage": stage,
        "source_specification": "configs/completion/cifar_sources_v3.json",
        "source_specification_sha256": sha256(specification),
        "expected": len(expected),
        "complete": dict(sorted(complete.items())),
        "components": sorted(components, key=lambda row: row["path"]),
        "missing": [],
        "invalid": [],
    }
    if output.exists() and json.loads(output.read_text()) != merged:
        raise ValueError(f"Refusing to replace conflicting merged collection: {output}")
    atomic_json(output, merged)
    return merged


def validate_collection_report(repo: str | Path, dataset: str, stage: str,
                               report_path: str | Path,
                               bound_commits: set[str] | None = None) -> tuple[dict, list[Path]]:
    """Validate a full v1 or provenance-preserving v2 collection report."""
    repo = Path(repo).resolve()
    path = _inside(repo, Path(report_path), "CIFAR collection report")
    report = json.loads(path.read_text())
    expected = expected_cell_ids(repo, dataset, stage)
    if report.get("schema") == "feature-topology.remote-cifar-collection.v1":
        report = validate_component_report(repo, dataset, stage, path)
        if set(report["complete"]) != expected:
            raise ValueError(f"Hosted CIFAR collection incomplete: {dataset}/{stage}")
        if bound_commits is not None and report["source_commit"] not in bound_commits:
            raise ValueError(f"Unbound hosted CIFAR source commit: {dataset}/{stage}")
        return report, [path]
    specification = repo / "configs/completion/cifar_sources_v3.json"
    complete = report.get("complete", {})
    if (report.get("schema") != "feature-topology.remote-cifar-collection.v2"
            or report.get("dataset") != dataset or report.get("stage") != stage
            or report.get("source_specification") != "configs/completion/cifar_sources_v3.json"
            or report.get("source_specification_sha256") != sha256(specification)
            or report.get("expected") != len(expected) or set(complete) != expected
            or report.get("missing") != [] or report.get("invalid") != []
            or not isinstance(report.get("components"), list)
            or not report["components"]):
        raise ValueError(f"Hosted CIFAR collection incomplete: {dataset}/{stage}")
    paths = [path]
    reconstructed: dict[str, dict] = {}
    for component in report["components"]:
        component_path = _inside(repo, repo / str(component.get("path", "")),
                                 "CIFAR component report")
        if sha256(component_path) != component.get("sha256"):
            raise ValueError(f"Hosted CIFAR component hash mismatch: {component_path}")
        source = validate_component_report(repo, dataset, stage, component_path)
        identifiers = sorted(source["complete"])
        if (component.get("cell_ids") != identifiers
                or component.get("result_tag") != source["result_tag"]
                or component.get("source_commit") != source["source_commit"]
                or (bound_commits is not None
                    and source["source_commit"] not in bound_commits)):
            raise ValueError(f"Hosted CIFAR component provenance mismatch: {component_path}")
        for identifier, archive in source["complete"].items():
            if identifier in reconstructed:
                raise ValueError(f"Duplicate hosted CIFAR cell provenance: {identifier}")
            reconstructed[identifier] = {
                "archive": archive,
                "component_report": component_path.relative_to(repo).as_posix(),
                "component_report_sha256": component["sha256"],
                "result_tag": source["result_tag"],
                "source_commit": source["source_commit"],
            }
        paths.append(component_path)
    if reconstructed != complete:
        raise ValueError(f"Hosted CIFAR merged provenance mismatch: {dataset}/{stage}")
    return report, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], required=True)
    parser.add_argument("--stage", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sources = [Path(value) if Path(value).is_absolute() else repo / value
               for value in args.source]
    output = Path(args.output)
    if not output.is_absolute():
        output = repo / output
    report = merge_collection_reports(repo, args.dataset, args.stage, sources, output)
    print(json.dumps({"dataset": args.dataset, "stage": args.stage,
                      "expected": report["expected"],
                      "components": len(report["components"]),
                      "output": output.relative_to(repo).as_posix()}))


if __name__ == "__main__":
    main()
