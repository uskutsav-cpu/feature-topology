"""Promote a complete hosted-CIFAR controller ledger to its canonical v3 name.

Controllers use tag-specific state paths so a restarted hosted campaign cannot
silently inherit an unrelated source commit.  The results freeze, however,
consumes one canonical ledger per dataset.  This module bridges those two roles
only after independently validating the completed collection reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cifar_provenance import validate_collection_report
from scripts.freeze_results import sha256
from src.training.checkpoints import atomic_json


EXPECTED = {"calibration": 49, "production": 35}


def validate_complete_ledger(repo: Path, dataset: str, source: Path) -> dict:
    """Return a complete, source-bound queue ledger or fail without writing."""
    repo = repo.resolve()
    source = source.resolve()
    if not source.is_relative_to(repo):
        raise ValueError("CIFAR queue ledger must be inside the repository")
    state = json.loads(source.read_text())
    commit = state.get("source_commit")
    if (state.get("schema") != "feature-topology.remote-cifar-queue.v1"
            or not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("Invalid hosted CIFAR queue ledger")
    for attempt in state.get("attempts", []):
        if (attempt.get("dataset") != dataset
                or attempt.get("stage") not in EXPECTED
                or not str(attempt.get("workflow_run", "")).isdigit()
                or not str(attempt.get("url", "")).startswith("https://github.com/")
                or not attempt.get("cell_ids")
                or len(attempt["cell_ids"]) > 64
                or any(not re.fullmatch(r"[0-9a-f]{16}", value)
                       for value in attempt["cell_ids"])
                or "started" not in attempt or "finished" not in attempt
                or not attempt.get("conclusion")):
            raise ValueError("Incomplete hosted CIFAR attempt provenance")
    stages = state.get(dataset, {})
    key = dataset.lower()
    for stage, expected in EXPECTED.items():
        row = stages.get(stage, {})
        if (row.get("expected") != expected or row.get("complete") != expected
                or row.get("missing") != 0 or row.get("invalid") != []):
            raise ValueError(f"Hosted CIFAR stage incomplete: {dataset}/{stage}")
        report_path = (repo / "results/completion/remote_collections"
                       / f"{key}_{stage}.json")
        report, _ = validate_collection_report(repo, dataset, stage, report_path, {commit})
        if report.get("expected") != expected:
            raise ValueError(f"Hosted CIFAR collection incomplete: {dataset}/{stage}")
    return state


def promote(repo: Path, dataset: str, source: Path, output: Path) -> dict:
    state = validate_complete_ledger(repo, dataset, source)
    output = output.resolve()
    if not output.is_relative_to(repo.resolve()):
        raise ValueError("Canonical CIFAR queue ledger must be inside the repository")
    if output.exists() and json.loads(output.read_text()) != state:
        raise ValueError(f"Refusing to replace conflicting canonical ledger: {output}")
    atomic_json(output, state)
    return state


def _validate_source_ledger(repo: Path, dataset: str, source: Path) -> dict:
    source = source.resolve()
    if not source.is_relative_to(repo.resolve()):
        raise ValueError("CIFAR source ledger must be inside the repository")
    state = json.loads(source.read_text())
    commit = state.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid hosted CIFAR source commit")
    if state.get("schema") == "feature-topology.remote-cifar-queue.v1":
        attempts = state.get("attempts", [])
    elif state.get("schema") == "feature-topology.remote-cifar-correction.v1":
        if (state.get("dataset") != dataset
                or state.get("stage") not in EXPECTED
                or not re.fullmatch(r"[0-9A-Za-z._-]+", str(state.get("result_tag", "")))
                or not str(state.get("workflow_run", "")).isdigit()
                or not str(state.get("url", "")).startswith("https://github.com/")
                or not state.get("cell_ids")
                or state.get("expected_cells") != len(state["cell_ids"])
                or any(not re.fullmatch(r"[0-9a-f]{16}", value)
                       for value in state["cell_ids"])
                or "started" not in state or "finished" not in state
                or not state.get("conclusion")):
            raise ValueError("Incomplete hosted CIFAR correction provenance")
        attempts = [{key: state[key] for key in [
            "dataset", "stage", "workflow_run", "url", "cell_ids",
            "started", "finished", "conclusion"]}]
    else:
        raise ValueError("Invalid hosted CIFAR source ledger")
    for attempt in attempts:
        if (attempt.get("dataset") != dataset
                or attempt.get("stage") not in EXPECTED
                or not str(attempt.get("workflow_run", "")).isdigit()
                or not str(attempt.get("url", "")).startswith("https://github.com/")
                or not attempt.get("cell_ids") or len(attempt["cell_ids"]) > 64
                or any(not re.fullmatch(r"[0-9a-f]{16}", value)
                       for value in attempt["cell_ids"])
                or "started" not in attempt or "finished" not in attempt
                or not attempt.get("conclusion")):
            raise ValueError("Incomplete hosted CIFAR attempt provenance")
    return {
        "path": source.relative_to(repo).as_posix(),
        "sha256": sha256(source),
        "schema": state["schema"],
        "source_commit": commit,
        "workflow_runs": [str(attempt["workflow_run"]) for attempt in attempts],
    }


def promote_multisource(repo: Path, dataset: str, sources: list[Path],
                        collections: dict[str, Path], output: Path) -> dict:
    """Promote exact multi-commit provenance without flattening source identity."""
    repo = repo.resolve()
    bound = [_validate_source_ledger(repo, dataset, path) for path in sources]
    commits = {row["source_commit"] for row in bound}
    stages = {}
    used_commits = set()
    for stage, expected in EXPECTED.items():
        path = collections[stage].resolve()
        report, _ = validate_collection_report(repo, dataset, stage, path, commits)
        if report["schema"] == "feature-topology.remote-cifar-collection.v1":
            used_commits.add(report["source_commit"])
        else:
            used_commits.update(row["source_commit"] for row in report["components"])
        if report.get("expected") != expected:
            raise ValueError(f"Hosted CIFAR stage incomplete: {dataset}/{stage}")
        stages[stage] = {
            "expected": expected,
            "complete": len(report["complete"]),
            "missing": 0,
            "invalid": [],
            "collection_report": path.relative_to(repo).as_posix(),
            "collection_report_sha256": sha256(path),
        }
    if used_commits != commits:
        raise ValueError("Canonical CIFAR sources and collection cohorts differ")
    state = {
        "schema": "feature-topology.remote-cifar-queue.v2",
        "dataset": dataset,
        "sources": sorted(bound, key=lambda row: row["path"]),
        "stages": stages,
    }
    output = output.resolve()
    if not output.is_relative_to(repo):
        raise ValueError("Canonical CIFAR queue ledger must be inside the repository")
    if output.exists() and json.loads(output.read_text()) != state:
        raise ValueError(f"Refusing to replace conflicting canonical ledger: {output}")
    atomic_json(output, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], required=True)
    parser.add_argument("--source", required=True, action="append")
    parser.add_argument("--collection", action="append", default=[],
                        help="For multi-source promotion: STAGE=REPORT")
    parser.add_argument("--output")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sources = []
    for value in args.source:
        source = Path(value)
        sources.append(source if source.is_absolute() else repo / source)
    output = (Path(args.output) if args.output else
              Path("results/completion") / f"{args.dataset.lower()}_remote_queue_v3.json")
    if not output.is_absolute():
        output = repo / output
    if len(sources) == 1 and not args.collection:
        state = promote(repo, args.dataset, sources[0], output)
        summary = {"dataset": args.dataset, "source_commit": state["source_commit"]}
    else:
        collections = {}
        for value in args.collection:
            stage, separator, path = value.partition("=")
            if not separator or stage not in EXPECTED:
                raise ValueError("Collections must be CALIBRATION=PATH or PRODUCTION=PATH")
            candidate = Path(path)
            collections[stage] = candidate if candidate.is_absolute() else repo / candidate
        if set(collections) != set(EXPECTED):
            raise ValueError("Multi-source promotion requires both stage collection reports")
        state = promote_multisource(repo, args.dataset, sources, collections, output)
        summary = {"dataset": args.dataset,
                   "source_commits": [row["source_commit"] for row in state["sources"]]}
    summary["canonical_ledger"] = output.relative_to(repo).as_posix()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
