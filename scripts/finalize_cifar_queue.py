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
    specification = repo / "configs/completion/cifar_sources_v3.json"
    specification_hash = sha256(specification)
    key = dataset.lower()
    for stage, expected in EXPECTED.items():
        row = stages.get(stage, {})
        if (row.get("expected") != expected or row.get("complete") != expected
                or row.get("missing") != 0 or row.get("invalid") != []):
            raise ValueError(f"Hosted CIFAR stage incomplete: {dataset}/{stage}")
        report_path = (repo / "results/completion/remote_collections"
                       / f"{key}_{stage}.json")
        report = json.loads(report_path.read_text())
        if (report.get("schema") != "feature-topology.remote-cifar-collection.v1"
                or report.get("source_commit") != commit
                or report.get("source_specification")
                    != "configs/completion/cifar_sources_v3.json"
                or report.get("source_specification_sha256") != specification_hash
                or report.get("expected") != expected
                or len(report.get("complete", {})) != expected
                or report.get("missing") != [] or report.get("invalid") != []):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    source = Path(args.source)
    if not source.is_absolute():
        source = repo / source
    output = (Path(args.output) if args.output else
              Path("results/completion") / f"{args.dataset.lower()}_remote_queue_v3.json")
    if not output.is_absolute():
        output = repo / output
    state = promote(repo, args.dataset, source, output)
    print(json.dumps({"dataset": args.dataset, "source_commit": state["source_commit"],
                      "canonical_ledger": output.relative_to(repo).as_posix()}))


if __name__ == "__main__":
    main()
