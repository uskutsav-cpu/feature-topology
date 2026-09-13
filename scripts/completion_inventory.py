"""Account for planned experiments using actual, validated saved checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.checkpoints import atomic_json, fingerprint

DEFAULTS = dict(width=256, depth=4, dimension=16, manifold="torus", swap=False,
                relevance=0., relevance_mode="periodic", centered=True)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_run(path, check_tensors=False):
    summary = json.loads((path / "summary.json").read_text())
    config = json.loads((path / "config.json").read_text())
    if summary["config"] != config or summary["run_id"] != fingerprint(config):
        raise ValueError(f"Configuration/summary identity mismatch: {path}")
    if path.name != summary["run_id"]:
        raise ValueError(f"Directory identity mismatch: {path}")
    if summary["status"] not in {"converged", "budget_exhausted", "diverged"}:
        raise ValueError(f"Nonterminal status: {path}")
    checkpoint = path / "final.pt"
    record = dict(run_id=path.name, config=config, status=summary["status"],
                  path=str(path), final_sha256=None)
    if summary["status"] != "diverged":
        if not checkpoint.is_file():
            raise ValueError(f"Missing checkpoint: {path}")
        record["final_sha256"] = sha256(checkpoint)
        if check_tensors:
            import torch
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if state.get("config") != config:
                raise ValueError(f"Checkpoint config mismatch: {path}")
            weights = state.get("model", state.get("network"))
            if not weights or any(not torch.isfinite(v).all() for v in weights.values()):
                raise ValueError(f"Invalid checkpoint tensors: {path}")
            record["checkpoint_step"] = state["step"]
            if state["step"] != summary["history"][-1]["step"]:
                raise ValueError(f"Terminal checkpoint/history mismatch: {path}")
        record["final_loss"] = summary["history"][-1]["training_loss"]
        if summary["status"] == "converged" and record["final_loss"] > config["target_loss"]:
            raise ValueError(f"False convergence status: {path}")
    return record


def inventory(repo, check_tensors=False):
    repo = Path(repo).resolve()
    records, failures, seen = [], [], set()
    folders = [repo / "results/main", repo / "results/pilot"]
    for name in ["width", "depth", "relevance", "swapped", "cylinder", "small_network"]:
        for parent in [repo / "results" / name, repo / "results/extended" / name]:
            if parent.exists():
                folders.extend(p.parent for p in parent.glob("*/runs"))
    for folder in folders:
        for summary in sorted(folder.glob("runs/*/summary.json")):
            path = summary.parent
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                records.append(inspect_run(path, check_tensors))
            except (ValueError, KeyError, OSError) as exc:
                failures.append(dict(path=str(path), error=str(exc)))
    studies = {}
    for plan in sorted((repo / "configs/sweeps").glob("*.json")):
        design = json.loads(plan.read_text())
        accounted, missing = [], []
        for requested in design["runs"]:
            candidates = [r for r in records if all(
                {**DEFAULTS, **r["config"]}.get(k) == v for k, v in requested.items())]
            if candidates:
                chosen = sorted(candidates, key=lambda r: (
                    r["status"] != "converged", "/results/main/" not in r["path"], r["run_id"]))[0]
                accounted.append(dict(requested=requested, run_id=chosen["run_id"],
                                      status=chosen["status"], final_sha256=chosen["final_sha256"]))
            else:
                missing.append(requested)
        studies[plan.stem] = dict(expected=len(design["runs"]), accounted=len(accounted),
                                 missing=missing, runs=accounted)
    images = {}
    for name in ["rotated_digits", "circle_quotient", "dsprites", "cifar10", "cifar100"]:
        rows = []
        for summary in sorted((repo / "results" / name).glob("runs/*/summary.json")):
            try:
                r = inspect_run(summary.parent, check_tensors)
                metric = summary.parent / ("quotient.json" if name == "circle_quotient" else "metrics.json")
                r["metric_sha256"] = sha256(metric) if metric.is_file() else None
                rows.append(r)
            except (ValueError, KeyError, OSError) as exc:
                failures.append(dict(path=str(summary.parent), error=str(exc)))
        images[name] = rows
    return dict(schema="feature-topology.completion-inventory.v1", repo=str(repo),
                tensor_validation=check_tensors, studies=studies, images=images,
                validated_runs=records, failures=failures,
                note="Training accounting is separate from full production metric coverage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-tensors", action="store_true")
    args = parser.parse_args()
    result = inventory(args.repo, args.check_tensors)
    atomic_json(Path(args.output), result)
    print(json.dumps(dict(studies={k: {a:v[a] for a in ["expected", "accounted"]}
                                  for k,v in result["studies"].items()},
                          images={k:len(v) for k,v in result["images"].items()},
                          failures=result["failures"]), indent=2))
    sys.exit(bool(result["failures"]))
