"""Read upstream JSON artifacts without unpickling model checkpoints.

Exact step joins, experiment/profile separation, alias deduplication and explicit
missingness prevent convenient but invalid pooled comparisons.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import math
import numpy as np
import pandas as pd
from .io import read_json, digest, original_fingerprint, file_digest

PRODUCTION = dict(jacobian_points=10000, ntk_points=128, ph_size=500,
                  ph_repeats=20, ph_maxdim=2, probe_train=5000,
                  probe_test=2000, probe_iterations=300, all_checkpoints=True)
# Different scientific conditions must NEVER be pooled into the same contrast.
DEFAULTS = dict(dimension=16, width=256, depth=4, manifold="torus", swap=False,
                relevance=0., relevance_mode="periodic", centered=True,
                n_train=20000, n_validation=5000, n_grid=10000, batch_size=256,
                target_loss=.05, eval_every=64)
EXECUTION_AXES = {"gamma", "seed", "lr", "calibration_id", "max_steps", "threads"}


def condition_id(config: dict) -> str:
    return digest({k: v for k, v in {**DEFAULTS, **config}.items() if k not in EXECUTION_AXES})[:16]


def production_profile(options: dict) -> dict:
    missing = []
    for key, expected in PRODUCTION.items():
        actual = options.get(key)
        if isinstance(expected, bool):
            good = actual is expected
        else:
            good = (isinstance(actual, int) and not isinstance(actual, bool) and actual >= expected)
        if not good:
            missing.append(key)
    return {"production_resolution": not missing, "insufficient_settings": missing}


@dataclass
class Catalog:
    rows: list[dict] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)

    def issue(self, code: str, path: Path, detail: str, severity: str = "error") -> None:
        self.issues.append(dict(code=code, path=str(path), detail=detail, severity=severity))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def require_clean(self) -> None:
        errors = [i for i in self.issues if i["severity"] == "error"]
        if errors:
            first = errors[0]
            raise ValueError(f"{len(errors)} artifact error(s): {first['code']}: {first['detail']}")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric metric, got {type(value).__name__}")
    return float(value) if math.isfinite(value) else None


def _get(obj: dict, *keys: str) -> float | None:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return _number(obj)


def _ph_mean(layer: dict, degree: str, item: str) -> float | None:
    values = [_get(r, degree, item) for r in layer.get("persistence", [])]
    # Missing replicates are NOT silently dropped.
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _flatten(layer: dict, record: dict) -> dict:
    return dict(layer=int(layer["layer"]), test_accuracy=_get(record, "test_accuracy"),
                test_loss=_get(record, "test_loss"), ntk_drift=_get(record, "ntk_drift"),
                cka_drift=_get(layer, "cka_drift"), effective_rank=_get(layer, "effective_rank"),
                scale=_get(layer, "scale"), local_q01=_get(layer, "jacobian", "q01"),
                local_normalized_q01=_get(layer, "jacobian", "normalized_q01"),
                task_jacobian=_get(layer, "jacobian", "task_norm"),
                nuisance_jacobian=_get(layer, "jacobian", "nuisance_norm"),
                global_q01=_get(layer, "global_margin", "q01"),
                global_normalized_q01=_get(layer, "global_margin", "normalized_q01"),
                collision_score=_get(layer, "collisions", "mean"),
                linear_nuisance_cosine=_get(layer, "probes", "linear", "angular_cosine"),
                mlp_nuisance_cosine=_get(layer, "probes", "mlp", "angular_cosine"),
                linear_task_accuracy=_get(layer, "probes", "linear", "task_accuracy"),
                mlp_task_accuracy=_get(layer, "probes", "mlp", "task_accuracy"),
                ph_h1_top1=_ph_mean(layer, "H1", "top1"),
                ph_h1_top2=_ph_mean(layer, "H1", "top2"),
                ph_h2_top1=_ph_mean(layer, "H2", "top1"))


def load_catalog(roots: Iterable[str | Path], *, profile: str | None = None) -> Catalog:
    """Roots are study directories containing runs/, or the runs directories.

    Profiles are keyed by actual options content, not an arbitrary directory name.
    A requested profile may be an upstream folder fingerprint or our full hash.
    """
    out, seen = Catalog(), set()
    for root in roots:
        root = Path(root)
        run_root = root / "runs" if (root / "runs").is_dir() else root
        if not run_root.is_dir():
            out.issue("missing_study", root, "Study directory is absent")
            continue
        for run in sorted(run_root.iterdir()):
            if not run.is_dir() or not (run / "config.json").exists():
                continue
            resolved = run.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                config = read_json(run / "config.json")
                rid = original_fingerprint(config)
                info = {"run_id": rid, "directory": str(run), "gamma": config["gamma"],
                        "seed": config["seed"], "condition": condition_id(config), "status": "invalid",
                        "profiles": [], "checkpoint_count": len(list(run.glob("step_*.pt"))),
                        "has_final_checkpoint": (run/"final.pt").is_file(),
                        "has_resume_checkpoint": (run/"resume.pt").is_file()}
                out.runs.append(info)
                if not (run/"summary.json").exists():
                    info["status"] = "incomplete"
                    out.issue("missing_summary", run, "Training is not recorded complete", "warning")
                    continue
                summary = read_json(run/"summary.json")
                info["status"] = summary["status"]
                if summary.get("config", config) != config or summary.get("run_id", rid) != rid:
                    out.issue("summary_identity_mismatch", run, "Summary does not match config")
                    continue
                if summary["status"] not in {"converged", "budget_exhausted", "diverged"}:
                    out.issue("unknown_status", run, str(summary["status"]))
                    continue
                if summary["status"] != "diverged" and not info["has_final_checkpoint"]:
                    out.issue("missing_final_checkpoint", run, "Summary exists but final.pt is absent", "warning")
                history = {}
                for h in summary.get("history", []):
                    step = int(h["step"])
                    if step in history and h != history[step]:
                        raise ValueError(f"Conflicting training history at step {step}")
                    history[step] = h
                info["last_logged_loss"] = history[max(history)].get("training_loss") if history else None
                if summary["status"] == "diverged":
                    continue
                for directory in sorted((run/"metrics").glob("*")):
                    if not directory.is_dir() or not (directory/"options.json").exists():
                        continue
                    options = read_json(directory/"options.json")
                    pid = digest(options)
                    pinfo = {"id": pid, "directory_name": directory.name, "options": options,
                             **production_profile(options), "metric_files": 0, "missing_checkpoints": []}
                    info["profiles"].append(pinfo)
                    if profile and profile not in {pid, pid[:16], directory.name}:
                        continue
                    expected = ({p.stem for p in run.glob("step_*.pt")} | {"final"}
                                if options.get("all_checkpoints") else {"step_0000000", "final"})
                    actual = {p.stem for p in directory.glob("*.json")} - {"options"}
                    pinfo["metric_files"] = len(actual)
                    pinfo["missing_checkpoints"] = sorted(expected-actual)
                    canonical = {}
                    for path in sorted(directory.glob("*.json")):
                        if path.name == "options.json":
                            continue
                        record = read_json(path)
                        step = int(record["step"])
                        if record.get("gamma") != config["gamma"] or record.get("seed") != config["seed"]:
                            out.issue("metric_identity_mismatch", path, "Metric gamma/seed differs from config")
                            continue
                        h = history.get(step)
                        if h is None:
                            out.issue("unmatched_step", path, f"No training evaluation at exact step {step}", "warning")
                        for layer in record["layers"]:
                            row = {"run_id": rid, "run_directory": str(run), "condition": info["condition"],
                                   "profile": pid, "profile_directory": directory.name,
                                   "checkpoint": path.stem, "gamma": float(config["gamma"]),
                                   "seed": int(config["seed"]), "step": step, "status": summary["status"],
                                   "training_loss": _get(h, "training_loss") if h else None,
                                   "training_accuracy": _get(h, "training_accuracy") if h else None,
                                   "risk_join": "exact" if h else "missing",
                                   **_flatten(layer, record)}
                            key = (step, row["layer"])
                            if key in canonical:
                                old = canonical[key]
                                excluded = {"checkpoint"}
                                if {k:v for k,v in row.items() if k not in excluded} != {k:v for k,v in old.items() if k not in excluded}:
                                    out.issue("conflicting_checkpoint_alias", path, f"Two metrics differ for {key}")
                                    continue
                                if path.stem != "final":
                                    continue
                            canonical[key] = row
                    out.rows.extend(canonical.values())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                out.issue("invalid_artifact", run, str(exc))
    # A copied run (not just a symlink) must not inflate seed counts.
    unique = {}
    for row in out.rows:
        key = (row["run_id"], row["profile"], row["step"], row["layer"])
        prior = unique.get(key)
        if prior:
            ignored = {"run_directory", "checkpoint", "profile_directory"}
            if {k:v for k,v in prior.items() if k not in ignored} != {k:v for k,v in row.items() if k not in ignored}:
                out.issue("conflicting_copied_run", Path(row["run_directory"]), str(key))
            continue
        unique[key] = row
    out.rows = list(unique.values())
    return out


def matched_risk(frame: pd.DataFrame, target: float) -> tuple[pd.DataFrame, list[dict]]:
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    if frame.empty:
        return frame.copy(), []
    selected, exclusions = [], []
    for key, group in frame.groupby(["run_id", "profile", "layer"], sort=True):
        eligible = group[(group.training_loss.notna()) & (group.training_loss <= target)]
        if eligible.empty:
            exclusions.append(dict(run_id=key[0], profile=key[1], layer=int(key[2]),
                                   reason="no_measured_checkpoint_at_target", target=target))
            continue
        row = eligible.sort_values("step").iloc[0].to_dict()
        row["target_loss"] = target
        row["comparison"] = "first measured checkpoint at or below target, not identical risk"
        selected.append(row)
    result = pd.DataFrame(selected)
    if not result.empty:
        keys = ["condition", "profile", "layer", "gamma", "seed"]
        if result.duplicated(keys).any():
            raise ValueError("Multiple runs for a condition/gamma/seed; choose a study, do not pool them")
    return result, exclusions


def coverage(catalog: Catalog, gammas: Iterable[float], seeds: Iterable[int]) -> dict:
    expected = {(float(g), int(s)) for g in gammas for s in seeds}
    by_condition = {}
    for run in catalog.runs:
        by_condition.setdefault(run["condition"], []).append(run)
    conditions = []
    for condition, runs in by_condition.items():
        observed = {(float(r["gamma"]), int(r["seed"])) for r in runs}
        done = {(float(r["gamma"]), int(r["seed"])) for r in runs
                if r["status"] == "converged" and r["has_final_checkpoint"]}
        full_metrics = {(float(r["gamma"]), int(r["seed"])) for r in runs
                        if any(p["production_resolution"] and not p["missing_checkpoints"]
                               and p["metric_files"] for p in r["profiles"])}
        conditions.append({"condition": condition, "expected": len(expected),
                           "observed": len(observed & expected), "converged_with_weights": len(done & expected),
                           "production_metrics_present": len(full_metrics & expected),
                           "missing_runs": [list(p) for p in sorted(expected-observed)],
                           "not_converged_with_weights": [list(p) for p in sorted(expected-done)],
                           "missing_production_metrics": [list(p) for p in sorted(expected-full_metrics)]})
    return {"conditions": conditions, "issues": catalog.issues,
            "no_runs_found": not catalog.runs,
            "note": "Filesystem snapshot, not evidence that recorded worker PIDs are currently running."}
