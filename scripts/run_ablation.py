import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate import calibrate
from run_pilot import run_sweep
from src.training.checkpoints import fingerprint, atomic_json


def run(plan, main_calibration, output, main_runs, gate_path):
    if not json.loads(Path(gate_path).read_text()).get("proceed_to_main"):
        raise RuntimeError("Pilot gate must pass")
    design = json.loads(Path(plan).read_text())
    main = json.loads(Path(main_calibration).read_text())
    groups = {}
    for row in design["runs"]:
        condition = {k:v for k,v in row.items() if k not in ["gamma", "seed"]}
        key = fingerprint(condition)
        groups.setdefault(key, dict(condition=condition, gammas=set(), seeds=set()))
        groups[key]["gammas"].add(row["gamma"]); groups[key]["seeds"].add(row["seed"])
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, group in groups.items():
        condition = group["condition"]
        default = dict(width=256, depth=4, manifold="torus", swap=False, relevance=0., relevance_mode="periodic")
        primary = all(v == default.get(k) for k,v in condition.items())
        group_root = root/key
        if primary:
            calibration_path = Path(main_calibration)
        else:
            config = {**main["config"], **condition, "gammas":sorted(group["gammas"])}
            calibration_root = group_root/"calibration"
            calibrate(config, calibration_root, root.parent/"ablation_calibration_trials")
            calibration_path = calibration_root/"gamma_to_lr.json"
        completed = run_sweep(calibration_path, group_root, sorted(group["seeds"]),
                              max_steps=8192, reuse_runs=main_runs, gammas=sorted(group["gammas"]))
        manifest.append(dict(condition=condition, path=str(group_root), runs=completed))
        atomic_json(root/"manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True); p.add_argument("--output", required=True)
    p.add_argument("--main-calibration", default="results/calibration_main/gamma_to_lr.json")
    p.add_argument("--main-runs", default="results/main/runs")
    p.add_argument("--gate", default="results/pilot/gate.json")
    a = p.parse_args(); run(a.plan, a.main_calibration, a.output, a.main_runs, a.gate)
