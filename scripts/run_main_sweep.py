import argparse
import json
from pathlib import Path
from run_pilot import run_sweep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", required=True); p.add_argument("--output", required=True)
    p.add_argument("--pilot-gate", required=True)
    p.add_argument("--max-steps", type=int, default=8192)
    p.add_argument("--reuse-runs", default="results/pilot/runs")
    a = p.parse_args()
    gate = json.loads(Path(a.pilot_gate).read_text())
    if not gate.get("proceed_to_main", False):
        raise RuntimeError("Pilot gate has not passed; inspect matched-risk results before production")
    calibration = json.loads(Path(a.calibration).read_text())
    if len(calibration["selection"]) != 13:
        raise RuntimeError("Production requires all 13 calibrated gamma values")
    run_sweep(a.calibration, a.output, list(range(10)), a.max_steps, a.reuse_runs)
