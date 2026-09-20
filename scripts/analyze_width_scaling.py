"""Apply the frozen finite-width terminology gate to production artifacts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_ext.finite_width import analyze_width_scaling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+",
                        default=["results/main", "results/width", "results/extended/width"])
    parser.add_argument("--output", default="results/analysis/width_scaling_gate.json")
    parser.add_argument("--specification", default="configs/width_scaling_gate_v1.json")
    parser.add_argument("--profile")
    args = parser.parse_args()
    result = analyze_width_scaling(args.roots, args.output, args.specification, args.profile)
    print(json.dumps({key: result[key] for key in
                      ["status", "terminology", "phase_transition_language_allowed"]}, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 2)
