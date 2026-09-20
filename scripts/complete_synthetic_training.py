"""Resume the original frozen synthetic designs, with one writer and durable logs."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.checkpoints import atomic_json


def main():
    repo = Path(__file__).resolve().parents[1]
    output = repo / "results/completion"
    output.mkdir(parents=True, exist_ok=True)
    commands = [("main", ["scripts/run_main_sweep.py", "--calibration", "results/calibration_main/gamma_to_lr.json",
                 "--output", "results/main", "--pilot-gate", "results/pilot/gate.json"])]
    commands += [(name, ["scripts/run_ablation.py", "--plan", f"configs/sweeps/{name}.json",
                        "--output", f"results/{name}"])
                 for name in ["ood", "width", "cylinder", "swapped", "depth", "relevance", "small_network"]]
    with (output / "synthetic_training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for name, command in commands:
            row = dict(study=name, command=[sys.executable, *command], started=time.time(), state="running")
            records.append(row)
            atomic_json(output / "synthetic_training.json", records)
            with (output / f"training_{name}.log").open("a") as log:
                process = subprocess.run(row["command"], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=process.returncode, finished=time.time(),
                       state="command_completed" if process.returncode == 0 else "failed")
            atomic_json(output / "synthetic_training.json", records)
            print(json.dumps(row), flush=True)
        return int(any(r["returncode"] for r in records))


if __name__ == "__main__":
    sys.exit(main())
