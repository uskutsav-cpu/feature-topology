"""Check completeness and produce an immutable-content manifest, without recomputing."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.training.checkpoints import atomic_json


def audit(root):
    root = Path(root)
    report = dict(environment=dict(python=platform.python_version(), machine=platform.machine(),
                                    torch=torch.__version__, mps_available=torch.backends.mps.is_available()),
                  experiments={}, files={})
    for folder in sorted((root/"results").iterdir()):
        if not folder.is_dir():
            continue
        rows = []
        for summary in folder.glob("runs/*/summary.json"):
            data = json.loads(summary.read_text())
            path = summary.parent
            rows.append(dict(run_id=path.name, status=data["status"],
                             final_checkpoint=(path/"final.pt").exists(),
                             metrics=len(list(path.glob("metrics/*/final.json")))))
        report["experiments"][folder.name] = rows
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in [".py", ".json", ".csv", ".md", ".toml", ".txt"] and path.name != "audit.json" and ".pytest_cache" not in path.parts:
            report["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(root/"audit.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", default=".")
    a = p.parse_args(); audit(a.root)
