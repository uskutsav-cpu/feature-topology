import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.train import build, data_for
from src.metrics.persistence import persistence
from src.training.checkpoints import atomic_json


root = Path("results/pilot")
output = root/"ph_feasibility"
output.mkdir(exist_ok=True)
for run in root.glob("runs/*"):
    config = json.loads((run/"config.json").read_text())
    if config["seed"] != 0:
        continue
    path = output/(run.name+".json")
    if path.exists():
        continue
    torch.set_num_threads(1)
    model = build(config, 0)
    model.load_state_dict(torch.load(run/"final.pt", weights_only=False)["model"])
    _, _, (x, _, _, _) = data_for(config)
    with torch.no_grad():
        h = torch.cat([model.network.representations(a)[-1] for a in x.split(1024)]).numpy()
    began = time.monotonic()
    rows, diagrams = persistence(h, size=500, repeats=1, maxdim=2, cache_dir=root.parent/"ph_cache")
    np.savez_compressed(output/(run.name+".npz"), **{f"H{i}":d for i,d in enumerate(diagrams[0])})
    value = dict(gamma=config["gamma"], seconds=time.monotonic()-began, statistics=rows)
    atomic_json(path, value)
    print(json.dumps(value), flush=True)
