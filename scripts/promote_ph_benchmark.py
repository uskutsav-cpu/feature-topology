"""Import already-computed H2 benchmark repeats into the production PH cache."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.train import build, data_for
from src.metrics.geometry import rms_scale


torch.set_num_threads(1)
root = Path("results/pilot")
for result in (root/"ph_feasibility").glob("*.json"):
    metadata = json.loads(result.read_text())
    run = root/"runs"/result.stem
    config = json.loads((run/"config.json").read_text())
    model = build(config, config["seed"])
    model.load_state_dict(torch.load(run/"final.pt", weights_only=False)["model"])
    _, _, (x, _, _, _) = data_for(config)
    with torch.no_grad():
        h = torch.cat([model.network.representations(a)[-1] for a in x.split(1024)]).numpy()
    if rms_scale(h) != metadata["statistics"][0]["scale"]:
        raise RuntimeError("Benchmark representation does not match")
    digest = hashlib.sha256(np.ascontiguousarray(h).tobytes())
    digest.update(json.dumps(dict(shape=h.shape, dtype=str(h.dtype), size=500, maxdim=2,
                                 seed=2026, normalize=True), sort_keys=True).encode())
    cache = root.parent/"ph_cache"/digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/"repeat_000.npz"
    if target.exists():
        continue
    with np.load(result.with_suffix(".npz")) as diagrams:
        np.savez_compressed(target, indices=np.random.default_rng(2026).choice(len(h), 500, replace=False),
                            statistics=json.dumps(metadata["statistics"][0]), **dict(diagrams))
    print("Imported", result.stem)
