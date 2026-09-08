import argparse
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.train import build, data_for
from src.training.checkpoints import atomic_json


def extract(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/"config.json").read_text())
    model = build(config, config["seed"])
    _, _, (gx, gy, latent, q) = data_for(config)
    torch.set_num_threads(2)
    manifest = {}
    for path in sorted(run.glob("step_*.pt"))+[run/"final.pt"]:
        target = output/(path.stem+".npz")
        if not target.exists():
            estimate = len(gx)*config.get("width", 256)*config.get("depth", 4)*4
            if shutil.disk_usage(output).free < estimate+2*1024**3:
                raise RuntimeError("Insufficient disk: preserving a 2 GiB reserve. Use an external output volume.")
            state = torch.load(path, weights_only=False)
            model.load_state_dict(state["model"])
            with torch.no_grad():
                chunks = [model.network.representations(x) for x in gx.split(1024)]
            arrays = {f"h{i+1}":torch.cat([chunk[i] for chunk in chunks]).numpy() for i in range(len(chunks[0]))}
            np.savez_compressed(target, latent=latent.numpy(), labels=gy.numpy(), **arrays)
        manifest[path.stem] = str(target)
    atomic_json(output/"manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); extract(a.run, a.output)
