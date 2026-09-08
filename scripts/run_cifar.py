"""CIFAR robustness runner: no latent-manifold or quotient claims."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split
from src.models.resnet import CIFARResNet
from src.models.mlp import ScaledModel
from src.training.checkpoints import atomic_json, fingerprint, save_checkpoint
from src.metrics.geometry import cka, effective_rank, distortion
from src.metrics.persistence import persistence


def load_data(name, path, download=False):
    cls = CIFAR10 if name == "CIFAR10" else CIFAR100
    train = cls(path, train=True, download=download)
    test = cls(path, train=False, download=download)
    ids = np.arange(len(train.targets))
    training, validation = train_test_split(ids, test_size=5000, random_state=543,
                                           stratify=train.targets)
    x = torch.from_numpy(train.data).permute(0, 3, 1, 2)
    y = torch.tensor(train.targets)
    return dict(train=(x[training], y[training]), validation=(x[validation], y[validation]),
                test=(torch.from_numpy(test.data).permute(0, 3, 1, 2), torch.tensor(test.targets)))


def inputs(x, device):
    return (x.to(device).float()/255.-.5)/.5


@torch.no_grad()
def evaluate(model, pair, device, batch=128):
    mode = model.training; model.eval()
    x, y = pair; total = correct = 0
    for a,b in zip(x.split(batch), y.split(batch)):
        logits = model(inputs(a, device)); b = b.to(device)
        total += F.cross_entropy(logits, b, reduction="sum").item()
        correct += (logits.argmax(1) == b).sum().item()
    model.train(mode)
    return dict(loss=total/len(x), accuracy=correct/len(x))


def train(config, root, data, device):
    root = Path(root)/fingerprint(config)
    if (root/"summary.json").exists():
        return json.loads((root/"summary.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/"config.json", config)
    torch.manual_seed(config["seed"])
    model = ScaledModel(CIFARResNet(config["classes"]), config["gamma"]).to(device)
    optimizer = torch.optim.SGD(model.network.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(config["seed"]+1010)
    resume = root/"resume.pt"
    history, start = [], 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["rng"])
        history, start = saved["history"], saved["step"]
    x,y = data["train"]
    status = "budget_exhausted"
    for step in range(start, config["max_steps"]+1):
        if (step == 0 or step % config["eval_every"] == 0 or step == config["max_steps"]) and (not history or history[-1]["step"] != step):
            risk = evaluate(model, data["train"], device)
            val = evaluate(model, data["validation"], device)
            if not np.isfinite(risk["loss"]) or risk["loss"] > 1e6:
                status = "diverged"; break
            history.append(dict(step=step, training_loss=risk["loss"], training_accuracy=risk["accuracy"],
                                validation_loss=val["loss"], validation_accuracy=val["accuracy"]))
            save_checkpoint(resume, dict(network={k:v.cpu() for k,v in model.network.state_dict().items()},
                            optimizer=optimizer.state_dict(), rng=generator.get_state(), history=history, step=step))
            if risk["loss"] <= config["target_loss"]:
                status = "converged"; break
        if step == config["max_steps"]:
            break
        ids = torch.randint(len(x), (config["batch_size"],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs(x[ids], device)), y[ids].to(device))
        if not torch.isfinite(loss):
            status = "diverged"; break
        loss.backward(); optimizer.step()
    result = dict(config=config, status=status, history=history, run_id=root.name, execution_device=device)
    if status != "diverged":
        save_checkpoint(root/"final.pt", dict(network={k:v.cpu() for k,v in model.network.state_dict().items()},
                                              config=config, step=step))
    atomic_json(root/"summary.json", result)
    return result


def analyze(config, root, data, device):
    root = Path(root)/fingerprint(config)
    target = root/"metrics.json"
    if target.exists():
        return
    torch.manual_seed(config["seed"])
    model = ScaledModel(CIFARResNet(config["classes"]), config["gamma"]).to(device).eval()
    x = data["test"][0][:1000]
    def features():
        with torch.no_grad():
            chunks = [model.network.representations(inputs(a, device)) for a in x.split(128)]
        return [torch.cat([v[i].cpu() for v in chunks]).numpy() for i in range(4)]
    initial = features()
    model.network.load_state_dict(torch.load(root/"final.pt", map_location="cpu", weights_only=False)["network"])
    current = features()
    rows = []
    for i,h in enumerate(current):
        stats, diagrams = persistence(h, size=500, repeats=20, maxdim=1,
                                       cache_dir=root.parents[2]/"ph_cache")
        rows.append(dict(layer=i+1, cka_drift=1-cka(initial[i], h), effective_rank=effective_rank(h),
                         distortion=distortion(initial[i], h), persistence=stats))
    # Ambient input-to-logit Jacobian, not a known manifold tangent Jacobian.
    spectrum = []
    for sample in x[:4]:
        a = inputs(sample[None], device)[0]
        j = torch.func.jacrev(lambda t:model(t[None])[0], chunk_size=4)(a)
        spectrum.append(torch.linalg.svdvals(j.flatten(1).cpu()).numpy())
    np.savez_compressed(root/"representations.npz", **{f"initial_h{i+1}":h for i,h in enumerate(initial)},
                         **{f"final_h{i+1}":h for i,h in enumerate(current)},
                         input_logit_jacobian_singular_values=np.stack(spectrum))
    atomic_json(target, dict(test=evaluate(model, data["test"], device), layers=rows,
                  jacobian="Ambient normalized-input to scaled-logit Jacobian on four fixed test images",
                  limitation="No known latent manifold: no injectivity or latent topology claim"))


def main(a):
    torch.set_num_threads(2)
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this environment; use a GPU-enabled execution context")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    data = load_data(a.dataset, a.data, a.download)
    classes = 10 if a.dataset == "CIFAR10" else 100
    gammas = [.125,.5,1.,4.,16.,64.,128.]
    base = dict(classes=classes, dataset=a.dataset, batch_size=128, target_loss=.2, eval_every=500)
    frozen_path = root/"gamma_to_lr.json"
    if not frozen_path.exists():
        chosen = {}
        for gamma in gammas:
            candidates = []
            # Residual-network search center is heuristic; feed-forward depth
            # scaling is not asserted as a theorem for this architecture.
            center = .1*(gamma**2 if gamma <= 1 else gamma**.5)
            for multiplier in [.125,.25,.5,1.,2.,4.,8.]:
                c = dict(**base, gamma=gamma, lr=center*multiplier, seed=900,
                         max_steps=a.calibration_steps)
                result = train(c, root/"calibration", data, a.device)
                if result["status"] != "diverged":
                    last = result["history"][-1]
                    candidates.append(dict(lr=c["lr"], reached=result["status"] == "converged",
                                           loss=last["training_loss"], step=last["step"]))
                print(json.dumps(dict(stage="calibration", gamma=gamma, lr=c["lr"], status=result["status"])), flush=True)
            if not candidates:
                raise RuntimeError(f"No stable rate for gamma={gamma}")
            chosen[str(gamma)] = min(candidates, key=lambda c:(not c["reached"], c["step"] if c["reached"] else c["loss"]))
            atomic_json(root/"calibration_progress.json", chosen)
        atomic_json(frozen_path, dict(selection=chosen, calibration_steps=a.calibration_steps,
                                      protocol=base, device=a.device))
    frozen = json.loads(frozen_path.read_text())
    for gamma in gammas:
        for seed in range(5):
            c = dict(**base, gamma=gamma, seed=seed, lr=frozen["selection"][str(gamma)]["lr"],
                     max_steps=a.max_steps)
            result = train(c, root/"runs", data, a.device)
            if result["status"] != "diverged":
                analyze(c, root/"runs", data, a.device)
            print(json.dumps(dict(stage="main", gamma=gamma, seed=seed, status=result["status"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["CIFAR10","CIFAR100"], required=True)
    p.add_argument("--output", required=True); p.add_argument("--data", default="data/cifar")
    p.add_argument("--download", action="store_true"); p.add_argument("--device", choices=["cpu","mps"], default="mps")
    p.add_argument("--calibration-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=35200)
    main(p.parse_args())
