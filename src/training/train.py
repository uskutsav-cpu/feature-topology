import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.data.torus import dataset
from src.models.mlp import MLP, ScaledModel
from src.training.checkpoints import atomic_json, fingerprint, save_checkpoint


def build(config, seed):
    torch.manual_seed(seed)
    network = MLP(config.get("dimension", 16), config.get("width", 256),
                  config.get("depth", 4))
    return ScaledModel(network, config["gamma"], config.get("centered", True))


def data_for(config):
    options = {k: config[k] for k in ["dimension", "manifold", "swap", "relevance", "relevance_mode"] if k in config}
    return (dataset(config.get("n_train", 20000), seed=1001, **options),
            dataset(config.get("n_validation", 5000), seed=1002, **options),
            dataset(config.get("n_grid", 10000), grid=True, **options))


@torch.no_grad()
def evaluate(model, x, y, batch=2048):
    total_loss = 0.; correct = 0
    for a, b in zip(x.split(batch), y.split(batch)):
        logits = model(a)
        total_loss += F.cross_entropy(logits, b, reduction="sum").item()
        correct += (logits.argmax(1) == b).sum().item()
    return total_loss/len(x), correct/len(x)


def train(config, root, save=True):
    run_id = fingerprint(config)
    directory = Path(root)/run_id
    done = directory/"summary.json"
    if done.exists():
        return json.loads(done.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory/"config.json", config)
    torch.set_num_threads(config.get("threads", 2))
    model = build(config, config["seed"])
    initial = copy.deepcopy(model.network).requires_grad_(False)
    (x, y, _, _), (vx, vy, _, _), _ = data_for(config)
    optimizer = torch.optim.SGD(model.network.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(config["seed"]+9876)
    start_step, history, crossed = 0, [], []
    state_path = directory/"resume.pt"
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["batch_rng"])
        start_step, history, crossed = state["step"], state["history"], state["crossed"]
    began = time.monotonic()
    previous_seconds = history[-1]["seconds"] if history else 0.
    max_steps = config.get("max_steps", 4096)
    status = "budget_exhausted"
    for step in range(start_step, max_steps+1):
        should_eval = step == 0 or step == max_steps or step & (step-1) == 0 or step % config.get("eval_every", 64) == 0
        if should_eval and (not history or history[-1]["step"] != step):
            loss, accuracy = evaluate(model, x, y)
            vl, va = evaluate(model, vx, vy)
            if not np.isfinite(loss) or loss > 1e6:
                status = "diverged"; break
            fresh = [t for t in [.25, .5, .75, .9, .95, .99] if accuracy >= t and t not in crossed]
            crossed.extend(fresh)
            record = dict(step=step, training_loss=loss, training_accuracy=accuracy,
                          validation_loss=vl, validation_accuracy=va,
                          seconds=previous_seconds+time.monotonic()-began, accuracy_crossings=fresh)
            history.append(record)
            payload = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           batch_rng=generator.get_state(), history=history, crossed=crossed, config=config)
            if save:
                save_checkpoint(state_path, payload)
                if step == 0 or step == max_steps or step & (step-1) == 0 or fresh or loss <= config.get("target_loss", .05):
                    save_checkpoint(directory/f"step_{step:07d}.pt", payload)
            if loss <= config.get("target_loss", .05):
                status = "converged"; break
        if step == max_steps:
            break
        ids = torch.randint(len(x), (config.get("batch_size", 256),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[ids]), y[ids])
        if not torch.isfinite(loss):
            status = "diverged"; break
        loss.backward(); optimizer.step()
    displacement = sum((p-initial.state_dict()[name]).square().sum().item()
                       for name, p in model.network.state_dict().items())**.5
    norm0 = sum(p.square().sum().item() for p in initial.state_dict().values())**.5
    result = dict(run_id=run_id, config=config, status=status, history=history,
                  weight_displacement=displacement/norm0,
                  elapsed_seconds=previous_seconds+time.monotonic()-began)
    # Nonfinite parameters make the run invalid; preserve status without NaNs.
    if not np.isfinite(result["weight_displacement"]):
        result["weight_displacement"] = None
    if save and status != "diverged":
        save_checkpoint(directory/"final.pt", dict(model=model.state_dict(), config=config, step=step))
    atomic_json(done, result)
    return result
