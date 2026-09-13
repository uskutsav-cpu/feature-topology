"""Real dSprites shape classification with held-out factor identities.

The fixed position subgrid and all scales/orientations are declared before runs.
Symmetry-adjusted nuisance probes are reported separately for each shape.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from functools import lru_cache
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from src.models.cnn import CNN
from src.models.mlp import ScaledModel
from src.training.checkpoints import atomic_json, save_checkpoint, fingerprint
from src.training.fls import reference_lr
from src.training.train import evaluate
from src.metrics.geometry import cka, effective_rank
from src.metrics.probes import evaluate_probes
from src.metrics.persistence import persistence

SOURCE_BLOB = "d98996d3f8ee1550e06b9d4ed78d0beb478910d5"
GAMMAS = [.125, .5, 1., 4., 16., 64., 128.]
POSITIONS = [4, 12, 20, 28]


@lru_cache(maxsize=32)
def pooling_weights(length,size,dtype,device):
    weights=torch.zeros(size,length,dtype=dtype)
    for i in range(size):
        start,end=i*length//size,((i+1)*length+size-1)//size
        weights[i,start:end]=1/(end-start)
    return weights.to(device)


def adaptive_mean(x, size=6):
    """The same adaptive pooling bins as a separable linear operation.

    MPS does not implement non-divisible adaptive_avg_pool2d dimensions.
    This retains the full 64x64 images and the CNN's specified 6x6 pooling.
    """
    height,width=x.shape[-2:]
    a=pooling_weights(height,size,x.dtype,x.device)
    b=pooling_weights(width,size,x.dtype,x.device)
    return torch.matmul(torch.matmul(a,x),b.T)


class DSpritesCNN(CNN):
    def forward(self,x):
        a=torch.relu(self.conv1(x))
        b=torch.relu(self.conv2(F.avg_pool2d(a,2)))
        return self.head(torch.relu(self.projection(adaptive_mean(b).flatten(1))))

    def representations(self,x):
        a=torch.relu(self.conv1(x))
        b=torch.relu(self.conv2(F.avg_pool2d(a,2)))
        c=torch.relu(self.projection(adaptive_mean(b).flatten(1)))
        return [adaptive_mean(a).flatten(1),adaptive_mean(b).flatten(1),c]


def split_ids(classes):
    """Stratify by shape; split scale/position identities before orientations."""
    selected = np.flatnonzero(np.isin(classes[:, 4], POSITIONS) & np.isin(classes[:, 5], POSITIONS))
    identities = ((classes[:, 1] * 6 + classes[:, 2]) * 32 + classes[:, 4]) * 32 + classes[:, 5]
    groups = {name: [] for name in ["train", "validation", "test"]}
    rng = np.random.default_rng(5317)
    for shape in range(3):
        unique = np.unique(identities[selected[classes[selected, 1] == shape]])
        rng.shuffle(unique)
        # Each shape has 6 scales x 4 x 4 positions = 96 base identities.
        if len(unique) != 96:
            raise ValueError("Unexpected number of factor identities")
        for name, ids in zip(groups, [unique[:58], unique[58:77], unique[77:]]):
            groups[name].extend(ids.tolist())
    return {name: selected[np.isin(identities[selected], ids)] for name, ids in groups.items()}


def prepare(path, root):
    path, root = Path(path), Path(root)
    with path.open("rb") as stream:
        h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    if h.hexdigest() != SOURCE_BLOB:
        raise ValueError("Dataset does not match the official dSprites Git blob")
    with np.load(path, allow_pickle=False) as archive:
        classes, factors = archive["latents_classes"], archive["latents_values"]
    if classes.shape != (737280, 6) or factors.shape != classes.shape:
        raise ValueError("Unexpected official latent schema")
    # Extract only the known image member and memory-map it; do not load 3 GB into RAM.
    image_path = path.parent / "official_imgs.npy"
    if not image_path.exists():
        temporary = image_path.with_suffix(".partial")
        with zipfile.ZipFile(path) as archive, archive.open("imgs.npy") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        temporary.replace(image_path)
    images = np.load(image_path, mmap_mode="r", allow_pickle=False)
    if images.shape != (737280, 64, 64):
        raise ValueError("Unexpected image dimensions")
    partitions = split_ids(classes)
    data = {}
    for name, ids in partitions.items():
        data[name] = dict(x=torch.from_numpy(np.asarray(images[ids], dtype=np.float32)[:, None]),
                          y=torch.from_numpy(classes[ids, 1].astype(np.int64)),
                          phi=factors[ids, 3], factors=classes[ids], ids=ids)
    specification = dict(schema="feature-topology.dsprites-design.v1", official_git_blob=SOURCE_BLOB,
                         task="shape", nuisance="orientation", positions=POSITIONS,
                         all_scales=True, all_orientations=True, image_shape=[64, 64],
                         split_seed=5317, split_identity_factors=["shape", "scale", "x", "y"],
                         partitions={name:ids.tolist() for name,ids in partitions.items()},
                         symmetry_multipliers={"square":4, "ellipse":2, "heart":1},
                         limitation="A fixed position subgrid of official dSprites; rasterization and symmetries preclude exact manifold claims.")
    identity = fingerprint(specification)
    specification["dataset_id"] = identity
    target = root / "dataset_design.json"
    if target.exists() and json.loads(target.read_text()) != specification:
        raise ValueError("Frozen dataset design changed")
    atomic_json(target, specification)
    return data, identity


def build(config):
    torch.manual_seed(config["seed"])
    return ScaledModel(DSpritesCNN(classes=3, width=16), config["gamma"], centered=True)


def train(config, root, data, device):
    directory = Path(root) / fingerprint(config)
    done = directory / "summary.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["config"] != config or (result["status"] != "diverged" and not (directory/"final.pt").exists()):
            raise ValueError("Invalid completed dSprites run")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "config.json", config)
    model = build(config).to(device)
    optimizer = torch.optim.SGD(model.network.parameters(), lr=config["lr"])
    rng = torch.Generator().manual_seed(config["seed"] + 8127)
    pairs = {name:(data[name]["x"].to(device),data[name]["y"].to(device)) for name in ["train","validation"]}
    history, start = [], 0
    resume = directory / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        history, start = state["history"], state["step"]
    status = "budget_exhausted"
    for step in range(start, config["max_steps"] + 1):
        if history and history[-1]["step"] == step and history[-1]["training_loss"] <= config["target_loss"]:
            status = "converged"
            break
        if (step % 100 == 0 or step == config["max_steps"]) and (not history or history[-1]["step"] != step):
            loss, acc = evaluate(model, *pairs["train"], batch=128)
            vl, va = evaluate(model, *pairs["validation"], batch=128)
            if not np.isfinite(loss) or loss > 1e6:
                status = "diverged"
                break
            history.append(dict(step=step, training_loss=loss, training_accuracy=acc,
                                validation_loss=vl, validation_accuracy=va))
            save_checkpoint(resume, dict(model={k:v.cpu() for k,v in model.state_dict().items()},
                            optimizer=optimizer.state_dict(), rng=rng.get_state(), history=history,
                            step=step, config=config))
            if loss <= config["target_loss"]:
                status = "converged"
                break
        if step == config["max_steps"]:
            break
        ids = torch.randint(len(pairs["train"][0]), (64,), generator=rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(pairs["train"][0][ids]), pairs["train"][1][ids])
        if not torch.isfinite(loss):
            status = "diverged"
            break
        loss.backward()
        optimizer.step()
    result = dict(run_id=directory.name, config=config, status=status, history=history, execution_device=device)
    if status != "diverged":
        save_checkpoint(directory / "final.pt", dict(model={k:v.cpu() for k,v in model.state_dict().items()}, config=config, step=step))
    atomic_json(done, result)
    return result


def analyze(config, root, data, device):
    directory = Path(root) / fingerprint(config)
    target = directory / "metrics.json"
    if target.exists():
        return
    model = build(config).to(device).eval()
    def features(x):
        with torch.no_grad():
            batches = [model.network.representations(a.to(device)) for a in x.split(128)]
        return [torch.cat([b[i].cpu() for b in batches]).numpy() for i in range(3)]
    initial = features(data["test"]["x"])
    state = torch.load(directory/"final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    current, training = features(data["test"]["x"]), features(data["train"]["x"])
    tx, ty = data["test"]["x"].to(device), data["test"]["y"].to(device)
    loss, accuracy = evaluate(model, tx, ty, batch=128)
    layers = []
    for layer, h in enumerate(current):
        shape_probes = {}
        for shape, order in enumerate([4, 2, 1]):
            a = data["train"]["y"].numpy() == shape
            b = data["test"]["y"].numpy() == shape
            # Keep a two-class orientation-sector auxiliary target for the shared probe API.
            # Its task_accuracy is explicitly not shape-classification accuracy.
            phi, tphi = data["train"]["phi"][a]*order, data["test"]["phi"][b]*order
            probes = evaluate_probes(training[layer][a], h[b], (np.cos(phi)>0).astype(int),
                      (np.cos(tphi)>0).astype(int), phi, tphi, config["seed"])
            shape_probes[str(shape)] = dict(symmetry_multiplier=order, nuisance_probes=probes,
                     auxiliary_task="sign(cos(symmetry_multiplier * orientation))")
        stats, _ = persistence(h, size=500, repeats=20, maxdim=1,
                               cache_dir=directory.parents[2]/"ph_cache")
        layers.append(dict(layer=layer+1, cka_drift=1-cka(initial[layer],h),
                           effective_rank=effective_rank(h), persistence=stats, shape_probes=shape_probes))
    with (directory/"final.pt").open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream,"sha256").hexdigest()
    atomic_json(target, dict(dataset_id=config["dataset_id"], checkpoint_sha256=checkpoint_hash,
                test=dict(loss=loss,accuracy=accuracy), layers=layers,
                schema="feature-topology.dsprites-metrics.v1",
                limitation="Raster images on a fixed position subgrid; symmetry-adjusted orientation decoding and H1 diagnostics do not establish continuum injectivity."))


def main(args):
    torch.set_num_threads(2)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    data, dataset_id = prepare(args.data, root)
    base = dict(dataset_id=dataset_id, architecture="CNN", width=16, classes=3,
                centered=True, target_loss=.2, pooling="separable_adaptive_mean_v2")
    frozen_path = root/"gamma_to_lr.json"
    calibration_config = dict(base=base, gammas=GAMMAS, calibration_steps=args.calibration_steps,
                              multipliers=[.125,.25,.5,1.,2.,4.,8.], seed=900)
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen["config"] != calibration_config:
            raise ValueError("Frozen dSprites calibration config changed")
    else:
        selection = {}
        for gamma in GAMMAS:
            candidates = []
            for multiplier in calibration_config["multipliers"]:
                c = dict(**base, gamma=gamma, seed=900, max_steps=args.calibration_steps,
                         lr=reference_lr(gamma,3,.05)*multiplier)
                result = train(c,root/"calibration",data,args.device)
                if result["status"] != "diverged":
                    last = result["history"][-1]
                    candidates.append(dict(lr=c["lr"], reached=result["status"]=="converged",
                                           loss=last["training_loss"],step=last["step"]))
                print(json.dumps(dict(stage="calibration",gamma=gamma,lr=c["lr"],status=result["status"])),flush=True)
            if not candidates:
                raise RuntimeError(f"No stable calibration for gamma={gamma}")
            selection[str(gamma)] = min(candidates,key=lambda c:(not c["reached"],c["step"] if c["reached"] else c["loss"],c["loss"]))
            atomic_json(root/"calibration_progress.json",selection)
        frozen = dict(config=calibration_config,selection=selection)
        atomic_json(frozen_path,frozen)
    manifest = []
    for gamma in GAMMAS:
        for seed in range(5):
            c = dict(**base,gamma=gamma,seed=seed,max_steps=args.max_steps,lr=frozen["selection"][str(gamma)]["lr"])
            result = train(c,root/"runs",data,args.device)
            if result["status"] != "diverged":
                analyze(c,root/"runs",data,args.device)
            manifest.append(dict(run_id=result["run_id"],gamma=gamma,seed=seed,status=result["status"]))
            atomic_json(root/"manifest.json",manifest)
            print(json.dumps(manifest[-1]),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/dsprites/dsprites.npz")
    parser.add_argument("--output",default="results/dsprites")
    parser.add_argument("--device",choices=["cpu","mps"],default="mps")
    parser.add_argument("--calibration-steps",type=int,default=1000)
    parser.add_argument("--max-steps",type=int,default=5000)
    main(parser.parse_args())
