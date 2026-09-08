import argparse
import json
import os
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.train import build, data_for, evaluate
from src.data.torus import dataset
from src.training.checkpoints import atomic_json, fingerprint
from src.metrics.geometry import cka, effective_rank, distortion, rms_scale
from src.metrics.jacobian import tangent_jacobians, summarize
from src.metrics.injectivity import global_margin, collisions
from src.metrics.ntk import empirical_ntk
from src.metrics.probes import evaluate_probes
from src.metrics.persistence import persistence


def features(net, x):
    with torch.no_grad():
        chunks = [net.representations(v) for v in x.split(1024)]
    return [torch.cat([c[i] for c in chunks]).numpy() for i in range(len(chunks[0]))]


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compute(run, options):
    run = Path(run)
    expected = ([p.stem for p in run.glob("step_*.pt")]+["final"] if options["all_checkpoints"] else ["step_0000000", "final"])
    cached_output = run/"metrics"/fingerprint(options)
    if expected and all((cached_output/(name+".json")).exists() for name in expected):
        return
    config = json.loads((run/"config.json").read_text())
    if json.loads((run/"summary.json").read_text())["status"] == "diverged":
        return
    torch.set_num_threads(2)
    model = build(config, config["seed"])
    _, _, (gx, gy, z, q) = data_for(config)
    initial = features(model.network, gx)
    rng = np.random.default_rng(777)
    ids = np.sort(rng.choice(len(gx), min(options["jacobian_points"], len(gx)), replace=False))
    ntk_ids = np.sort(rng.choice(len(gx), min(options["ntk_points"], len(gx)), replace=False))
    options_id = fingerprint(options)
    output = run/"metrics"/options_id
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"options.json", options)
    baseline_config = {k:v for k,v in config.items() if k not in ["gamma", "lr", "max_steps", "calibration_id"]}
    baseline_options = {k:v for k,v in options.items() if k != "all_checkpoints"}
    baseline_key = fingerprint(dict(config=baseline_config, options=baseline_options))
    shared_initial = run.parents[2]/"initial_metric_cache"/baseline_key
    shared_initial.mkdir(parents=True, exist_ok=True)
    k0_path = output/"initial_ntk.npy"
    if k0_path.exists():
        k0 = np.load(k0_path)
    else:
        raw_path = shared_initial/"raw_ntk.npy"
        if raw_path.exists():
            raw_k0 = np.load(raw_path)
        else:
            raw_k0 = empirical_ntk(model.network, gx[ntk_ids], gamma=1.)
            with raw_path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, raw_k0)
            os.replace(raw_path.with_suffix(".tmp"), raw_path)
        k0 = raw_k0/(config["gamma"]**2)
        np.save(k0_path, k0)
    common = {k:config[k] for k in ["dimension", "manifold", "swap", "relevance", "relevance_mode"] if k in config}
    px, py, pz, _ = dataset(options["probe_train"], seed=4001, **common)
    tx, ty, tz, _ = dataset(options["probe_test"], seed=4002, **common)
    checkpoints = sorted(run.glob("step_*.pt")) if options["all_checkpoints"] else [run/"step_0000000.pt", run/"final.pt"]
    if options["all_checkpoints"]:
        checkpoints.append(run/"final.pt")
    for path in checkpoints:
        target = output/(path.stem+".json")
        if target.exists():
            continue
        is_initial = path.stem == "step_0000000" and config.get("centered", True)
        if is_initial and (shared_initial/"row.json").exists():
            row = json.loads((shared_initial/"row.json").read_text())
            row["gamma"] = config["gamma"]
            row["initial_metric_reused"] = str(shared_initial)
            for artifact in shared_initial.glob("step_0000000_*.npz"):
                destination = output/artifact.name
                if not destination.exists():
                    try:
                        os.link(artifact, destination)
                    except OSError:
                        shutil.copy2(artifact, destination)
            atomic_json(target, row)
            continue
        state = torch.load(path, weights_only=False)
        model.load_state_dict(state["model"])
        h = features(model.network, gx)
        k = empirical_ntk(model.network, gx[ntk_ids], config["gamma"])
        test_loss, test_accuracy = evaluate(model, tx, ty)
        row = dict(step=state["step"], gamma=config["gamma"], seed=config["seed"],
                   test_loss=test_loss, test_accuracy=test_accuracy,
                   ntk_drift=float(np.linalg.norm(k-k0)/np.linalg.norm(k0)),
                   ntk_points=len(ntk_ids), layers=[])
        ph = features(model.network, px); th = features(model.network, tx)
        for layer, current in enumerate(h):
            scale = rms_scale(current)
            j = tangent_jacobians(model.network, z[ids], q, layer, config.get("manifold", "torus"))
            jac, singular = summarize(j, scale, config.get("swap", False))
            np.savez_compressed(output/f"{path.stem}_layer{layer+1}_tangents.npz", indices=ids,
                                sigma_min=singular, factor_norms=np.linalg.norm(j, axis=1))
            margin = global_margin(current, z.numpy(), manifold=config.get("manifold", "torus"))
            margin["normalized_q01"] = margin["q01"]/scale if scale > 0 else 0.
            nuisance_column = int(not config.get("swap", False))
            probes = evaluate_probes(ph[layer], th[layer], py.numpy(), ty.numpy(),
                                     pz[:, nuisance_column].numpy(), tz[:, nuisance_column].numpy(),
                                     seed=config["seed"], max_iter=options["probe_iterations"],
                                     periodic=config.get("manifold", "torus") == "torus" or config.get("swap", False))
            stats, diagrams = persistence(current, size=options["ph_size"], repeats=options["ph_repeats"],
                                          maxdim=options["ph_maxdim"], cache_dir=run.parents[2]/"ph_cache")
            np.savez_compressed(output/f"{path.stem}_layer{layer+1}_ph.npz",
                                **{f"r{r}_h{d}":v for r, ds in enumerate(diagrams) for d,v in enumerate(ds)})
            row["layers"].append(dict(layer=layer+1, cka_drift=1-cka(initial[layer], current),
                 effective_rank=effective_rank(current), scale=scale, distortion=distortion(initial[layer], current),
                 jacobian=jac, global_margin=margin,
                 collisions=collisions(current, z.numpy(), manifold=config.get("manifold", "torus")),
                 probes=probes, persistence=stats))
        atomic_json(target, clean(row))
        if is_initial:
            for artifact in output.glob("step_0000000_*.npz"):
                destination = shared_initial/artifact.name
                if not destination.exists():
                    try:
                        os.link(artifact, destination)
                    except OSError:
                        shutil.copy2(artifact, destination)
            atomic_json(shared_initial/"row.json", clean(row))
        print(f"metrics {run.name} {path.stem}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--jacobian-points", type=int, default=10000)
    p.add_argument("--ntk-points", type=int, default=128)
    p.add_argument("--ph-size", type=int, default=500)
    p.add_argument("--ph-repeats", type=int, default=20)
    p.add_argument("--ph-maxdim", type=int, default=2)
    p.add_argument("--probe-train", type=int, default=5000)
    p.add_argument("--probe-test", type=int, default=2000)
    p.add_argument("--probe-iterations", type=int, default=300)
    p.add_argument("--all-checkpoints", action="store_true")
    a = vars(p.parse_args()); runs = Path(a.pop("runs"))
    for run in sorted(runs.iterdir()):
        if (run/"summary.json").exists():
            compute(run, a)
