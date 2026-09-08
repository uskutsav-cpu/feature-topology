import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3]/"work/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[3]/"work/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    root = Path(root); output = root/"figures"; output.mkdir(exist_ok=True)
    frame = pd.read_csv(root/"seed_bootstrap_ci.csv")
    metrics = [("cka_drift", "Geometric deformation (1 − CKA)"),
               ("local_normalized_q01", "Normalized local margin, 1st percentile"),
               ("global_normalized_q01", "Sampled global margin, 1st percentile"),
               ("mlp_nuisance_cosine", "Held-out nuisance probe cosine"),
               ("ph_h1_top2", "Second H1 lifetime (normalized PH)"),
               ("test_accuracy", "Test accuracy")]
    for profile, profile_frame in frame.groupby("profile"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, (metric, label) in zip(axes.flat, metrics):
            for layer, layer_frame in profile_frame[profile_frame.metric == metric].groupby("layer"):
                if metric == "test_accuracy" and layer != profile_frame.layer.max():
                    continue
                a = layer_frame.sort_values("gamma")
                ax.plot(a.gamma, a["mean"], marker="o", ms=3, label=f"Layer {layer}")
                ax.fill_between(a.gamma, a.lower, a.upper, alpha=.12)
            ax.set_xscale("log", base=2); ax.set_xlabel("Feature-learning strength γ")
            ax.set_ylabel(label); ax.grid(alpha=.15)
            if metric == "mlp_nuisance_cosine":
                ax.set_ylim(-.05, 1.05)
        axes.flat[0].legend(frameon=False)
        fig.suptitle(f"{root.name.capitalize()}: feature geometry and information diagnostics\nShading: 95% bootstrap CI across training seeds", fontsize=13)
        fig.savefig(output/f"phase_summary_{profile}.png", dpi=180)
        fig.savefig(output/f"phase_summary_{profile}.pdf")
        plt.close(fig)
    long = pd.read_csv(root/"metrics_long.csv")
    for (profile, layer), part in long.groupby(["profile", "layer"]):
        part = part[part.checkpoint != "final"]
        if part.step.nunique() < 3:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, metric in zip(axes, ["cka_drift", "local_normalized_q01", "mlp_nuisance_cosine"]):
            matrix = part.pivot_table(index="gamma", columns="step", values=metric, aggfunc="mean")
            im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_yticks(range(len(matrix)), [f"{x:g}" for x in matrix.index])
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60)
            ax.set_xlabel("Checkpoint SGD step"); ax.set_ylabel("γ"); ax.set_title(metric)
            fig.colorbar(im, ax=ax)
        fig.savefig(output/f"trajectory_layer{layer}_{profile}.png", dpi=180)
        plt.close(fig)


def null_plot(root):
    root = Path(root)
    path = root/"controls.json"
    value = json.loads(path.read_text()) if path.exists() else {"nulls":json.loads((root/"controls_progress.json").read_text())}
    labels = ["Identity", "Scale ×0.1", "Rotation", "Latent shuffle", "Random cloud", "Nuisance removed"]
    keys = ["identity", "scale_0.1", "rotation", "latent_alignment_permutation", "covariance_cloud", "nuisance_collapsed"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, label, values in [
        (axes[0], "1 − CKA", [value["nulls"][k]["cka_drift"] for k in keys]),
        (axes[1], "Latent collision score", [value["nulls"][k]["collisions"]["mean"] for k in keys]),
        (axes[2], "Second H1 lifetime", [np.mean([p["H1"]["top2"] for p in value["nulls"][k]["ph"]]) for k in keys])]:
        ax.bar(range(6), values, color=["#637c98"]*3+["#d09248", "#aa8686", "#278878"])
        ax.set_xticks(range(6), labels, rotation=55, ha="right"); ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Known controls separate geometric structure from latent correspondence\nPH: 200 points, 3 subsamples, RMS normalized; these are untrained controls", fontsize=12)
    fig.savefig(root/"null_controls.png", dpi=180); fig.savefig(root/"null_controls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--nulls", action="store_true")
    a = p.parse_args(); null_plot(a.root) if a.nulls else plot(a.root)
