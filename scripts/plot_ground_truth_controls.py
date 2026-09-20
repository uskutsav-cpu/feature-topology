"""Create the diagnostic-separation figure from the exact control results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "isometry": "isometry",
    "deformed": "injective\ndeformation",
    "near_singular": "near-\nsingular",
    "double_cover": "double cover\n(collision)",
    "quotient": "task\nquotient",
}


def run(source, output):
    with open(source) as handle:
        data = json.load(handle)
    names = list(data["controls"])
    rows = [data["controls"][name] for name in names]
    labels = [LABELS[name] for name in names]
    x = np.arange(len(names))

    drift = [row["geometry"]["cka_drift_from_isometry"] for row in rows]
    local = [row["local_fiber_margin"]["minimum"] for row in rows]
    global_ = [row["global_fiber_separation"]["sampled_minimum"] for row in rows]
    probe = [row["held_out_probes"]["mlp"]["angular_cosine"] for row in rows]
    ph_h1 = [row["persistent_homology"][0]["H1"]["top1"] for row in rows]
    graph_h1 = [row["nuisance_fiber_image_graph"]["beta1"] for row in rows]
    collisions = [row["global_fiber_separation"]["tolerance_collision_pairs"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    axes[0, 0].bar(x, drift, color="#3B82F6")
    axes[0, 0].set_title("A. Geometric deformation")
    axes[0, 0].set_ylabel("1 − CKA from isometry")

    floor = 1e-12
    axes[0, 1].plot(x, np.maximum(local, floor), "o-", label="local nuisance margin")
    axes[0, 1].plot(x, np.maximum(global_, floor), "s-", label="global fiber margin")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(data["thresholds"]["near_singular"], color="0.5", ls="--", lw=1)
    axes[0, 1].set_title("B. Local regularity ≠ global survival")
    axes[0, 1].set_ylabel("margin (log scale; zero at floor)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    colors = ["#10B981" if count == 0 else "#EF4444" for count in collisions]
    axes[1, 0].bar(x, probe, color=colors)
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set_ylim(-.1, 1.05)
    axes[1, 0].set_title("C. Held-out nuisance decodability")
    axes[1, 0].set_ylabel("circular cosine (MLP probe)")
    for i, count in enumerate(collisions):
        if count:
            axes[1, 0].text(i, max(probe[i], 0)+.05, f"{count} grid collisions",
                            ha="center", va="bottom", fontsize=7, rotation=90)

    width = .36
    axes[1, 1].bar(x-width/2, ph_h1, width, label="PH H1 top lifetime", color="#8B5CF6")
    axes[1, 1].bar(x+width/2, graph_h1, width, label="fiber image β1", color="#F59E0B")
    axes[1, 1].set_title("D. Metric topology vs. fiber identification")
    axes[1, 1].set_ylabel("diagnostic value")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Ground-truth controls separate deformation, near-singularity, and quotienting",
                 fontsize=13)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/"fiber_survival_controls.png", dpi=220)
    fig.savefig(output/"fiber_survival_controls.pdf")
    plt.close(fig)

    with open(output/"summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["control", "truth_regime", "empirical_regime", "cka_drift",
                         "local_margin", "global_margin", "collision_pairs",
                         "nuisance_probe_cosine", "fiber_image_beta1", "ph_h1_top1"])
        for name, row, values in zip(names, rows, zip(drift, local, global_, collisions,
                                                       probe, graph_h1, ph_h1)):
            writer.writerow([name, row["truth"]["regime"], row["empirical_regime"], *values])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/ground_truth_controls/controls.json")
    parser.add_argument("--output", default="results/ground_truth_controls")
    args = parser.parse_args()
    run(args.source, args.output)
