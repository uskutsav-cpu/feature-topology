import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from src.data.torus import dataset, coordinates
from src.metrics.geometry import cka, effective_rank, rms_scale
from src.metrics.injectivity import collisions, global_margin
from src.metrics.persistence import persistence
from src.metrics.nulls import controls, covariance_cloud
from src.metrics.quotient import image_graph
from src.training.checkpoints import atomic_json


def run(output="results/controls"):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    target = root/"controls.json"
    if target.exists():
        return
    _, _, latent, _ = dataset(400, grid=True)
    latent = latent.numpy()
    h = np.stack([np.cos(latent[:, 0]), np.sin(latent[:, 0]), np.cos(latent[:, 1]), np.sin(latent[:, 1])], 1)
    variants = {"identity":h, **controls(h), "covariance_cloud":covariance_cloud(h),
                "nuisance_collapsed":np.c_[h[:, :2], np.zeros((len(h), 2))]}
    rows = {}
    original_distances = pdist(h)
    for name, values in variants.items():
        stats, _ = persistence(values, size=200, repeats=3, maxdim=2)
        raw, _ = persistence(values, size=200, repeats=1, maxdim=1, normalize=False)
        rows[name] = dict(cka_drift=1-cka(h, values), effective_rank=effective_rank(values),
                          rms_scale=rms_scale(values), collisions=collisions(values, latent),
                          global_margin=global_margin(values, latent), ph=stats, raw_ph=raw,
                          covariance_error=float(np.linalg.norm(np.cov(values.T)-np.cov(h.T))),
                          norm_histogram_wasserstein=float(wasserstein_distance(np.linalg.norm(h, axis=1), np.linalg.norm(values, axis=1))),
                          distance_histogram_wasserstein=float(wasserstein_distance(original_distances, pdist(values))))
        atomic_json(root/"controls_progress.json", rows)
    theta = np.arange(24)*2*np.pi/24
    polygon = np.c_[np.cos(theta), np.sin(theta)]
    quotient = {"circle":image_graph(polygon, [(np.eye(2), np.zeros(2), False)]),
                "interval":image_graph(polygon, [(np.array([[1., 0.]]), np.zeros(1), False)]),
                "point":image_graph(polygon, [(np.zeros((1, 2)), np.zeros(1), False)])}
    atomic_json(target, dict(nulls=rows, polygon_quotients=quotient))


if __name__ == "__main__":
    run()
