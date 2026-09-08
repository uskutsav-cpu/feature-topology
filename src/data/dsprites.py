from pathlib import Path
import numpy as np


def load_dsprites(path):
    """Read an explicitly provided official dSprites archive, without pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dSprites archive not found: {path}")
    archive = np.load(path, allow_pickle=False)
    images = archive["imgs"]
    factors = archive["latents_values"]
    classes = archive["latents_classes"]
    if factors.shape[1] != 6 or len(images) != len(factors):
        raise ValueError("Unexpected dSprites latent schema")
    return dict(images=images, factors=factors, classes=classes,
                factor_names=["color", "shape", "scale", "orientation", "x", "y"])
