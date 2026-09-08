import numpy as np
import torch


def circle_dataset(n=4000, seed=0, grid=False, classes=2):
    """Known S1 data with equal angular sectors as labels."""
    if grid:
        theta = np.arange(n, dtype=np.float32) * (2 * np.pi / n)
    else:
        theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, n).astype(np.float32)
    x = np.stack((np.cos(theta), np.sin(theta)), axis=1).astype(np.float32)
    y = np.floor(np.mod(theta, 2 * np.pi) / (2 * np.pi / classes)).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(theta)
