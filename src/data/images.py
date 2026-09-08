import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.data.augmentations import rotate_images


def rotated_digits(seed=0, rotations=4, size=24):
    """Real handwritten digits with known synthetic orientation.

    Base identities are split before rotations to prevent identity leakage.
    Rotation can make digit labels ambiguous (notably 6/9); report this limit.
    """
    data = load_digits()
    all_ids = np.arange(len(data.target))
    train_ids, rest = train_test_split(all_ids, test_size=.4, stratify=data.target, random_state=42)
    validation_ids, test_ids = train_test_split(rest, test_size=.5, stratify=data.target[rest], random_state=43)
    rng = np.random.default_rng(seed)
    output = {}
    for name, ids in [("train", train_ids), ("validation", validation_ids), ("test", test_ids)]:
        repeated = np.repeat(ids, rotations)
        phi = rng.uniform(0, 2*np.pi, len(repeated))
        x = rotate_images(data.images[repeated]/16., phi, size)
        output[name] = (torch.from_numpy(x), torch.tensor(data.target[repeated]), phi, repeated)
    grid_ids = test_ids[:24]
    repeated = np.repeat(grid_ids, 36)
    phi = np.tile(np.arange(36)*np.pi/18, len(grid_ids))
    output["grid"] = (torch.from_numpy(rotate_images(data.images[repeated]/16., phi, size)),
                       torch.tensor(data.target[repeated]), phi, repeated)
    return output
