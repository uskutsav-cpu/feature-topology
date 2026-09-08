import numpy as np
from scipy.ndimage import rotate, zoom


def rotate_images(images, angles, size=24):
    """Deterministic periodic rotations, zero padding, bilinear interpolation."""
    out = []
    for image, angle in zip(images, angles):
        expanded = zoom(image, size/image.shape[-1], order=1)
        out.append(rotate(expanded, float(angle)*180/np.pi, reshape=False, order=1,
                          mode="constant", cval=0., prefilter=False))
    return np.asarray(out, np.float32)[:, None]
