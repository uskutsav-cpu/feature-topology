"""Class-balanced covariance baselines, including Xu--Liu (ICML 2023) VCI.

VCI uses total-covariance pseudoinversion (Definition 5.3), not within-class
pseudoinversion. It is a linear representation statistic, not an injectivity or
information-loss certificate. Numerical invariance is limited by the declared
relative eigenvalue cutoff.
"""
import numpy as np


def class_covariances(features, labels, classes=4):
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (values.ndim != 2 or not values.shape[1] or len(values) != len(labels)
            or labels.ndim != 1 or not np.isfinite(values).all()):
        raise ValueError('Expected finite sample-by-feature values and one label per sample')
    if classes < 2 or not np.array_equal(np.unique(labels), np.arange(classes)):
        raise ValueError('Every declared class must be present with labels 0 through classes-1')
    means, within = [], np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for label in range(classes):
        group = values[labels == label]
        mean = group.mean(axis=0)
        centered = group-mean
        within += centered.T @ centered / len(group) / classes
        means.append(mean)
    centered_means = np.asarray(means)-np.mean(means, axis=0)
    between = centered_means.T @ centered_means / classes
    return within, between, within+between


def variability_baselines(features, labels, classes=4, relative_cutoff=1e-10):
    if not 0 < relative_cutoff < 1:
        raise ValueError('Eigenvalue cutoff must lie strictly between zero and one')
    within, between, total = class_covariances(features, labels, classes)
    eigenvalues, vectors = np.linalg.eigh(total)
    between_eigenvalues = np.linalg.eigvalsh(between)
    total_max = max(float(eigenvalues[-1]), 0.)
    between_max = max(float(between_eigenvalues[-1]), 0.)
    retained = eigenvalues > relative_cutoff*total_max
    rank_between = int(np.sum(between_eigenvalues > relative_cutoff*between_max))
    trace_between, trace_within = float(np.trace(between)), float(np.trace(within))
    vci = None
    if rank_between and retained.any():
        directions = vectors[:, retained]
        # Equivalent to trace(pinv(total) @ between), without constructing pinv.
        explained = np.sum(np.sum(directions*(between @ directions), axis=0)
                           / eigenvalues[retained])
        vci = float(1-explained/rank_between)
        if not np.isfinite(vci) or not -1e-7 <= vci <= 1+1e-7:
            raise ValueError('VCI outside numerical tolerance; inspect covariance conditioning')
    return dict(
        status='defined' if vci is not None and trace_between > 0 else 'undefined_between_covariance',
        vci=vci,
        within_to_between_covariance_trace_ratio=(trace_within/trace_between
                                                 if trace_between > 0 else None),
        rank_between=rank_between, rank_total=int(retained.sum()),
        trace_within=trace_within, trace_between=trace_between,
        relative_eigenvalue_cutoff=relative_cutoff,
        smallest_retained_total_eigenvalue=(float(eigenvalues[retained][0])
                                           if retained.any() else None),
        largest_total_eigenvalue=total_max,
        classes=classes, samples=len(labels), dimension=total.shape[0],
        weighting='Equal class weights; population within-class covariance',
    )
