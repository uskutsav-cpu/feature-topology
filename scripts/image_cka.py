"""Image-only reporting around the frozen linear CKA definition."""
import numpy as np

from src.metrics.geometry import cka


def cka_diagnostics(initial, current):
    """Report CKA or its exact zero-variance undefinedness without imputation."""
    initial=np.asarray(initial,dtype=np.float64)
    current=np.asarray(current,dtype=np.float64)
    initial=initial-initial.mean(0)
    current=current-current.mean(0)
    initial_norm=float(np.linalg.norm(initial.T@initial))
    current_norm=float(np.linalg.norm(current.T@current))
    if initial_norm*current_norm>0:
        score=cka(initial,current)
        status='defined'
    else:
        score=None
        if initial_norm==0 and current_norm==0:
            status='undefined_zero_variance_both'
        elif initial_norm==0:
            status='undefined_zero_variance_initial'
        else:
            status='undefined_zero_variance_current'
    return dict(score=score,status=status,
                initial_centered_gram_norm=initial_norm,
                current_centered_gram_norm=current_norm)
