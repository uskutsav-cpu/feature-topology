import numpy as np
import src.metrics.persistence as module


def test_ph_subsample_cache(tmp_path, monkeypatch):
    a = np.arange(30)*2*np.pi/30
    h = np.c_[np.cos(a), np.sin(a)]
    first, _ = module.persistence(h, size=25, repeats=2, maxdim=1, cache_dir=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed persistence must not be recomputed")
    monkeypatch.setattr(module, "ripser", forbidden)
    second, _ = module.persistence(h, size=25, repeats=2, maxdim=1, cache_dir=tmp_path)
    assert first == second
