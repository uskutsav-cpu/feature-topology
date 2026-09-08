"""I/O tests use a stub backend, NOT evidence of computed persistent homology."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import numpy as np
import pytest
module = importlib.import_module('src.metrics.persistence')


def fake_ripser(h, maxdim, coeff):
    return {'dgms': [np.array([[0., np.inf]])] + [np.array([[.1, .8]]) for _ in range(maxdim)]}


def test_missing_backend_is_explicit(monkeypatch):
    monkeypatch.setattr(module, 'ripser', None)
    with pytest.raises(RuntimeError, match='No substitute results'):
        module.persistence(np.eye(3), size=3, repeats=1, maxdim=1)


def test_concurrent_cache_writes_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ripser', fake_ripser)
    h = np.eye(10)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: module.persistence(h, size=8, repeats=2, maxdim=1, cache_dir=tmp_path)[0], range(8)))
    assert all(r == rows[0] for r in rows)
    monkeypatch.setattr(module, 'ripser', None)
    cached, _ = module.persistence(h, size=8, repeats=2, maxdim=1, cache_dir=tmp_path)
    assert cached == rows[0]
    assert not list(tmp_path.rglob('*.tmp'))


@pytest.mark.parametrize('kwargs', [{'size': 0}, {'size': 1001}, {'repeats': 0}, {'maxdim': -1}])
def test_invalid_sizes(kwargs):
    with pytest.raises(ValueError): module.persistence(np.eye(3), **kwargs)


def test_actual_ripser_circle_when_installed():
    pytest.importorskip('ripser')
    theta = np.arange(100)*2*np.pi/100
    h = np.column_stack((np.cos(theta), np.sin(theta)))
    rows, _ = module.persistence(h, size=100, repeats=1, maxdim=1)
    assert rows[0]['H1']['top1'] > 1.0
