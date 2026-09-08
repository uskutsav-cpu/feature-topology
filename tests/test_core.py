import numpy as np
import torch
from src.data.torus import dataset, coordinates, geodesic, labels
from src.models.mlp import MLP, ScaledModel
from src.metrics.geometry import cka, effective_rank
from src.metrics.jacobian import tangent_jacobians, summarize
from src.metrics.injectivity import collisions
from src.metrics.ntk import empirical_ntk
from src.metrics.persistence import persistence
from src.training.train import train


def test_exact_manifold():
    x, y, z, q = dataset(100, grid=True)
    assert torch.allclose(q.T@q, torch.eye(4), atol=1e-6)
    assert torch.allclose(x@q, coordinates(z), atol=1e-6)
    assert np.isclose(geodesic([[0, 0]], [[2*np.pi-.1, 0]])[0], .1)
    assert len(set(y.tolist())) == 4


def test_geometry_invariance():
    h = np.random.default_rng(0).normal(size=(100, 4))
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))
    assert np.isclose(cka(h, .1*h@q+3), 1)
    assert np.isclose(effective_rank(h), effective_rank(h*.01))
    assert effective_rank(np.zeros((100, 4))) == 0


def test_centering_and_gradient_scaling():
    torch.manual_seed(1)
    net = MLP(width=8, depth=2)
    model = ScaledModel(net, gamma=4)
    x = torch.randn(5, 16)
    assert torch.equal(model(x), torch.zeros(5, 4))
    a = torch.autograd.grad(model(x).sum(), net.head.weight)[0]
    b = torch.autograd.grad(net(x).sum(), net.head.weight)[0]
    assert torch.allclose(a*4, b)


def test_analytic_tangent():
    class Identity:
        def representations(self, x):
            return [x]
    _, _, z, q = dataset(100, grid=True)
    j = tangent_jacobians(Identity(), z, q, 0)
    assert np.allclose(np.linalg.svd(j, compute_uv=False), 1, atol=1e-5)
    collapsed = j.copy(); collapsed[..., 1] = 0
    summary, _ = summarize(collapsed)
    assert summary["median"] == 0


def test_ntk_matches_direct_parameter_gradient():
    torch.manual_seed(2)
    net = MLP(dimension=4, width=5, depth=1, classes=2)
    x = torch.randn(3, 4)
    rows = []
    for sample in x:
        outputs = net(sample)
        rows.append(torch.stack([torch.cat([v.flatten() for v in torch.autograd.grad(
            outputs[c], tuple(net.parameters()), retain_graph=True)]) for c in range(2)]))
    j = torch.stack(rows)
    reference = torch.einsum("ncp,mcp->nm", j, j).numpy()/4
    assert np.allclose(empirical_ntk(net, x, gamma=2), reference, atol=1e-5)


def test_known_circle_ph():
    a = np.arange(60)*2*np.pi/60
    h = np.stack((np.cos(a), np.sin(a)), 1)
    rows, _ = persistence(h, size=60, repeats=1, maxdim=1)
    assert rows[0]["H1"]["top1"] > 1
    zero, _ = persistence(h*0, size=60, repeats=1, maxdim=1)
    assert zero[0]["H1"]["top1"] == 0


def test_collision_detects_factor_collapse():
    x, _, z, _ = dataset(400, grid=True)
    full = coordinates(z).numpy()
    collapsed = full.copy(); collapsed[:, 2:] = 0
    assert collisions(collapsed, z.numpy(), k=10)["mean"] > collisions(full, z.numpy(), k=10)["mean"]


def test_no_repeat(tmp_path):
    c = dict(gamma=1., lr=.01, seed=0, width=8, depth=1, n_train=64,
             n_validation=32, n_grid=25, max_steps=2)
    first = train(c, tmp_path)
    summary = tmp_path/first["run_id"]/"summary.json"
    stamp = summary.stat().st_mtime_ns
    assert train(c, tmp_path) == first
    assert summary.stat().st_mtime_ns == stamp
