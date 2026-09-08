import torch
from torch import nn
from src.models.mlp import ScaledModel


def test_frozen_batchnorm_buffers():
    base = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
    model = ScaledModel(base, gamma=2)
    model.train()
    before = model.initial[0].running_mean.clone()
    logits = model(torch.randn(10, 3)+5)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    assert torch.equal(before, model.initial[0].running_mean)
    assert not torch.equal(before, model.network[0].running_mean)
