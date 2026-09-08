import json
from pathlib import Path
import pytest
import torch
from src.training.train import train


def test_interruption_resumes_same_minibatches(tmp_path, monkeypatch):
    config = dict(gamma=1., lr=.01, seed=3, width=8, depth=1, n_train=64,
                  n_validation=32, n_grid=25, max_steps=8, eval_every=2, target_loss=.0001)
    full = train(config, tmp_path/"full")
    original = torch.optim.SGD.step
    calls = [0]
    def interrupted(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError("simulated interruption")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.SGD, "step", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(config, tmp_path/"resume")
    monkeypatch.setattr(torch.optim.SGD, "step", original)
    resumed = train(config, tmp_path/"resume")
    a = torch.load(tmp_path/"full"/full["run_id"]/"final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path/"resume"/resumed["run_id"]/"final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
