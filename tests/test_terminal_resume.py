"""Regression: a saved terminal evaluation must not cause another SGD update."""
import importlib
from pathlib import Path
import pytest
import torch

training = importlib.import_module('src.training.train')


@pytest.mark.parametrize('target', [2.0, 1.3])
def test_crash_after_terminal_resume_preserves_weights(tmp_path, monkeypatch, target):
    config = dict(gamma=1., lr=.2, seed=3, width=8, depth=1, n_train=64,
                  n_validation=32, n_grid=25, max_steps=256, eval_every=2,
                  target_loss=target, threads=1)
    full = training.train(config, tmp_path/'full')
    assert full['status'] == 'converged'
    original = training.save_checkpoint

    def crash(path, payload):
        if (Path(path).name.startswith('step_') and payload['history']
                and payload['history'][-1]['training_loss'] <= target):
            raise RuntimeError('crash after terminal resume was saved')
        return original(path, payload)

    monkeypatch.setattr(training, 'save_checkpoint', crash)
    with pytest.raises(RuntimeError, match='terminal resume'):
        training.train(config, tmp_path/'resumed')
    monkeypatch.setattr(training, 'save_checkpoint', original)
    resumed = training.train(config, tmp_path/'resumed')
    a = torch.load(tmp_path/'full'/full['run_id']/'final.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed'/resumed['run_id']/'final.pt', weights_only=False)
    assert resumed['status'] == 'converged'
    assert a['step'] == b['step']
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert (tmp_path/'resumed'/resumed['run_id']/f"step_{b['step']:07d}.pt").exists()
