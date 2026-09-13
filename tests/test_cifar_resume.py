import torch
import pytest
from scripts import run_cifar


def test_cifar_terminal_resume_and_config_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(run_cifar,'CIFARResNet',lambda classes:torch.nn.Sequential(
        torch.nn.Flatten(),torch.nn.Linear(3*4*4,classes)))
    data={name:(torch.randint(256,(8,3,4,4),dtype=torch.uint8),torch.arange(8)%3)
          for name in ['train','validation']}
    config=dict(seed=1,gamma=1.,lr=.1,classes=3,max_steps=2,eval_every=1,target_loss=100.,batch_size=2)
    first=run_cifar.train(config,tmp_path,data,'cpu')
    root=tmp_path/first['run_id']
    original=torch.load(root/'final.pt',weights_only=True)
    (root/'summary.json').unlink()
    (root/'final.pt').unlink()
    resumed=run_cifar.train(config,tmp_path,data,'cpu')
    after=torch.load(root/'final.pt',weights_only=True)
    assert resumed['status']=='converged'
    assert after['step']==original['step']==0
    assert all(torch.equal(v,after['network'][k]) for k,v in original['network'].items())
    (root/'summary.json').unlink()
    state=torch.load(root/'resume.pt',weights_only=True)
    state['config']={**config,'seed':2}
    torch.save(state,root/'resume.pt')
    with pytest.raises(ValueError,match='configuration mismatch'):
        run_cifar.train(config,tmp_path,data,'cpu')
