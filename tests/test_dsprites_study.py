import itertools
import numpy as np
import torch
from scripts.run_dsprites import split_ids, train, adaptive_mean


def test_explicit_pool_matches_pytorch_values_and_gradients():
    x=torch.randn(2,3,32,32,requires_grad=True)
    expected=torch.nn.functional.adaptive_avg_pool2d(x,(6,6))
    actual=adaptive_mean(x)
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(a,b)


def test_dsprites_groups_never_leak_and_all_orientations_remain():
    classes=np.array([[0,*v] for v in itertools.product(range(3),range(6),range(40),[4,12,20,28],[4,12,20,28])])
    partitions=split_ids(classes)
    groups=[]
    for ids in partitions.values():
        unique={tuple(v) for v in classes[ids][:,[1,2,4,5]]}
        groups.append(unique)
        assert len(ids)==40*len(unique)
        for shape in range(3):
            assert set(classes[ids][classes[ids,1]==shape,3])==set(range(40))
    assert all(not a&b for a,b in itertools.combinations(groups,2))
    assert set(np.concatenate(list(partitions.values())))==set(range(len(classes)))


def test_dsprites_terminal_resume_does_not_take_an_extra_update(tmp_path):
    torch.set_num_threads(1)
    data={name:dict(x=torch.rand(8,1,12,12),y=torch.arange(8)%3)
          for name in ["train","validation"]}
    config=dict(seed=1,gamma=1.,lr=.1,max_steps=2,target_loss=100.)
    first=train(config,tmp_path,data,"cpu")
    root=tmp_path/first["run_id"]
    state=torch.load(root/"final.pt",weights_only=True)
    (root/"final.pt").unlink()
    (root/"summary.json").unlink()
    resumed=train(config,tmp_path,data,"cpu")
    after=torch.load(root/"final.pt",weights_only=True)
    assert resumed["status"]=="converged"
    assert after["step"]==state["step"]==0
    assert all(torch.equal(value,after["model"][key]) for key,value in state["model"].items())
