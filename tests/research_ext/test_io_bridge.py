from pathlib import Path
import math
import pytest
from research_ext.io import atomic_json,read_json,exclusive_lock,digest
from research_ext.bridge import export_checkpoint
from research_ext.exact import Layer,forward,stored_float

def test_strict_json(tmp_path):
    path=tmp_path/'a.json'
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError): read_json(path)
    path.write_text('{"x":NaN}')
    with pytest.raises(ValueError): read_json(path)
    with pytest.raises(ValueError): atomic_json(path,{'x':float('inf')})

def test_lock_never_stolen(tmp_path):
    path=tmp_path/'lock'
    with exclusive_lock(path):
        with pytest.raises(RuntimeError):
            with exclusive_lock(path): pass
    assert not path.exists()

def test_atomic_write(tmp_path):
    path=tmp_path/'sub'/'x.json'
    atomic_json(path,{'b':2,'a':1})
    assert read_json(path)=={'a':1,'b':2}
    assert not list(path.parent.glob('.*.tmp'))

def test_checkpoint_export(tmp_path):
    import torch
    weights={'network.hidden.0.weight':torch.tensor([[1.,0.],[0.,1.]]),
             'network.hidden.0.bias':torch.tensor([.5,.25])}
    path=tmp_path/'model.pt';torch.save({'model':weights,'step':10},path)
    exported=export_checkpoint(path)
    layers=[Layer.from_dict(v) for v in exported['layers']]
    assert forward(tuple(map(stored_float,[-1.,2.])),layers)==tuple(map(stored_float,[0.,2.25]))
    assert exported['provenance']['checkpoint_sha256']
    with pytest.raises(ValueError): export_checkpoint(path,max_parameters=1)
