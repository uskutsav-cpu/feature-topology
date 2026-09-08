import json
import sys
from pathlib import Path
import pytest
from scripts import run_continuation


def tiny_config():
    return dict(study='unit-test', source_commit='test', calibration_id='test',
        gammas=[.5,1.], seeds=[0], learning_rates={'0.5':.01,'1.0':.01},
        training=dict(width=8, depth=1, n_train=64, n_validation=32, n_grid=25,
                      max_steps=2, target_loss=2., threads=1))


def invoke(monkeypatch, config, output, *extra):
    monkeypatch.setattr(sys,'argv',['run_continuation','--config',str(config),'--output',str(output),*extra])
    run_continuation.main()


def test_limit_does_not_truncate_existing_manifest(tmp_path,monkeypatch):
    cfg=tmp_path/'cfg.json';cfg.write_text(json.dumps(tiny_config()))
    root=tmp_path/'out';invoke(monkeypatch,cfg,root)
    before=json.loads((root/'manifest.json').read_text())
    invoke(monkeypatch,cfg,root,'--limit','1')
    assert json.loads((root/'manifest.json').read_text()) == before
    assert len(before)==2


def test_changed_study_is_rejected(tmp_path,monkeypatch):
    cfg=tmp_path/'cfg.json';data=tiny_config();cfg.write_text(json.dumps(data))
    root=tmp_path/'out';invoke(monkeypatch,cfg,root)
    data['training']['lr']=999;cfg.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='configuration changed'): invoke(monkeypatch,cfg,root)


def test_changed_environment_is_rejected(tmp_path,monkeypatch):
    cfg=tmp_path/'cfg.json';cfg.write_text(json.dumps(tiny_config()))
    root=tmp_path/'out';invoke(monkeypatch,cfg,root)
    path=root/'environment.json';env=json.loads(path.read_text());env['torch']='different';path.write_text(json.dumps(env))
    with pytest.raises(RuntimeError,match='environment changed'): invoke(monkeypatch,cfg,root)
