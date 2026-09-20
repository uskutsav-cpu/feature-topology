import json
from pathlib import Path
import pytest
import torch
from scripts.prepare_ablation_inputs import DATA_KEYS, STUDIES, groups
from scripts import prepare_primary_inputs as exporter
from scripts.remote_metrics import unpack_input


def test_conditions_and_shared_baselines():
    rows=[dict(run_id=str(i),status='converged',path=str(i),
               config=dict(gamma=1.,seed=i,relevance=float(i),width=64)) for i in range(2)]
    report=dict(studies=dict(relevance=dict(expected=3,accounted=3,missing=[],
                runs=[dict(run_id='0'),dict(run_id='0'),dict(run_id='1')])),validated_runs=rows)
    result,excluded=groups(report,'relevance')
    assert len(result)==2 and not excluded
    assert sum(len(g['rows']) for g in result.values())==2
    assert {g['data_config']['relevance'] for g in result.values()}=={0.,1.}
    report['studies']['relevance']['missing']=[{}]
    with pytest.raises(ValueError,match='incomplete'):groups(report,'relevance')


def test_ood_is_a_first_class_separate_data_condition():
    assert 'ood' in STUDIES
    assert 'nuisance_condition' in DATA_KEYS


def test_ablation_archive_round_trip_and_stale_rejection(tmp_path,monkeypatch):
    run=tmp_path/'0123456789abcdef';run.mkdir()
    for name in ['config.json','summary.json','final.pt','step_0000000.pt']:
        (run/name).write_bytes(name.encode())
    condition=dict(dimension=16,manifold='cylinder',swap=False,relevance=0.,relevance_mode='periodic')
    config=dict(**condition,gamma=1.,seed=0)
    monkeypatch.setattr(exporter,'inspect_run',lambda *a,**k:dict(config=config))
    tensor=torch.zeros(3,2)
    monkeypatch.setattr(exporter,'data_for',lambda c:(None,None,(tensor,)*4))
    monkeypatch.setattr(exporter,'dataset',lambda *a,**k:(tensor,)*4)
    kwargs=dict(rows=[dict(run_id=run.name,gamma=1.,seed=0)],data_config=condition,
                run_paths={run.name:run})
    index=exporter.package(tmp_path,tmp_path/'inputs',**kwargs)
    restored,arrays=unpack_input(index,run.name,tmp_path/'inputs',tmp_path/'worker')
    assert (restored/'final.pt').read_bytes()==b'final.pt'
    assert arrays.exists()
    # A repeat uses the verified archive; a changed checkpoint is rejected.
    assert exporter.package(tmp_path,tmp_path/'inputs',**kwargs)['runs']==index['runs']
    (run/'final.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='checkpoint bytes'):
        exporter.package(tmp_path,tmp_path/'inputs',**kwargs)
    kwargs['data_config']={**condition,'manifold':'torus'}
    with pytest.raises(ValueError,match='different condition'):
        exporter.package(tmp_path,tmp_path/'inputs',**kwargs)
