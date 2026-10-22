import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_production_noise import close_numeric,evaluate,plan,validate_probe_arrays
from src.training.train import build


def arrays(config):
    from scripts.compute_metrics import analysis_condition
    rng=np.random.default_rng(9001)
    a=rng.uniform(0,2*np.pi,(5000,2)).astype(np.float32)
    b=rng.uniform(0,2*np.pi,(2000,2)).astype(np.float32)
    q=np.eye(16,4,dtype=np.float32)
    def inputs(z):return np.column_stack([np.cos(z[:,0]),np.sin(z[:,0]),np.cos(z[:,1]),np.sin(z[:,1])])@q.T
    return dict(data_config=np.array(json.dumps(analysis_condition(config))),px=inputs(a),pz=a,
        tx=inputs(b),tz=b,q=q)


def test_plan_covers_original_main_and_relevance_without_double_counting_shared_models():
    rows=plan(Path(__file__).resolve().parents[1])
    assert len(rows)==305 and len({r['record']['run_id'] for r in rows})==305
    assert sum('main' in r['roles'] for r in rows)==130
    assert sum('relevance' in r['roles'] for r in rows)==210
    assert sum(len(r['roles'])==2 for r in rows)==35
    assert {r['relevance'] for r in rows}=={0.,.05,.1,.25,.5,1.}


def test_probe_split_rejects_overlap_condition_changes_and_nonfinite_arrays():
    config=dict(relevance=.25)
    data=arrays(config);validate_probe_arrays(data,config)
    bad=copy.deepcopy(data);bad['tz'][0]=bad['pz'][0]
    with pytest.raises(ValueError,match='overlap'):validate_probe_arrays(bad,config)
    with pytest.raises(ValueError,match='different task condition'):validate_probe_arrays(data,dict(relevance=.5))
    bad=copy.deepcopy(data);bad['px'][0,0]=np.nan
    with pytest.raises(ValueError,match='frozen split'):validate_probe_arrays(bad,config)


def source_fixture(tmp_path,reached):
    # Synthetic metadata fixture; no claim that these unit-test weights were trained.
    config=dict(width=8,depth=1,dimension=16,gamma=1.,seed=0,centered=True,relevance=.25)
    directory=tmp_path/'results/main/runs/unit-model';directory.mkdir(parents=True)
    (directory/'config.json').write_text(json.dumps(config))
    summary=dict(config=config,run_id='unit-model',status='converged' if reached else 'budget_exhausted',
        history=[dict(step=0,training_loss=1.),dict(step=1,training_loss=.08 if reached else .3)])
    (directory/'summary.json').write_text(json.dumps(summary))
    model=build(config,0)
    torch.save(dict(config=config,step=1,model=model.state_dict()),directory/'step_0000001.pt')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    protocol=dict(noise_levels=[0.,.001,.01,.1],scope='Test fixture, not an empirical production result')
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/production_noise_completion_v1.json').write_text(json.dumps(protocol))
    row=dict(record=dict(run_id='unit-model',gamma=1.,seed=0,config=config,files=files,archive_sha256='a'*64),
        cohort='primary',index='fixture-index.json',index_sha256='b'*64,relevance=.25,roles=['main'],
        array_path='fixture-array.npz',analysis_data={'sha256':'c'*64})
    return row,protocol,arrays(config)


def test_existing_noise_method_evaluates_same_selected_state_and_replays_all_levels(tmp_path):
    torch.set_num_threads(2)
    row,protocol,data=source_fixture(tmp_path,True)
    first=evaluate(tmp_path,row,protocol,{},data)
    second=evaluate(tmp_path,row,protocol,{},data)
    assert first['binding']['selected_step']==1 and first['binding']['actual_training_loss']==.08
    assert first['status']=='evaluated' and len(first['noise_rows'])==4
    assert [r['noise_level'] for r in first['noise_rows']]==[0.,.001,.01,.1]
    assert first['information_destruction_proved'] is False
    close_numeric(first,second)
    changed=copy.deepcopy(first);changed['noise_rows'][1]['angular_cosine']+=.01
    with pytest.raises(ValueError,match='numeric difference'):close_numeric(first,changed)


def test_unreached_risk_is_preserved_without_substituting_the_available_final_state(tmp_path):
    row,protocol,data=source_fixture(tmp_path,False)
    result=evaluate(tmp_path,row,protocol,{},data)
    assert result['status']=='unreached_common_risk' and result['noise_rows']==[]
    assert result['binding']['checkpoint'] is None and result['binding']['selected_step'] is None
    assert result['minimum_stored_training_loss']==.3


def test_replay_tolerance_does_not_accept_changed_statuses_or_nan():
    close_numeric({'score':.1},{'score':.1+1e-13})
    with pytest.raises(ValueError,match='value changed'):close_numeric({'status':True},{'status':1})
    with pytest.raises(ValueError,match='numeric difference'):close_numeric({'score':float('nan')},{'score':0.})
