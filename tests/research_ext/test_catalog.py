import json
import shutil
import pytest
from research_ext.io import atomic_json,read_json
from research_ext.catalog import load_catalog,matched_risk,coverage,production_profile,PRODUCTION,condition_id
from research_ext.report import analyze

def test_alias_step_not_pseudoreplicated(make_run):
    root,_,_=make_run()
    c=load_catalog([root]); c.require_clean()
    assert len(c.rows)==2
    selected,excluded=matched_risk(c.frame(),.1)
    assert len(selected)==1 and not excluded
    assert selected.iloc[0].step==10

def test_no_nearest_loss_substitution(make_run):
    root,run,metrics=make_run()
    data=read_json(metrics/'final.json'); data['step']=9
    atomic_json(metrics/'final.json',data)
    (metrics/'step_0000010.json').unlink()
    c=load_catalog([root]); c.require_clean()
    row=next(r for r in c.rows if r['step']==9)
    assert row['training_loss'] is None
    selected,excluded=matched_risk(c.frame(),.1)
    assert selected.empty and len(excluded)==1
    assert any(i['code']=='unmatched_step' for i in c.issues)

def test_conflicting_alias_fails(make_run):
    root,_,metrics=make_run()
    d=read_json(metrics/'final.json');d['test_accuracy']=.01
    atomic_json(metrics/'final.json',d)
    with pytest.raises(ValueError): load_catalog([root]).require_clean()

def test_symlinks_and_copies_deduplicated(make_run,tmp_path):
    root,run,_=make_run()
    other=tmp_path/'other'/'runs';other.mkdir(parents=True)
    (other/run.name).symlink_to(run,target_is_directory=True)
    assert len(load_catalog([root,other]).rows)==2
    copy=tmp_path/'copy';shutil.copytree(root,copy)
    assert len(load_catalog([root,copy]).rows)==2

def test_wrong_seed_rejected(make_run):
    root,_,metrics=make_run()
    d=read_json(metrics/'final.json');d['seed']=99
    atomic_json(metrics/'final.json',d)
    with pytest.raises(ValueError): load_catalog([root]).require_clean()

def test_no_claim_production_for_pilot(make_run):
    root,_,_=make_run()
    result=coverage(load_catalog([root]),[.5],[0])
    assert result['conditions'][0]['production_metrics_present']==0
    assert result['conditions'][0]['converged_with_weights']==1

def test_production_profile_requires_all_checkpoints():
    assert production_profile(PRODUCTION)['production_resolution']
    assert not production_profile({**PRODUCTION,'all_checkpoints':False})['production_resolution']

def test_condition_separates_tasks_not_gamma():
    assert condition_id({'gamma':.5,'seed':1})==condition_id({'gamma':16,'seed':2})
    assert condition_id({'swap':True})!=condition_id({'swap':False})

def test_different_configs_cannot_double_count_seed(make_run):
    root,_,_=make_run()
    make_run(root=root,extra={'lr':.02})
    with pytest.raises(ValueError): matched_risk(load_catalog([root]).frame(),.1)

def test_report_end_to_end(make_run,tmp_path):
    root,_,_=make_run(seed=0)
    make_run(seed=1,root=root)
    out=tmp_path/'report'
    result=analyze([root],out,gammas=[.5],seeds=[0,1],repeats=10,transition_repeats=2)
    assert result['matched_risk_rows']==2
    assert (out/'input_hashes.json').exists()
    assert read_json(out/'model_comparisons.json')[0]['status']=='insufficient_data'
    assert read_json(out/'event_intervals.json')==[]
