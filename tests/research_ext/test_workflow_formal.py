from pathlib import Path
import sys
import shutil
import pytest
from research_ext.workflow import execute,make_plan
from research_ext.formal import export_graph,check_formal,_without_comments
from research_ext.cli import exact_controls,main
from research_ext.io import atomic_json,read_json


def minimal_plan(tmp_path,command=None):
    repo=tmp_path/'repo';repo.mkdir()
    (repo/'input.txt').write_text('one')
    script=repo/'task.py'
    script.write_text("from pathlib import Path\np=Path('calls');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\nPath('out.txt').write_text(Path('input.txt').read_text())\n")
    return {'schema':'feature-topology.workflow.v1','repo':str(repo),'suite':'test','min_free_bytes':0,
            'tasks':[{'name':'one','command':command or [sys.executable,'task.py'],
                      'inputs':['input.txt','task.py'],'outputs':['out.txt'],'dependencies':[],'guard':{}}]}

def test_workflow_runs_caches_and_rechecks_inputs(tmp_path):
    plan=minimal_plan(tmp_path);repo=Path(plan['repo'])
    result=execute(plan)
    assert result['events'][0]['status']=='completed'
    assert execute(plan)['events'][0]['status']=='validated_cache_hit'
    assert (repo/'calls').read_text()=='1'
    (repo/'input.txt').write_text('two')
    assert execute(plan)['events'][0]['status']=='completed'
    assert (repo/'out.txt').read_text()=='two'
    (repo/'out.txt').unlink()
    assert execute(plan)['events'][0]['status']=='completed'
    assert (repo/'calls').read_text()=='3'

def test_failure_does_not_mark_complete(tmp_path):
    plan=minimal_plan(tmp_path,[sys.executable,'-c','raise SystemExit(7)'])
    with pytest.raises(RuntimeError):execute(plan)
    status=read_json(Path(plan['repo'])/'results/research_ext/workflow_state/one.json')
    assert status['status']=='failed'

def test_cycles_rejected(tmp_path):
    plan=minimal_plan(tmp_path);plan['tasks'][0]['dependencies']=['one']
    with pytest.raises(ValueError):execute(plan)

def test_unknown_only_rejected(tmp_path):
    plan=minimal_plan(tmp_path)
    with pytest.raises(ValueError):execute(plan,only=['typo'])

def test_main_plan_keeps_gate(tmp_path):
    plan=make_plan(tmp_path)
    assert any('results/pilot/gate.json' in t['inputs'] for t in plan['tasks'])
    assert '--all-checkpoints' in next(t['command'] for t in plan['tasks'] if t['name']=='main-metrics')
    with pytest.raises(FileNotFoundError):execute(plan)

def test_exact_controls_export_and_roundtrip(tmp_path):
    result=exact_controls(tmp_path/'controls')
    assert result['controls']['relu_same_cycle_rank_collision']['beta1']==1
    assert not result['controls']['relu_same_cycle_rank_collision']['injective_on_polygon_subset']
    cert=read_json(tmp_path/'controls/identity.json')
    status=export_graph(cert,tmp_path/'Generated.lean')
    assert status['status']=='lean_source_generated_not_compiled'
    source=(tmp_path/'Generated.lean').read_text()
    assert 'by decide' in source and 'native_decide' not in source

def test_no_lean_never_claims_verified(tmp_path,monkeypatch):
    root=tmp_path/'formal'
    shutil.copytree(Path(__file__).resolve().parents[2]/'formal',root)
    monkeypatch.setattr(shutil,'which',lambda _:None)
    status=check_formal(root,tmp_path/'result')
    assert status['status']=='not_run_no_lean_toolchain'
    assert not status['lean_verified']

def test_proof_placeholder_rejected(tmp_path):
    root=tmp_path/'formal';root.mkdir();(root/'A.lean').write_text('theorem bad : False := by sorry\n')
    status=check_formal(root,tmp_path/'result')
    assert status['status']=='rejected_forbidden_constructs'

def test_nested_comments_do_not_hide_tokens():
    assert 'sorry' in _without_comments('/- outer /- nested -/ end -/ sorry')
    assert 'sorry' not in _without_comments('/- sorry -/ def one := 1')

def test_cli_real_commands(tmp_path):
    assert main(['controls','--output',str(tmp_path/'controls')])==0
    assert main(['verify-certificate',str(tmp_path/'controls/identity.json')])==0
    assert main(['plan','--repo',str(tmp_path),'--output',str(tmp_path/'plan.json')])==0


def test_empty_formal_project_rejected(tmp_path):
    root=tmp_path/'empty';root.mkdir()
    assert check_formal(root,tmp_path/'out')['status']=='incomplete_formal_project'


def test_invalid_guard_is_recorded_as_failure(tmp_path):
    plan=minimal_plan(tmp_path)
    plan['tasks'][0]['guard']={'kind':'json_true'}
    with pytest.raises(KeyError):execute(plan)
    assert read_json(Path(plan['repo'])/'results/research_ext/workflow_state/one.json')['status']=='failed'

def test_runtime_provenance_is_recorded(tmp_path):
    plan=minimal_plan(tmp_path);execute(plan)
    result=read_json(Path(plan['repo'])/'results/research_ext/workflow_state/one.json')
    assert 'python' in result['environment'] and result['source_id']
