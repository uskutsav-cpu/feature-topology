import hashlib
import json
import pytest
import gzip
import io
import tarfile
from scripts.freeze_results import (exact_control_paths,nonfinite_paths,readiness,
                                    calibration_artifact_paths,production_queue_paths,
                                    validate_numeric_circle_quotient,
                                    validate_ood_evaluation_row,verify_manifest)
from scripts.pack_release import pack
from scripts.completion_inventory import inspect_run
from src.training.checkpoints import atomic_json, fingerprint
from research_ext.cli import exact_controls


def test_missing_studies_cannot_be_frozen(tmp_path):
    report,_=readiness(tmp_path)
    assert not report['ready']
    assert any(p.get('study')=='design' for p in report['problems'])


def test_freeze_invokes_tensor_validation(tmp_path,monkeypatch):
    import scripts.freeze_results as freezer
    called={}
    def fake_inventory(repo,check_tensors=False):
        called['check_tensors']=check_tensors
        return {'failures':[],'studies':{},'validated_runs':[]}
    monkeypatch.setattr(freezer,'inventory',fake_inventory)
    freezer.readiness(tmp_path)
    assert called['check_tensors'] is True


def test_nonfinite_metric_paths_are_never_silent():
    value={'ok':1.,'bad':None,'nested':[2.,float('inf')]}
    assert nonfinite_paths(value)==['bad','nested[1]']


def test_inventory_rejects_nonfinite_training_history(tmp_path):
    config={'seed':0,'target_loss':1.}
    run=tmp_path/fingerprint(config);run.mkdir()
    atomic_json(run/'config.json',config)
    atomic_json(run/'summary.json',{'config':config,'run_id':run.name,
                                    'status':'diverged','history':[{'training_loss':None}]})
    with pytest.raises(ValueError,match='Nonfinite summary values'):
        inspect_run(run)


def test_exact_controls_are_replayed_before_freeze(tmp_path):
    root=tmp_path/'results/research_ext/exact_controls'
    exact_controls(root)
    assert len(exact_control_paths(tmp_path))==8
    certificate=json.loads((root/'projection.json').read_text())
    certificate['graph']['beta1']=1
    atomic_json(root/'projection.json',certificate)
    with pytest.raises(ValueError,match='replay failed'):
        exact_control_paths(tmp_path)


def test_numerical_circle_quotient_validation_keeps_empirical_scope():
    layer={'beta0':1,'beta1':1,'nodes':10,'edges':10,'affine_segments':10,
           'pair_intersections':10,'tolerance':1e-7}
    report={'polygon_points':360,'tolerance':1e-7,'initial':[layer,layer],
            'final':[layer,layer],'qualification':'finite polygon only'}
    assert validate_numeric_circle_quotient(report) is report
    report['final'][0]={**layer,'nodes':None}
    with pytest.raises(ValueError,match='values'):
        validate_numeric_circle_quotient(report)


def test_hosted_queue_provenance_is_complete(tmp_path):
    config=tmp_path/'configs/completion';config.mkdir(parents=True)
    completion=tmp_path/'results/completion';(completion/'remote_collections').mkdir(parents=True)
    cohorts=[{'name':'primary','expected_unique_profiles':1},
             {'name':'width','expected_unique_profiles':2}]
    atomic_json(config/'production_cohorts_v3.json',{'cohorts':cohorts})
    attempt={'cohort':'primary','workflow_run':'123','url':'https://github.com/o/r/actions/runs/123',
             'run_ids':['a'*16],'started':1.,'finished':2.,'conclusion':'success'}
    atomic_json(completion/'production_queue_primary_v3.json',{
        'schema':'feature-topology.production-queue-state.v1','attempts':[attempt],
        'cohorts':{'primary':{'expected':1,'complete':1,'missing':0,'invalid':[]}}})
    atomic_json(completion/'production_queue_ablation_v3.json',{
        'schema':'feature-topology.production-queue-state.v1','attempts':[{**attempt,'cohort':'width'}],
        'cohorts':{'width':{'expected':2,'complete':2,'missing':0,'invalid':[]}}})
    for name,count in [('primary',1),('width',2)]:
        atomic_json(completion/'remote_collections'/f'{name}.json',{
            'expected':count,'complete':['x']*count,'missing':[],'invalid':[]})
    assert len(production_queue_paths(tmp_path))==5
    state=json.loads((completion/'production_queue_ablation_v3.json').read_text())
    state['cohorts']['width']['missing']=1
    atomic_json(completion/'production_queue_ablation_v3.json',state)
    with pytest.raises(ValueError,match='Hosted cohort incomplete'):
        production_queue_paths(tmp_path)


def test_ood_evaluation_must_be_finite_before_freeze():
    row={'schema':'feature-topology.nuisance-shift-evaluation.v1','status':'evaluated',
         'gamma':1.,'seed':0,'selected_step':4,'actual_training_loss':.05,
         'environments':{name:{'loss':.2,'accuracy':.9,'samples':5000}
                         for name in ['iid','concentrated','spurious','unseen']}}
    assert validate_ood_evaluation_row(row) is row
    row['environments']['unseen']['loss']=None
    with pytest.raises(ValueError,match='Nonfinite nuisance-shift'):
        validate_ood_evaluation_row(row)


def test_calibration_evidence_is_validated_and_retained(tmp_path):
    run=tmp_path/'results/example/calibration'
    config={'gamma':1.,'lr':.1,'seed':900}
    trial=run/fingerprint(config);trial.mkdir(parents=True)
    atomic_json(trial/'config.json',config)
    atomic_json(trial/'summary.json',{'config':config,'run_id':trial.name,
                                      'status':'diverged','history':[{'training_loss':1.}]})
    atomic_json(run.parent/'gamma_to_lr.json',{'selection':{'1.0':{'lr':.1,'loss':1.}}})
    assert len(calibration_artifact_paths(tmp_path))==3
    summary=json.loads((trial/'summary.json').read_text())
    summary['history'][0]['training_loss']=None
    atomic_json(trial/'summary.json',summary)
    with pytest.raises(ValueError,match='Nonfinite calibration trial'):
        calibration_artifact_paths(tmp_path)


def test_frozen_inputs_detect_changes_and_directory_escape(tmp_path):
    source=tmp_path/'result.txt'
    source.write_text('measured result')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=tmp_path/'frozen.json'
    value=dict(schema='feature-topology.frozen-results.v1',ready=True,files={'result.txt':digest})
    manifest.write_text(json.dumps(value))
    assert verify_manifest(tmp_path,manifest)['ready']
    source.write_text('changed result')
    with pytest.raises(ValueError,match='missing or changed'):
        verify_manifest(tmp_path,manifest)
    value['files']={'../outside.txt':digest}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='missing or changed'):
        verify_manifest(tmp_path,manifest)


def test_release_parts_reconstruct_and_are_reproducible(tmp_path):
    source=tmp_path/'measurement.txt'
    source.write_text('actual measured output\n')
    value=dict(schema='feature-topology.frozen-results.v1',ready=True,
               files={'measurement.txt':hashlib.sha256(source.read_bytes()).hexdigest()})
    manifest=tmp_path/'frozen.json'
    manifest.write_text(json.dumps(value))
    first=pack(tmp_path,manifest,tmp_path/'first',part_bytes=100)
    second=pack(tmp_path,manifest,tmp_path/'second',part_bytes=100)
    assert first['parts']==second['parts']
    assert all(p['bytes']<=100 for p in first['parts'])
    payload=b''.join((tmp_path/'first'/p['file']).read_bytes() for p in first['parts'])
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)),mode='r:') as archive:
        assert archive.extractfile('feature-topology/measurement.txt').read()==source.read_bytes()


def test_release_binds_derived_artifacts_to_freeze(tmp_path):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=tmp_path/'measurement.txt'
    source.write_text('measured')
    frozen=tmp_path/'frozen.json'
    frozen.write_text(json.dumps(dict(schema='feature-topology.frozen-results.v1',ready=True,
                                     files={'measurement.txt':digest(source)})))
    figure=tmp_path/'figure.svg'
    figure.write_text('<svg/>')
    analysis=tmp_path/'analysis_manifest.json'
    value=dict(schema='feature-topology.final-analysis.v1',frozen_manifest_sha256=digest(frozen),
               files={'figure.svg':digest(figure)})
    analysis.write_text(json.dumps(value))
    result=pack(tmp_path,frozen,tmp_path/'release',analysis_manifest=analysis)
    assert result['files']['figure.svg']==digest(figure)
    payload=b''.join((tmp_path/'release'/p['file']).read_bytes() for p in result['parts'])
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        assert archive.extractfile('feature-topology/figure.svg').read()==figure.read_bytes()
    figure.write_text('changed figure')
    with pytest.raises(ValueError,match='artifact missing, changed'):
        pack(tmp_path,frozen,tmp_path/'bad',analysis_manifest=analysis)
    value['frozen_manifest_sha256']='wrong freeze'
    analysis.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='not bound'):
        pack(tmp_path,frozen,tmp_path/'wrong',analysis_manifest=analysis)
