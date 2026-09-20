import hashlib
import json
import pytest
import gzip
import io
import tarfile
from scripts.freeze_results import nonfinite_paths,readiness,verify_manifest
from scripts.pack_release import pack
from scripts.completion_inventory import inspect_run
from src.training.checkpoints import atomic_json, fingerprint


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
