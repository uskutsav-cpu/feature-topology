import hashlib
import io
import json
import tarfile

import pytest

from scripts import pack_release as packaging


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source=tmp_path/'measurement.txt';source.write_text('original')
    original=tmp_path/'original_freeze.json'
    original.write_text(json.dumps(dict(ready=True,schema='feature-topology.frozen-results.v1',
        files={'measurement.txt':sha(source)}),indent=2)+'\n')
    frozen=tmp_path/'frozen.json'
    # Deliberately noncanonical whitespace/order: the archive must preserve its identity.
    frozen.write_text(json.dumps(dict(ready=True,schema='feature-topology.frozen-results.v1',
        files={'measurement.txt':sha(source),'original_freeze.json':sha(original)},
        original_freeze_path='original_freeze.json',original_freeze_sha256=sha(original),additional_research={
            'schema':'feature-topology.additional-research-freeze.v1'}),indent=3)+'\n')
    main=tmp_path/'main';main.mkdir()
    (main/'summary.csv').write_text('metric,value\naccuracy,0.8\n')
    analysis=main/'analysis_manifest.json'
    analysis.write_text(json.dumps(dict(schema='feature-topology.final-analysis.v1',
        frozen_manifest_sha256=sha(frozen),files={'main/summary.csv':sha(main/'summary.csv')})))
    extra=tmp_path/'additional';extra.mkdir()
    for name,payload in [('figure.svg','<svg/>'),('replication_models.csv','run_id\n'),
                         ('replication_domains.csv','run_id,theta\n'),('noise_metrics_long.csv','run_id,noise_level\n'),
                         ('noise_seed_confidence_intervals.csv','metric,n_seeds\n'),
                         ('noise_relevance_paired_contrasts.csv','metric,paired_seeds\n'),
                         ('task_fiber_metrics_long.csv','run_id,family\n'),
                         ('task_fiber_seed_confidence_intervals.csv','metric,n_seeds\n'),
                         ('task_fiber_family_paired_contrasts.csv','family,paired_seeds\n'),
                         ('task_fiber_relevance_paired_contrasts.csv','metric,paired_seeds\n')]:
        (extra/name).write_text(payload)
    additional=extra/'analysis.json'
    value=dict(schema='feature-topology.additional-research-summary.v1',combined_freeze_sha256=sha(frozen),
        replication={'complete_models':36,'complete_domains':144},input_sha256={'measurement.txt':sha(source)},
        production_noise={'unique_models':305},
        production_noise_freeze_audit={'source_bound_same_host_replays_checked':True},
        task_preserving_fibers={'models':210},
        task_preserving_freeze_audit={'source_bound_same_host_replays_checked':True},
        figure_sha256={'figure.svg':sha(extra/'figure.svg')},
        table_sha256={p.name:sha(p) for p in extra.glob('*.csv')})
    additional.write_text(json.dumps(value))
    return frozen,analysis,additional


def test_package_restores_both_analyses_with_their_exact_common_freeze_anchor(tmp_path):
    frozen,main,additional=fixture(tmp_path)
    output=tmp_path/'release'
    result=packaging.pack(tmp_path,frozen,output,part_bytes=200,
        analysis_manifest=main,additional_analysis_manifest=additional)
    assert result['additional_analysis_included'] is True
    payload=b''.join((output/row['file']).read_bytes() for row in result['parts'])
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        restored=archive.extractfile('feature-topology/frozen_manifest.json').read()
        assert restored==frozen.read_bytes()
        assert hashlib.sha256(restored).hexdigest()==result['frozen_manifest_sha256']
        original_payload=archive.extractfile('feature-topology/original_freeze.json').read()
        assert hashlib.sha256(original_payload).hexdigest()==json.loads(restored)['original_freeze_sha256']
        main_payload=json.loads(archive.extractfile('feature-topology/main/analysis_manifest.json').read())
        extra_payload=json.loads(archive.extractfile('feature-topology/additional/analysis.json').read())
        assert main_payload['frozen_manifest_sha256']==extra_payload['combined_freeze_sha256']==sha(frozen)
        for name,expected in extra_payload['table_sha256'].items():
            assert hashlib.sha256(archive.extractfile('feature-topology/additional/'+name).read()).hexdigest()==expected


@pytest.mark.parametrize('damage,match',[
    ('freeze','not bound'),('input','outside the frozen'),('figure','missing, changed'),
    ('table','incomplete'),('noise_table','incomplete'),('noise_cohort','not bound'),
    ('noise_replay','not bound'),('task_table','incomplete'),('task_cohort','not bound'),
    ('task_replay','not bound'),('escape','escapes'),('cohort','not bound')])
def test_unbound_or_incomplete_additional_outputs_cannot_enter_package(tmp_path,damage,match):
    frozen,main,additional=fixture(tmp_path)
    value=json.loads(additional.read_text())
    if damage=='freeze':value['combined_freeze_sha256']='0'*64
    elif damage=='input':value['input_sha256']['unfrozen.txt']='0'*64
    elif damage=='figure':(additional.parent/'figure.svg').write_text('changed')
    elif damage=='table':value['table_sha256'].pop('replication_domains.csv')
    elif damage=='noise_table':value['table_sha256'].pop('noise_metrics_long.csv')
    elif damage=='noise_cohort':value['production_noise']['unique_models']=304
    elif damage=='noise_replay':value['production_noise_freeze_audit']['source_bound_same_host_replays_checked']=False
    elif damage=='task_table':value['table_sha256'].pop('task_fiber_metrics_long.csv')
    elif damage=='task_cohort':value['task_preserving_fibers']['models']=209
    elif damage=='task_replay':value['task_preserving_freeze_audit']['source_bound_same_host_replays_checked']=False
    elif damage=='escape':value['figure_sha256']={'../figure.svg':'0'*64}
    else:value['replication']['complete_domains']=143
    additional.write_text(json.dumps(value))
    with pytest.raises(ValueError,match=match):
        packaging.pack(tmp_path,frozen,tmp_path/'bad',analysis_manifest=main,additional_analysis_manifest=additional)
    assert not (tmp_path/'bad/release_parts.json').exists()


def test_source_change_after_initial_validation_cannot_be_archived_under_old_hash(tmp_path,monkeypatch):
    frozen,main,additional=fixture(tmp_path)
    original_verify=packaging.verify_manifest
    def concurrent_edit(repo,path):
        value=original_verify(repo,path)
        (tmp_path/'measurement.txt').write_text('mutated!')
        return value
    monkeypatch.setattr(packaging,'verify_manifest',concurrent_edit)
    with pytest.raises(ValueError,match='changed during read'):
        packaging.pack(tmp_path,frozen,tmp_path/'changed')
    assert not (tmp_path/'changed/release_parts.json').exists()


def test_zero_part_limit_and_reserved_manifest_path_are_rejected(tmp_path):
    frozen,main,additional=fixture(tmp_path)
    with pytest.raises(ValueError,match='positive integer'):
        packaging.pack(tmp_path,frozen,tmp_path/'zero',part_bytes=0)
    value=json.loads(frozen.read_text())
    reserved=tmp_path/'frozen_manifest.json';reserved.write_text('conflicting research input')
    value['files']['frozen_manifest.json']=sha(reserved)
    frozen.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='reserved release manifest'):
        packaging.pack(tmp_path,frozen,tmp_path/'reserved')


def test_additional_package_cannot_drop_original_inputs_even_with_coherent_new_hashes(tmp_path):
    frozen,main,additional=fixture(tmp_path)
    hidden=tmp_path/'hidden.txt';hidden.write_text('original study input omitted by a bad combined manifest')
    original=tmp_path/'original_freeze.json'
    value=json.loads(original.read_text());value['files']['hidden.txt']=sha(hidden)
    original.write_text(json.dumps(value))
    combined=json.loads(frozen.read_text())
    combined['files']['original_freeze.json']=combined['original_freeze_sha256']=sha(original)
    frozen.write_text(json.dumps(combined))
    analysis=json.loads(additional.read_text());analysis['combined_freeze_sha256']=sha(frozen)
    additional.write_text(json.dumps(analysis))
    with pytest.raises(ValueError,match='Original-study frozen inputs are missing'):
        packaging.pack(tmp_path,frozen,tmp_path/'missing-original',additional_analysis_manifest=additional)
