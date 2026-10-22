import json

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts import collect_fixed_fiber_results as retention_collector
from scripts import collect_nuisance_decision_results as collector
from scripts.recover_release_evidence import digest
from src.metrics.nuisance_decision_verifier import SCOPE
from tests.test_fixed_fiber_collection import fixture as retention_fixture


def fixture(tmp_path, monkeypatch):
    repo, nested = retention_fixture(tmp_path, monkeypatch)
    witnesses = repo/'results/continuous_fiber_feasibility'
    for path in witnesses.glob('model-*.json'):
        witness = json.loads(path.read_text())
        witness['selected_step'] = 64
        path.write_text(json.dumps(witness))
    for path in nested.rglob('model-*.json'):
        if path.name.endswith('.replay.json'):
            continue
        result = json.loads(path.read_text())
        rid = result['binding']['run_id']
        result['binding']['witness_sha256'] = digest(witnesses/(rid+'.json'))
        path.write_text(json.dumps(result))
        receipt_path = path.with_name(path.stem+'.replay.json')
        receipt = json.loads(receipt_path.read_text())
        receipt['result_sha256'] = digest(path)
        receipt_path.write_text(json.dumps(receipt))
    root = repo/'results/fixed_fiber_verification_v4'
    retained = retention_collector.collect(repo, nested, root)
    source = repo/'verifier-source.py'
    monkeypatch.setattr(collector, 'source_files', lambda root:[source])
    protocol = repo/'configs/nuisance_decision_verifier_v1.json'
    protocol.write_text(json.dumps({'budget':{'maximum_boxes':20000, 'maximum_depth':18}}))
    artifacts = tmp_path/'decisions'
    for prior in retained['records']:
        rid, theta = prior['run_id'], prior['theta_fraction'][0]
        stem = rid+f'-theta-{theta}-8'
        retention = json.loads((root/(stem+'.json')).read_text())
        binding = dict(retention['binding'])
        binding.pop('witness_sha256')
        binding.update(selected_step=64, target_class=theta//2, protocol_sha256=digest(protocol),
            retention_certificate_sha256=digest(root/(stem+'.json')),
            retention_replay_sha256=digest(root/(stem+'.replay.json')))
        positive, negative = theta in [1,3], theta==5
        status = ('certified_correct_class_circle' if positive else
                  'target_class_violation' if negative else 'inconclusive')
        result = dict(schema='feature-topology.nuisance-decision-verification.v1', binding=binding,
            theta_fraction=[theta,8], target_class=theta//2, scope=SCOPE, status=status,
            decision_certificate=positive, counterexample_certificate=negative, checked_boxes=1,
            arithmetic={'python_flint':'0.9.0', 'precision_bits':128},
            host={'source_commit':'a'*40, 'workflow_run':'123'})
        if positive:
            result.update(leaf_codes=[''], scaled_margin_lower_rational={
                'numerator':'100', 'denominator':str(1<<64)})
        elif negative:
            result.update(violation_code='', violating_class=0,
                point_margin_upper_rational={'numerator':'-1', 'denominator':str(1<<64)})
        directory = artifacts/f'gamma-{prior["gamma"]}'/f'theta-{theta}'
        directory.mkdir(parents=True)
        path = directory/(stem+'.json')
        path.write_text(json.dumps(result))
        receipt = dict(schema='feature-topology.nuisance-decision-replay.v1',run_id=rid,
            theta_fraction=[theta,8], target_class=theta//2, result_sha256=digest(path),
            decision_certificate=positive, counterexample_certificate=negative,
            standalone_replay_passed=positive or negative,
            status='passed' if positive or negative else 'not_a_certificate',
            scope='Separate-process execution of the same Arb verifier, not an independent implementation or Lean proof')
        path.with_name(path.stem+'.replay.json').write_text(json.dumps(receipt))
    return repo, artifacts


def test_mixed_outcomes_are_preserved_and_flat_collection_is_stable(tmp_path, monkeypatch):
    repo, artifacts = fixture(tmp_path, monkeypatch)
    output = tmp_path/'flat'
    manifest = collector.collect(repo, artifacts, output, '123', 'a'*40)
    assert manifest['decision_certificates']==8
    assert manifest['counterexample_certificates']==4
    assert manifest['inconclusive']==4
    assert manifest['joint_retention_and_correct_decision']==8
    assert collector.collect(repo, output)==manifest
    with pytest.raises(ValueError, match='workflow run'):
        collector.collect(repo, output, workflow_run='999')


@pytest.mark.parametrize('damage,match', [
    ('retention', 'retention or scope'), ('flags', 'status or flags'),
    ('cover', 'unresolved'), ('negative', 'negative target-class margin'),
    ('receipt', 'replay receipt'), ('source', 'source commit or workflow run')])
def test_tampered_claims_rejected_even_after_result_hash_is_updated(tmp_path, monkeypatch, damage, match):
    repo, artifacts = fixture(tmp_path, monkeypatch)
    theta = 5 if damage=='negative' else 1
    path = next(artifacts.rglob(f'model-*-theta-{theta}-8.json'))
    result = json.loads(path.read_text())
    receipt_path = path.with_name(path.stem+'.replay.json')
    receipt = json.loads(receipt_path.read_text())
    if damage=='retention':result['binding']['retention_certificate_sha256']='0'*64
    elif damage=='flags':result['decision_certificate']=1
    elif damage=='cover':result['leaf_codes']=['0']
    elif damage=='negative':result['point_margin_upper_rational']['numerator']='1'
    elif damage=='receipt':receipt['counterexample_certificate']=True
    elif damage=='source':result['host']['source_commit']='unbound'
    path.write_text(json.dumps(result))
    receipt['result_sha256']=digest(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        collector.collect(repo, artifacts)


def test_missing_and_duplicate_domains_cannot_complete_collection(tmp_path, monkeypatch):
    repo, artifacts = fixture(tmp_path, monkeypatch)
    path = next(artifacts.rglob('model-*-theta-1-8.json'))
    directory = artifacts/'duplicate'
    directory.mkdir()
    duplicate = directory/path.name
    duplicate.write_bytes(path.read_bytes())
    duplicate.with_name(duplicate.stem+'.replay.json').write_bytes(
        path.with_name(path.stem+'.replay.json').read_bytes())
    with pytest.raises(ValueError, match='duplicate'):
        collector.collect(repo, artifacts)
    duplicate.unlink()
    duplicate.with_name(duplicate.stem+'.replay.json').unlink()
    path.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        collector.collect(repo, artifacts)
