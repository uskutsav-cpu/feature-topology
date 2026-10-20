import json
from pathlib import Path

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts import collect_fixed_fiber_results as collector
from scripts.recover_release_evidence import digest
from src.metrics.rational_fixed_fiber_verifier import verify_circle
from tests.test_fixed_fiber_verifier import network


def fixture(tmp_path, monkeypatch):
    repo, artifacts = tmp_path/'repo', tmp_path/'nested'
    witnesses = repo/'results/continuous_fiber_feasibility'
    index_path = repo/'configs/completion/primary_inputs_v3.json'
    witnesses.mkdir(parents=True)
    index_path.parent.mkdir(parents=True)
    protocol = repo/'configs/fixed_fiber_verifier_v4.json'
    protocol.write_text('{}')
    source = repo/'verifier-source.py'
    source.write_text('# controlled collection fixture\n')
    monkeypatch.setattr(collector, 'source_files', lambda root: [source])
    rows = [dict(run_id=f'model-{3-i}', gamma=gamma, seed=0,
                 archive_sha256='2'*64, files={'selected.pt':'1'*64})
            for i,gamma in enumerate([.03125,.5,1.,8.])]
    index = {'runs':rows, 'analysis_data':{'sha256':'3'*64}}
    index_path.write_text(json.dumps(index))
    fixed = []
    for row in rows:
        witness = dict(run_id=row['run_id'],gamma=row['gamma'],seed=0,
            empirical_screen_passed=True,checkpoint='selected.pt',checkpoint_sha256='1'*64,
            readout_hex=[[float(value).hex() for value in line] for line in [[1,0],[0,1]]])
        fixed.append(witness)
        (witnesses/(row['run_id']+'.json')).write_text(json.dumps(witness))
    (witnesses/'manifest.json').write_text(json.dumps({'runs':fixed}))
    base = verify_circle(network(), maximum_boxes=2000, seconds=60)
    assert base['continuous_certificate']
    for row in rows:
        for theta in [1,3,5,7]:
            # Nested order is deliberately different from flattened filename order.
            directory = artifacts/f"gamma-{row['gamma']}"/f'theta-{theta}'
            directory.mkdir(parents=True)
            stem = row['run_id']+f'-theta-{theta}-8'
            binding = dict(run_id=row['run_id'],gamma=row['gamma'],seed=0,
                theta_fraction=[theta,8],checkpoint_sha256='1'*64,input_archive_sha256='2'*64,
                input_index_sha256=digest(index_path),analysis_data_sha256='3'*64,
                witness_sha256=digest(witnesses/(row['run_id']+'.json')),
                protocol_sha256=digest(protocol),source_sha256={'verifier-source.py':digest(source)})
            result = dict(base,schema='feature-topology.fixed-fiber-verification.v4',binding=binding,
                theta_fraction=[theta,8],arithmetic={'python_flint':'0.9.0','precision_bits':128})
            path = directory/(stem+'.json')
            path.write_text(json.dumps(result))
            receipt = dict(schema='feature-topology.fixed-fiber-replay.v4',
                run_id=row['run_id'],theta_fraction=[theta,8],result_sha256=digest(path),
                positive_certificate=True,standalone_replay_passed=True,status='passed')
            (directory/(stem+'.replay.json')).write_text(json.dumps(receipt))
    return repo, artifacts


def test_collection_is_stable_when_hosted_artifacts_are_flattened(tmp_path, monkeypatch):
    repo, nested = fixture(tmp_path, monkeypatch)
    output = tmp_path/'flat'
    stored = collector.collect(repo,nested,output)
    assert stored['positive_certificates']==16
    assert stored['trained_models']==4 and stored['training_seeds']==[0]
    assert collector.collect(repo,output)==stored
    assert [row['gamma'] for row in stored['records']]==[g for g in [.03125,.5,1.,8.] for _ in range(4)]


def test_collection_rejects_an_inflated_bound_even_if_the_receipt_hash_is_updated(tmp_path, monkeypatch):
    repo, nested = fixture(tmp_path, monkeypatch)
    output = tmp_path/'flat'
    collector.collect(repo,nested,output)
    path = next(p for p in output.glob('model-*.json') if not p.name.endswith('.replay.json'))
    result = json.loads(path.read_text())
    result['chord_lower_factor_rational']['numerator']=str(100<<64)
    path.write_text(json.dumps(result))
    receipt_path = path.with_name(path.stem+'.replay.json')
    receipt = json.loads(receipt_path.read_text())
    receipt['result_sha256']=digest(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Separation factor differs'):
        collector.collect(repo,output)
