import json

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts import freeze_additional_research as gate
from scripts.collect_nuisance_decision_results import collect
from scripts.recover_release_evidence import digest
from scripts.replay_nuisance_decision_collection_v2 import derived_claim
from tests.test_nuisance_decision_collection import fixture


def frozen_fixture(tmp_path, monkeypatch):
    repo, artifacts = fixture(tmp_path, monkeypatch)
    root = repo/'results/nuisance_decision_verification_v1'
    collection = collect(repo, artifacts, root)
    local = repo/'results/nuisance_decision_local_replays_v2'
    local.mkdir()
    worker = repo/'scripts/replay_nuisance_decision_collection_v2.py'
    worker.parent.mkdir()
    worker.write_text('# fixture worker\n')
    preflight = repo/'results/nuisance_decision_preflight_v1'
    preflight.mkdir()
    (preflight/'attempt.json').write_text('{}')
    (preflight/'failure_excerpt.log').write_text('retained source-format failure')
    failed_local = repo/'results/nuisance_decision_local_attempts_v1'
    failed_local.mkdir()
    (failed_local/'initial_margin_failure.json').write_text('{}')
    (failed_local/'initial_replayer.py').write_text('# retained initial local worker')
    original = {}
    for row in collection['records']:
        name = row['run_id']+f"-theta-{row['theta_fraction'][0]}-8.json"
        result = json.loads((root/name).read_text())
        checkpoint = repo/'results/main/runs'/row['run_id']/'selected.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'fixture checkpoint: its digest is stubbed, never a real certificate')
        original[checkpoint.relative_to(repo).as_posix()] = '1'*64
        certified = row['decision_certificate'] or row['counterexample_certificate']
        claim_path = local/(name[:-5]+'.local-claim.json')
        claim_path.write_text(json.dumps(derived_claim(result, row['result_sha256'])))
        receipt = dict(schema='feature-topology.nuisance-decision-local-replay.v2',
            run_id=row['run_id'], theta_fraction=row['theta_fraction'], target_class=row['target_class'],
            binding=result['binding'], result_sha256=row['result_sha256'],
            local_claim_sha256=digest(claim_path), hosted_numerical_margin_replay_claimed=False,
            decision_certificate=row['decision_certificate'], counterexample_certificate=row['counterexample_certificate'],
            replay_passed=certified, status='passed' if certified else 'not_a_certificate',
            arithmetic={'python_flint':'0.9.0', 'precision_bits':128},verifier_sha256=digest(worker),
            host={'system':'Test', 'machine':'fixture'})
        (local/name).write_text(json.dumps(receipt))
    monkeypatch.setattr(gate, 'digest', lambda path:'1'*64 if path.name=='selected.pt' else digest(path))
    return repo, original, local


def test_combined_gate_counts_counterexamples_separately_and_requires_original_checkpoints(tmp_path, monkeypatch):
    repo, original, local = frozen_fixture(tmp_path, monkeypatch)
    paths, result = gate.decision_paths(repo, original)
    assert result['decision_certificates']==8 and result['counterexample_certificates']==4
    assert result['joint_retention_and_correct_decision']==8
    assert result['additional_local_replays']==16 and len(paths)>=52
    with pytest.raises(ValueError, match='outside the original full freeze'):
        gate.decision_paths(repo, {})
    next(local.glob('*.json')).unlink()
    with pytest.raises(ValueError, match='incomplete'):
        gate.decision_paths(repo, original)


def test_local_counterexample_replay_cannot_be_relabeled_correct_class(tmp_path, monkeypatch):
    repo, original, local = frozen_fixture(tmp_path, monkeypatch)
    path = next(local.glob('*-theta-5-8.json'))
    receipt = json.loads(path.read_text())
    assert receipt['counterexample_certificate'] is True
    receipt.update(decision_certificate=True, counterexample_certificate=False)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='local classifier replay binding changed'):
        gate.decision_paths(repo, original)
