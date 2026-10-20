import hashlib
import json
import pytest

pytest.importorskip('torch')
from scripts.freeze_additional_research import close_replay, fixed_fiber_paths, freeze


def test_additional_gate_cannot_bypass_the_original_full_freeze(tmp_path):
    manifest = tmp_path/'original.json'
    manifest.write_text('{"schema":"feature-topology.frozen-results.v1","ready":false,"files":{}}')
    with pytest.raises(ValueError, match='Not a completed frozen-result manifest'):
        freeze(tmp_path, manifest, tmp_path/'evidence', tmp_path/'combined.json')
    assert not (tmp_path/'combined.json').exists()


def test_replay_tolerance_allows_roundoff_but_rejects_changed_conclusions():
    close_replay({'score': 0.0001, 'supported': False},
                 {'score': 0.00010000000000001, 'supported': False})
    with pytest.raises(ValueError, match='numeric difference'):
        close_replay({'score': .1}, {'score': .11})
    with pytest.raises(ValueError, match='value differs'):
        close_replay({'supported': False}, {'supported': True})


def test_original_freeze_outside_repository_cannot_become_unavailable_release_provenance(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    source=repo/'measurement.txt';source.write_text('original study input')
    original=tmp_path/'outside.json'
    original.write_text(json.dumps(dict(schema='feature-topology.frozen-results.v1',ready=True,
        files={'measurement.txt':hashlib.sha256(source.read_bytes()).hexdigest()})))
    with pytest.raises(ValueError,match='Original freeze must be inside'):
        freeze(repo,original,repo/'evidence',repo/'combined.json')


def test_fiber_gate_requires_the_actual_embedding_input(tmp_path):
    pytest.importorskip('flint')
    index = tmp_path/'configs/completion/primary_inputs_v3.json'
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'analysis_data': {'sha256': '0'*64}}))
    with pytest.raises(ValueError, match='embedding input is missing or changed'):
        fixed_fiber_paths(tmp_path, {})


def test_fiber_gate_rejects_a_changed_frozen_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip('flint')
    from scripts import collect_fixed_fiber_results
    arrays = tmp_path/'results/continuous_fiber_inputs/analysis_data.npz'
    arrays.parent.mkdir(parents=True)
    arrays.write_bytes(b'fixed embedding input')
    index = tmp_path/'configs/completion/primary_inputs_v3.json'
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'analysis_data': {'sha256': hashlib.sha256(arrays.read_bytes()).hexdigest()}}))
    root = tmp_path/'results/fixed_fiber_verification_v4'
    root.mkdir()
    (root/'collection.json').write_text('{}')
    monkeypatch.setattr(collect_fixed_fiber_results, 'collect', lambda *args: {})
    witness = tmp_path/'results/continuous_fiber_feasibility/manifest.json'
    witness.parent.mkdir()
    expected = hashlib.sha256(b'original checkpoint').hexdigest()
    witness.write_text(json.dumps({'runs': [dict(empirical_screen_passed=True,
        run_id='model', checkpoint='selected.pt', checkpoint_sha256=expected, source_sha256={})]}))
    checkpoint = tmp_path/'results/main/runs/model/selected.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='outside the original full freeze'):
        fixed_fiber_paths(tmp_path, {'results/main/runs/model/selected.pt': expected})
