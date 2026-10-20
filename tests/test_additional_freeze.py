import pytest

pytest.importorskip('torch')
from scripts.freeze_additional_research import close_replay, freeze


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
