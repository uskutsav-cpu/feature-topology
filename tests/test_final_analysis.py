import hashlib
import json

import pytest

from scripts.final_analysis import verify_consumed_inputs, verify_frozen_paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_consumed_analysis_inputs_must_be_frozen(tmp_path):
    source=tmp_path/'results/run/metrics/profile/final.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"measured": true}\n')
    frozen={'files':{'results/run/metrics/profile/final.json':digest(source)}}
    input_hashes=tmp_path/'input_hashes.json'
    input_hashes.write_text(json.dumps({str(source):digest(source)}))
    assert verify_consumed_inputs(tmp_path,frozen,input_hashes)=={str(source):digest(source)}

    unfrozen=tmp_path/'results/run/metrics/profile/extra.json'
    unfrozen.write_text('{"extra": true}\n')
    input_hashes.write_text(json.dumps({str(unfrozen):digest(unfrozen)}))
    with pytest.raises(ValueError,match='absent from frozen manifest'):
        verify_consumed_inputs(tmp_path,frozen,input_hashes)


def test_consumed_analysis_inputs_reject_changed_or_external_files(tmp_path):
    source=tmp_path/'metric.json'
    source.write_text('{"value": 1}\n')
    expected=digest(source)
    frozen={'files':{'metric.json':expected}}
    input_hashes=tmp_path/'input_hashes.json'
    source.write_text('{"value": 2}\n')
    input_hashes.write_text(json.dumps({str(source):expected}))
    with pytest.raises(ValueError,match='changed frozen input'):
        verify_consumed_inputs(tmp_path,frozen,input_hashes)

    external=tmp_path.parent/'external-analysis-input.json'
    external.write_text('{}\n')
    input_hashes.write_text(json.dumps({str(external):digest(external)}))
    with pytest.raises(ValueError,match='escapes repository'):
        verify_consumed_inputs(tmp_path,frozen,input_hashes)


def test_direct_analysis_inputs_must_be_frozen_and_unchanged(tmp_path):
    source=tmp_path/'results/image/runs/run/summary.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"status": "complete"}\n')
    expected=digest(source)
    frozen={'files':{'results/image/runs/run/summary.json':expected}}
    assert verify_frozen_paths(tmp_path,frozen,[source])=={
        'results/image/runs/run/summary.json':expected}

    extra=source.parent/'metrics.json'
    extra.write_text('{}\n')
    with pytest.raises(ValueError,match='absent from frozen manifest'):
        verify_frozen_paths(tmp_path,frozen,[extra])

    source.write_text('{"status": "changed"}\n')
    with pytest.raises(ValueError,match='changed frozen input'):
        verify_frozen_paths(tmp_path,frozen,[source])
