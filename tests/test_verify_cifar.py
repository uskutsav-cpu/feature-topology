import json

import numpy as np
import pytest

import scripts.verify_cifar as verifier
from scripts.audit_cifar_replay import replay_comparison
from scripts.verify_cifar import cached_replay,validate_dataset


def test_cached_replay_requires_every_artifact_binding():
    binding = {
        "dataset": "CIFAR10", "run_id": "a" * 16,
        "checkpoint_sha256": "b" * 64, "metrics_sha256": "c" * 64,
        "representations_sha256": "d" * 64,
        "saved_test": {"loss": 1.0, "accuracy": .5},
    }
    key = "CIFAR10/" + "a" * 16
    cache = {"records": {key: {**binding,
                                "replayed_test": {"loss": 1.0, "accuracy": .5}}}}
    assert cached_replay(cache, key, binding) == {"loss": 1.0, "accuracy": .5}
    changed = {**binding, "metrics_sha256": "e" * 64}
    assert cached_replay(cache, key, changed) is None


def test_cached_replay_rejects_nonfinite_measurement():
    key = "CIFAR100/run"
    binding = {"dataset": "CIFAR100"}
    cache = {"records": {key: {**binding,
                                "replayed_test": {"loss": None, "accuracy": .5}}}}
    with pytest.raises(ValueError, match="Non-finite cached"):
        cached_replay(cache, key, binding)


def test_incremental_validation_reports_missing_without_claiming_completion(tmp_path,monkeypatch):
    root=tmp_path/'results/cifar10'
    root.mkdir(parents=True)
    (root/'gamma_to_lr.json').write_text(json.dumps({
        'selection':{str(gamma):{'lr':.1} for gamma in verifier.GAMMAS},
        'calibration_steps':2000,
    }))
    monkeypatch.setattr(verifier,'load_data',lambda *args,**kwargs:{
        'train':(np.empty((45000,0)),None),
        'validation':(np.empty((5000,0)),None),
        'test':(np.empty((10000,0)),None),
    })
    cache={'records':{}}
    result=validate_dataset(tmp_path,'CIFAR10',cache,tmp_path/'cache.json',
                            allow_missing=True)
    assert result['expected_runs']==35
    assert result['validated_runs']==0
    assert len(result['missing'])==35
    with pytest.raises(ValueError,match='Missing CIFAR run'):
        validate_dataset(tmp_path,'CIFAR10',cache,tmp_path/'cache.json')


def test_hosted_replay_comparison_keeps_discrete_and_float_checks_separate():
    saved={'loss':2.5,'accuracy':.4266}
    same=replay_comparison(saved,{'loss':2.5000001,'accuracy':.4266},10000)
    assert same['accuracy_exact'] and same['loss_within_1e-4']
    shifted=replay_comparison(saved,{'loss':2.5000001,'accuracy':.4265},10000)
    assert shifted['correct_count_delta']==-1
    assert not shifted['accuracy_exact']
