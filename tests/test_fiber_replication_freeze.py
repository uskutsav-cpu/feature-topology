import json

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts import freeze_additional_research as gate
from scripts.collect_fiber_replication import collect
from scripts.recover_release_evidence import digest
from tests.test_fiber_replication_collection import fixture


def frozen_fixture(tmp_path,monkeypatch):
    repo,artifacts=fixture(tmp_path,monkeypatch)
    root=repo/'results/fiber_replication_verification_v1'
    collect(repo,artifacts,root)
    witnesses=json.loads((repo/'results/fiber_replication_screens_v1/collection.json').read_text())
    index=json.loads((repo/'configs/completion/primary_inputs_v3.json').read_text())
    arrays=repo/'results/continuous_fiber_inputs/analysis_data.npz'
    arrays.parent.mkdir();arrays.write_bytes(b'metadata-only fixture; actual numerical inputs are not represented')
    expected={str(arrays):index['analysis_data']['sha256']}
    original={}
    for row in witnesses['records']:
        checkpoint=repo/'results/main/runs'/row['run_id']/row['checkpoint']
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.write_bytes(b'metadata-only checkpoint fixture')
        expected[str(checkpoint)]=row['checkpoint_sha256']
        original[checkpoint.relative_to(repo).as_posix()]=row['checkpoint_sha256']
    monkeypatch.setattr(gate,'digest',lambda path:expected.get(str(path),digest(path)))
    (repo/'scripts/collect_fiber_replication.py').write_text('# test-only collector placeholder')
    return repo,original,arrays


def test_gate_requires_all_physical_replication_checkpoints_without_requiring_positive_outcomes(tmp_path,monkeypatch):
    repo,original,arrays=frozen_fixture(tmp_path,monkeypatch)
    paths,summary=gate.replication_paths(repo,original)
    assert summary['complete_domains']==144 and summary['complete_models']==36
    assert summary['joint_certificates']==7 and summary['all_four_joint_models']==0
    assert summary['failed_fixed_witnesses']==29
    assert summary['source_bound_standalone_replays_checked'] is True
    assert len([p for p in paths if p.suffix=='.pt'])==36
    original.pop(next(iter(original)))
    with pytest.raises(ValueError,match='outside the original full freeze'):
        gate.replication_paths(repo,original)


def test_replication_gate_requires_the_physical_embedding_asset(tmp_path,monkeypatch):
    repo,original,arrays=frozen_fixture(tmp_path,monkeypatch)
    arrays.unlink()
    with pytest.raises(ValueError,match='embedding is missing or changed'):
        gate.replication_paths(repo,original)
