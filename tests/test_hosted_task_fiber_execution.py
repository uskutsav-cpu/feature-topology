import hashlib
import json
from pathlib import Path

import pytest

from scripts.collect_hosted_task_fibers import collect_hosted,expected_ids
from scripts.remote_task_preserving_completion import verify_pins


def test_hosted_shards_cover_210_models_exactly_once():
    repo=Path(__file__).resolve().parents[1]
    shards=[expected_ids(repo,i) for i in range(8)]
    assert len([rid for shard in shards for rid in shard])==210
    assert len(set(rid for shard in shards for rid in shard))==210
    assert all(int(rid,16)%8==i for i,shard in enumerate(shards) for rid in shard)


def test_partial_hosted_artifacts_cannot_become_a_complete_collection(tmp_path):
    repo=Path(__file__).resolve().parents[1];artifacts=tmp_path/'artifacts';artifacts.mkdir()
    with pytest.raises(ValueError,match='eight complete'):
        collect_hosted(repo,artifacts,tmp_path/'output','123','a'*40)
    assert not (tmp_path/'output').exists()


def test_protected_mathematics_must_match_recorded_bytes_before_execution(tmp_path):
    source=tmp_path/'protected.py';source.write_text('fixed calculation')
    replay=tmp_path/'scripts/replay_task_preserving_fibers_v2.py';replay.parent.mkdir();replay.write_text('fixed metadata comparison')
    plan=dict(shards=8,max_parallel=4,math_changed=False,statistics_changed=False,
        scientific_source_sha256={'protected.py':hashlib.sha256(source.read_bytes()).hexdigest()},
        replay_worker_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
    target=tmp_path/'configs/task_fiber_hosted_execution_v1.json';target.parent.mkdir();target.write_text(json.dumps(plan))
    verify_pins(tmp_path)
    source.write_text('changed calculation')
    with pytest.raises(ValueError,match='Protected scientific source changed'):verify_pins(tmp_path)
