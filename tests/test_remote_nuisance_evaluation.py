import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.remote_nuisance_evaluation import evaluation_plan, verify_input


def test_plan_covers_exact_frozen_140_cells():
    repo = Path(__file__).resolve().parents[1]
    rows = evaluation_plan(repo)
    assert len(rows) == 140
    assert {condition: sum(row['training_condition'] == condition for row in rows)
            for condition in ['iid', 'concentrated', 'spurious', 'unseen']} == {
                'iid': 35, 'concentrated': 35, 'spurious': 35, 'unseen': 35}
    shards = [{r['record']['run_id'] for r in rows if int(r['record']['run_id'], 16) % 4 == shard}
              for shard in range(4)]
    assert sum(map(len, shards)) == len(set.union(*shards)) == 140


def fixture_archive(tmp_path, damaged=False):
    rid = '0'*16
    config = {'seed': 1}
    entries = {'config.json': json.dumps(config).encode(),
               'summary.json': json.dumps({'config': config, 'run_id': rid}).encode(),
               'final.pt': b'trained checkpoint'}
    path = tmp_path/'input.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for name, value in entries.items():
            payload = b'corrupt checkpoint' if damaged and name == 'final.pt' else value
            member = tarfile.TarInfo(rid+'/'+name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    record = {'run_id': rid, 'config': config,
              'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'files': {name: hashlib.sha256(value).hexdigest() for name, value in entries.items()}}
    return path, record


def test_verified_inputs_preserve_actual_checkpoint_bytes(tmp_path):
    archive, record = fixture_archive(tmp_path)
    directory = verify_input(archive, record, tmp_path/'restored')
    assert (directory/'final.pt').read_bytes() == b'trained checkpoint'


def test_corrupt_checkpoint_rejected_before_any_extraction(tmp_path):
    archive, record = fixture_archive(tmp_path, damaged=True)
    with pytest.raises(ValueError, match='member checksum'):
        verify_input(archive, record, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
