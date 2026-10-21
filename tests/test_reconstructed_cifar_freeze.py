import hashlib
import json

import pytest

from scripts.reconstruct_cifar_provenance import validate_reconstruction as reconstructed_source_paths


def fixture(tmp_path):
    raw=tmp_path/'status.json';raw.write_text('{"source":"synthetic metadata fixture"}\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    observation=dict(schema='feature-topology.cifar-provenance-reconstruction.v1',dataset='CIFAR10',
        source_commit='a'*40,result_tag='synthetic',complete_source_ledger=True,active_workflows=[],
        unresolved_jobs=[],terminal_attempts=[],excluded_runs=[],evidence_sha256={'status.json':digest(raw)})
    path=tmp_path/'observation.json';path.write_text(json.dumps(observation))
    state=dict(schema='feature-topology.remote-cifar-queue.v1',source_commit='a'*40,attempts=[],
        reconstruction=dict(method=observation['schema'],result_tag='synthetic',evidence_sha256=observation['evidence_sha256'],
            excluded_runs=[],observation_path='observation.json',observation_sha256=digest(path)))
    return state,raw,path


def test_reconstructed_source_binds_every_retained_api_and_status_byte(tmp_path):
    state,raw,path=fixture(tmp_path)
    assert set(reconstructed_source_paths(tmp_path,state,'CIFAR10'))=={raw,path}
    raw.write_text('changed')
    with pytest.raises(ValueError,match='missing, changed'):reconstructed_source_paths(tmp_path,state,'CIFAR10')


def test_reconstruction_cannot_promote_an_active_observation_by_updating_its_checksum(tmp_path):
    state,raw,path=fixture(tmp_path)
    value=json.loads(path.read_text());value['active_workflows']=['123'];path.write_text(json.dumps(value))
    state['reconstruction']['observation_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='contradicts'):reconstructed_source_paths(tmp_path,state,'CIFAR10')
