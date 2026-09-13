import hashlib
import json
import pytest
import gzip
import io
import tarfile
from scripts.freeze_results import readiness,verify_manifest
from scripts.pack_release import pack


def test_missing_studies_cannot_be_frozen(tmp_path):
    report,_=readiness(tmp_path)
    assert not report['ready']
    assert any(p.get('study')=='design' for p in report['problems'])


def test_frozen_inputs_detect_changes_and_directory_escape(tmp_path):
    source=tmp_path/'result.txt'
    source.write_text('measured result')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=tmp_path/'frozen.json'
    value=dict(schema='feature-topology.frozen-results.v1',ready=True,files={'result.txt':digest})
    manifest.write_text(json.dumps(value))
    assert verify_manifest(tmp_path,manifest)['ready']
    source.write_text('changed result')
    with pytest.raises(ValueError,match='missing or changed'):
        verify_manifest(tmp_path,manifest)
    value['files']={'../outside.txt':digest}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='missing or changed'):
        verify_manifest(tmp_path,manifest)


def test_release_parts_reconstruct_and_are_reproducible(tmp_path):
    source=tmp_path/'measurement.txt'
    source.write_text('actual measured output\n')
    value=dict(schema='feature-topology.frozen-results.v1',ready=True,
               files={'measurement.txt':hashlib.sha256(source.read_bytes()).hexdigest()})
    manifest=tmp_path/'frozen.json'
    manifest.write_text(json.dumps(value))
    first=pack(tmp_path,manifest,tmp_path/'first',part_bytes=100)
    second=pack(tmp_path,manifest,tmp_path/'second',part_bytes=100)
    assert first['parts']==second['parts']
    assert all(p['bytes']<=100 for p in first['parts'])
    payload=b''.join((tmp_path/'first'/p['file']).read_bytes() for p in first['parts'])
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)),mode='r:') as archive:
        assert archive.extractfile('feature-topology/measurement.txt').read()==source.read_bytes()
