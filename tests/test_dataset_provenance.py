import hashlib
import json

import pytest
from scripts import dataset_provenance as provenance


def test_dataset_provenance_binds_bytes_and_source_identity(tmp_path,monkeypatch):
    source=tmp_path/'data'/'example.bin';source.parent.mkdir(parents=True)
    source.write_bytes(b'official bytes')
    identity=provenance.git_blob_digest(source)
    monkeypatch.setattr(provenance,'EXPECTED_SOURCES',
                        {'data/example.bin':('git_blob_sha1',identity)})
    value=provenance.create_provenance(tmp_path)
    manifest=tmp_path/'provenance.json';manifest.write_text(json.dumps(value))
    assert provenance.verify_provenance(tmp_path,manifest)==[manifest,source]
    source.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='missing or changed'):
        provenance.verify_provenance(tmp_path,manifest)


def test_dataset_provenance_rejects_declared_identity_substitution(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'bytes')
    identity=hashlib.md5(source.read_bytes()).hexdigest()
    monkeypatch.setattr(provenance,'EXPECTED_SOURCES',{'source':('md5',identity)})
    value=provenance.create_provenance(tmp_path)
    value['sources'][0]['source_identity']['value']='forged'
    manifest=tmp_path/'provenance.json';manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='missing or changed'):
        provenance.verify_provenance(tmp_path,manifest)
