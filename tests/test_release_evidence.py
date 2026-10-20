"""Integrity regressions for bounded scientific-evidence recovery."""
import gzip
import hashlib
import io
import json
import tarfile

import pytest

from scripts.recover_release_evidence import publish, verify_stream


def packed(entries):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return gzip.compress(raw.getvalue(), mtime=0)


def test_all_members_hashed_while_only_json_retained():
    entries = [('run/config.json', b'{"seed": 1}'), ('run/final.pt', b'binary checkpoint')]
    archive = packed(entries)
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in entries}
    kept, observed, count = verify_stream(io.BytesIO(archive), hashlib.sha256(archive).hexdigest(),
                                           expected, lambda name: name.endswith('.json'))
    assert kept == {'run/config.json': entries[0][1]}
    assert observed == expected and count == len(archive)


def test_discarded_binary_corruption_still_rejected():
    entries = [('run/config.json', b'{}'), ('run/final.pt', b'wrong bytes')]
    archive = packed(entries)
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in entries}
    expected['run/final.pt'] = '0'*64
    with pytest.raises(ValueError, match='Member checksum'):
        verify_stream(io.BytesIO(archive), hashlib.sha256(archive).hexdigest(), expected,
                      lambda name: name.endswith('.json'))


@pytest.mark.parametrize('entries,expected,match', [
    ([('../outside.json', b'{}')], {'../outside.json': None}, 'Unsafe'),
    ([('one.json', b'{}'), ('one.json', b'{}')], {'one.json': None}, 'Duplicate'),
    ([('one.json', b'{}')], {'one.json': None, 'missing.json': None}, 'incomplete'),
    ([('one.json', b'{"value": NaN}')], {'one.json': None}, 'Nonfinite'),
])
def test_malformed_archive_fails_closed(entries, expected, match):
    archive = packed(entries)
    with pytest.raises(ValueError, match=match):
        verify_stream(io.BytesIO(archive), hashlib.sha256(archive).hexdigest(), expected,
                      lambda name: name.endswith('.json'))


def test_archive_digest_checked_separately():
    archive = packed([('one.json', b'{}')])
    with pytest.raises(ValueError, match='Archive checksum'):
        verify_stream(io.BytesIO(archive), '0'*64, {'one.json': None}, lambda name: True)


def test_conflicting_evidence_is_never_replaced(tmp_path):
    publish(tmp_path, {'runs/config.json': b'{"seed": 1}'})
    with pytest.raises(ValueError, match='Conflicting'):
        publish(tmp_path, {'runs/config.json': b'{"seed": 2}'})
    assert json.loads((tmp_path/'runs/config.json').read_text()) == {'seed': 1}
