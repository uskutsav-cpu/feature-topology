import gzip
import hashlib
import io
import json
import tarfile

import pytest

from scripts.verify_release_package import audit, release_opener


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


SOURCE = b'original frozen source\n'
FILES = {'source.txt': sha(SOURCE)}
FREEZE = json.dumps(dict(schema='feature-topology.frozen-results.v1', ready=True,
                         files=FILES)).encode()


def package(members=None, files=None, frozen_payload=FREEZE):
    members = [('feature-topology/source.txt', SOURCE)] if members is None else members
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, payload in [*members, ('feature-topology/frozen_manifest.json', frozen_payload)]:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.SYMTYPE
                info.linkname = '../outside'
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    compressed = gzip.compress(stream.getvalue(), mtime=0)
    parts, records = {}, []
    # Cross gzip, tar-header, member and trailer boundaries with tiny parts.
    for number, start in enumerate(range(0, len(compressed), 7)):
        name = f'feature-topology.tar.gz.part{number:03d}'
        payload = compressed[start:start + 7]
        parts[name] = payload
        records.append(dict(file=name, bytes=len(payload), sha256=sha(payload)))
    index = dict(schema='feature-topology.release-parts.v1', parts=records,
                 files=FILES if files is None else files, frozen_manifest_sha256=sha(FREEZE))
    payload = json.dumps(index).encode()
    return payload, sha(payload), sha(FREEZE), lambda row: io.BytesIO(parts[row['file']]), parts


def test_stream_audit_checks_every_tiny_part_and_member():
    payload, index_sha, freeze_sha, opener, parts = package()
    report = audit(payload, index_sha, freeze_sha, opener)
    assert report['verified'] and report['parts'] == len(parts)
    assert report['members'] == 2 and report['frozen_inputs'] == 1
    assert report['compressed_bytes'] == sum(map(len, parts.values()))


def test_last_part_bytes_are_checked_after_tar_end_marker():
    payload, index_sha, freeze_sha, opener, parts = package()
    name = list(parts)[-1]
    parts[name] += b'undocumented trailing bytes'
    with pytest.raises(ValueError, match='part'):
        audit(payload, index_sha, freeze_sha, opener)


def test_trusted_index_digest_cannot_be_replaced():
    payload, _, freeze_sha, opener, _ = package()
    with pytest.raises(ValueError, match='Trusted JSON digest'):
        audit(payload, '0' * 64, freeze_sha, opener)


def test_changed_member_is_rejected_even_with_new_part_and_index_hashes():
    args = package(members=[('feature-topology/source.txt', b'changed')])
    with pytest.raises(ValueError, match='member digest'):
        audit(*args[:4])


def test_equal_json_with_different_manifest_bytes_is_rejected():
    formatted = json.dumps(json.loads(FREEZE), indent=2).encode()
    args = package(frozen_payload=formatted)
    with pytest.raises(ValueError, match='member digest'):
        audit(*args[:4])


def test_package_cannot_omit_a_member_listed_in_its_index():
    args = package(members=[])
    with pytest.raises(ValueError, match='incomplete'):
        audit(*args[:4])


def test_index_and_archive_cannot_omit_an_input_of_the_trusted_freeze():
    args = package(members=[('feature-topology/other.txt', b'other')], files={'other.txt': sha(b'other')})
    with pytest.raises(ValueError, match='frozen input is missing'):
        audit(*args[:4])


@pytest.mark.parametrize('members', [
    [('feature-topology/../outside', SOURCE)],
    [('feature-topology/source.txt', None)],
    [('feature-topology/source.txt', SOURCE), ('feature-topology/source.txt', SOURCE)],
])
def test_unsafe_nonregular_and_duplicate_members_fail(members):
    args = package(members=members)
    with pytest.raises(ValueError):
        audit(*args[:4])


def test_remote_metadata_must_match_the_trusted_part_digest(monkeypatch):
    payload, _, _, _, _ = package()
    index = json.loads(payload)
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: json.dumps({'assets': []}).encode())
    with pytest.raises(ValueError, match='Remote release part identity'):
        release_opener('example/repo', 'research-v1', index)


def test_audit_the_actual_packer_output(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('flint')
    from scripts.pack_release import pack
    (tmp_path / 'source.txt').write_bytes(SOURCE)
    manifest = tmp_path / 'freeze.json'
    manifest.write_bytes(FREEZE)
    destination = tmp_path / 'package'
    index = pack(tmp_path, manifest, destination, part_bytes=17)
    payload = (destination / 'release_parts.json').read_bytes()
    report = audit(payload, sha(payload), sha(FREEZE),
                   lambda row: (destination / row['file']).open('rb'))
    assert report['verified'] and report['parts'] == len(index['parts'])
