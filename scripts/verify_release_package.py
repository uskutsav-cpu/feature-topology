"""Stream-audit a multipart release against trusted index and freeze digests.

Reads one part at a time, hashes every compressed byte and archive member, and
does not extract checkpoints. Trusted digests must come from the release build,
not from the downloaded package being audited.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.request


def checked_json(payload, expected):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Trusted JSON digest mismatch')
    return json.loads(payload)


def safe_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or path.as_posix() != name or name == '.'):
        raise ValueError('Unsafe release path')
    return path


class PartStream:
    """Join verified parts without buffering a whole part or the whole archive."""
    def __init__(self, records, opener):
        self.records, self.opener = records, opener
        self.index = 0
        self.current = None
        self.total_bytes = 0

    def read(self, size):
        if size < 0:
            raise ValueError('Release audit requires bounded reads')
        blocks = []
        remaining = size
        while remaining and self.index < len(self.records):
            row = self.records[self.index]
            if self.current is None:
                self.current = self.opener(row)
                self.hasher, self.count = hashlib.sha256(), 0
            block = self.current.read(remaining)
            if block:
                self.hasher.update(block)
                self.count += len(block)
                self.total_bytes += len(block)
                if self.count > row['bytes']:
                    raise ValueError('Release part exceeds recorded size')
                blocks.append(block)
                remaining -= len(block)
            else:
                self.current.close()
                self.current = None
                if self.count != row['bytes'] or self.hasher.hexdigest() != row['sha256']:
                    raise ValueError('Release part size or digest mismatch')
                self.index += 1
        return b''.join(blocks)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


def audit(index_payload, index_sha256, freeze_sha256, opener):
    index = checked_json(index_payload, index_sha256)
    if (index.get('schema') != 'feature-topology.release-parts.v1'
            or index.get('frozen_manifest_sha256') != freeze_sha256
            or not isinstance(index.get('parts'), list) or not index['parts']
            or not isinstance(index.get('files'), dict) or not index['files']
            or 'frozen_manifest.json' in index['files']):
        raise ValueError('Release index is incomplete or has a different freeze')
    for number, row in enumerate(index['parts']):
        if (row.get('file') != f'feature-topology.tar.gz.part{number:03d}'
                or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', ''))):
            raise ValueError('Release part order or identity is invalid')
    for relative, sha in index['files'].items():
        safe_path(relative)
        if not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Invalid release file digest')
    seen = set()
    frozen = None
    stream = PartStream(index['parts'], opener)
    try:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                path = safe_path(member.name)
                if (not member.isfile() or len(path.parts) < 2
                        or path.parts[0] != 'feature-topology'):
                    raise ValueError('Unexpected or nonregular release member')
                relative = '/'.join(path.parts[1:])
                if relative in seen or (relative != 'frozen_manifest.json' and relative not in index['files']):
                    raise ValueError('Duplicate or unlisted release member')
                seen.add(relative)
                digest = hashlib.sha256()
                payload = bytearray()
                if relative == 'frozen_manifest.json' and member.size > 128 * 1024**2:
                    raise ValueError('Frozen manifest exceeds bounded JSON audit size')
                with archive.extractfile(member) as source:
                    while block := source.read(1024**2):
                        digest.update(block)
                        if relative == 'frozen_manifest.json':
                            payload.extend(block)
                expected = freeze_sha256 if relative == 'frozen_manifest.json' else index['files'][relative]
                if digest.hexdigest() != expected:
                    raise ValueError(f'Release member digest mismatch: {relative}')
                if relative == 'frozen_manifest.json':
                    frozen = json.loads(payload)
        # Tar readers may stop before the compressed trailer or final part EOF.
        # Drain it to verify every part, including bytes beyond the last member.
        while stream.read(1024**2):
            pass
    finally:
        stream.close()
    if (seen != set(index['files']) | {'frozen_manifest.json'}
            or not isinstance(frozen, dict) or frozen.get('ready') is not True
            or frozen.get('schema') != 'feature-topology.frozen-results.v1'
            or not isinstance(frozen.get('files'), dict) or not frozen['files']):
        raise ValueError('Release members or frozen input manifest are incomplete')
    if any(index['files'].get(name) != sha for name, sha in frozen['files'].items()):
        raise ValueError('A frozen input is missing or changed in the release index')
    return dict(schema='feature-topology.release-package-audit.v1', verified=True,
                index_sha256=index_sha256, frozen_manifest_sha256=freeze_sha256,
                parts=len(index['parts']), compressed_bytes=stream.total_bytes,
                members=len(seen), frozen_inputs=len(frozen['files']),
                scope='Archive integrity against trusted build digests; scientific readiness requires the study audits')


def release_opener(repository, tag, index):
    if (not re.fullmatch('[0-9A-Za-z_.-]+/[0-9A-Za-z_.-]+', repository)
            or not re.fullmatch('[0-9A-Za-z_.-]+', tag)):
        raise ValueError('Unsafe release repository or tag')
    release = json.loads(subprocess.check_output(['gh', 'api', f'repos/{repository}/releases/tags/{tag}']))
    assets = {row['name']: row for row in release['assets']}
    for row in index['parts']:
        remote = assets.get(row['file'], {})
        if remote.get('size') != row['bytes'] or remote.get('digest') != 'sha256:' + row['sha256']:
            raise ValueError('Remote release part identity differs from the trusted index')
    def open_remote(row):
        return urllib.request.urlopen(assets[row['file']]['browser_download_url'], timeout=120)
    return open_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--index-sha256', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--parts-directory')
    location.add_argument('--release-tag')
    parser.add_argument('--repository', default='uskutsav-cpu/feature-topology')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = Path(args.index).read_bytes()
    index = checked_json(payload, args.index_sha256)
    opener = (release_opener(args.repository, args.release_tag, index) if args.release_tag
              else lambda row: (Path(args.parts_directory) / row['file']).open('rb'))
    report = audit(payload, args.index_sha256, args.freeze_sha256, opener)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
