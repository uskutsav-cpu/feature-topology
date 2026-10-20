"""Recover hash-verified JSON evidence without retaining large binary archives.

This produces an analysis mirror, not a full results freeze. Every downloaded
archive and every member is hashed, including binary members that are discarded.
Tensor/array semantics are not checked here. The existing full freeze remains
required for publication. No placeholder checkpoints are created.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(payload):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(payload, parse_constant=reject)


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive member: {name}')
    return path


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.hash.update(value)
        self.bytes += len(value)
        return value


def verify_stream(stream, expected_sha256, expected_files, keep):
    """Verify a gzip/tar stream before returning selected small JSON payloads."""
    reader = HashedReader(stream)
    observed, retained = {}, {}
    with tarfile.open(fileobj=reader, mode='r|gz') as archive:
        for member in archive:
            safe_name(member.name)
            if not member.isfile() or member.name in observed:
                raise ValueError(f'Duplicate or nonregular member: {member.name}')
            if member.name not in expected_files:
                raise ValueError(f'Unexpected archive member: {member.name}')
            selected = keep(member.name)
            if selected and member.size > 8*1024**2:
                raise ValueError(f'JSON evidence member too large: {member.name}')
            hasher, payload = hashlib.sha256(), bytearray()
            with archive.extractfile(member) as source:
                while chunk := source.read(1024*1024):
                    hasher.update(chunk)
                    if selected:
                        payload.extend(chunk)
            observed[member.name] = hasher.hexdigest()
            if expected_files[member.name] is not None and observed[member.name] != expected_files[member.name]:
                raise ValueError(f'Member checksum mismatch: {member.name}')
            if selected:
                load_json(payload)
                retained[member.name] = bytes(payload)
    # Include trailing compressed bytes in the archive digest as well.
    while reader.read(1024*1024):
        pass
    if set(observed) != set(expected_files):
        raise ValueError('Archive member set is incomplete')
    if reader.hash.hexdigest() != expected_sha256:
        raise ValueError('Archive checksum mismatch')
    return retained, observed, reader.bytes


def asset(catalog, tag, name):
    matches = [a for release in catalog if release['tag_name'] == tag
               for a in release['assets'] if a['name'] == name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one release asset: {tag}/{name}')
    return matches[0]


def fetch_json(row):
    request = urllib.request.Request(row['browser_download_url'],
                                     headers={'User-Agent': 'Feature-Topology-evidence-recovery'})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    expected = row.get('digest')
    if expected and expected != 'sha256:'+hashlib.sha256(payload).hexdigest():
        raise ValueError(f'Release JSON digest mismatch: {row["name"]}')
    return load_json(payload), payload


def publish(root, files):
    """Publish only already-verified payloads; never replace conflicting evidence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(len(payload) for name, payload in files.items()
                         if not (root/name).exists())
    if shutil.disk_usage(root).free < required_bytes + 256*1024**2:
        raise RuntimeError('Insufficient free space for verified JSON evidence')
    for name, payload in files.items():
        safe_name(name)
        destination = root.joinpath(*PurePosixPath(name).parts)
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f'Conflicting analysis evidence: {destination}')
    for name, payload in files.items():
        destination = root.joinpath(*PurePosixPath(name).parts)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary_name = temporary.name
        os.replace(temporary_name, destination)


def recover_one(record, index, collector, catalog, root, local_sources=()):
    rid = record['run_id']
    receipt_path = root/'receipts'/f'{rid}.json'
    if receipt_path.exists():
        receipt = load_json(receipt_path.read_bytes())
        if (receipt.get('input_archive_sha256') == record['archive_sha256']
                and receipt.get('metric_archive') == collector['complete'][rid]
                and all((root/path).is_file() and digest(root/path) == expected
                        for path, expected in receipt['retained_files'].items())):
            return dict(run_id=rid, status='verified_cached_receipt')
        raise ValueError(f'Changed cached recovery receipt: {rid}')
    input_files, input_method, transferred = {}, 'full_archive_hash_and_members', 0
    for source in local_sources:
        for directory in [Path(source)/'main/runs'/rid, Path(source)/'pilot/runs'/rid]:
            names = ['config.json', 'summary.json']
            if all((directory/name).is_file() and digest(directory/name) == record['files'][name]
                   for name in names):
                input_files = {name: (directory/name).read_bytes() for name in names}
                input_method = 'existing_json_member_hashes_only'
                break
        if input_files:
            break
    if not input_files:
        row = asset(catalog, collector['input_tag'], record['asset'])
        expected = {f'{rid}/{name}': value for name, value in record['files'].items()}
        request = urllib.request.Request(row['browser_download_url'],
                                         headers={'User-Agent': 'Feature-Topology-evidence-recovery'})
        with urllib.request.urlopen(request, timeout=90) as response:
            kept, _, count = verify_stream(response, record['archive_sha256'], expected,
                                           lambda name: name.endswith('.json'))
        transferred += count
        input_files = {name.split('/', 1)[1]: value for name, value in kept.items()}
    config = load_json(input_files['config.json'])
    summary = load_json(input_files['summary.json'])
    if (config != record['config'] or summary.get('config') != config
            or summary.get('run_id') != rid or summary.get('status') != record['status']):
        raise ValueError(f'Training JSON identity mismatch: {rid}')

    archive_name = collector['complete'][rid]
    prefix = f'metrics-{rid}-'
    if not archive_name.startswith(prefix) or not archive_name.endswith('.tar.gz'):
        raise ValueError(f'Unexpected metric archive name: {archive_name}')
    job_id = archive_name[len(prefix):-len('.tar.gz')]
    status_name = f'status-{rid}-{job_id}.json'
    status, status_payload = fetch_json(asset(catalog, collector['result_tag'], status_name))
    required = sorted(name[:-3] for name in record['files'] if name.endswith('.pt'))
    if not index['metric_options']['all_checkpoints']:
        required = ['final', 'step_0000000']
    current = (status.get('complete') is True and sorted(status.get('expected', [])) == required)
    legacy = (status.get('complete') is False and not index['metric_options']['all_checkpoints']
              and sorted(status.get('expected', [])) == sorted(
                  name[:-3] for name in record['files'] if name.endswith('.pt')))
    if (status.get('run_id') != rid or status.get('input_archive_sha256') != record['archive_sha256']
            or status.get('analysis_data_sha256') != index['analysis_data']['sha256']
            or sorted(status.get('completed', [])) != required or not (current or legacy)):
        raise ValueError(f'Incomplete or mismatched metric status: {rid}')
    metric_options = hashlib.sha256(json.dumps(index['metric_options'], sort_keys=True).encode()).hexdigest()[:16]
    metric_prefix = f'main/runs/{rid}/metrics/{metric_options}/'
    if any(not (name.startswith(metric_prefix) or name.startswith('ph_cache/'))
           for name in status['files']):
        raise ValueError(f'Metric archive contains an unrelated run/profile: {rid}')
    row = asset(catalog, collector['result_tag'], archive_name)
    request = urllib.request.Request(row['browser_download_url'],
                                     headers={'User-Agent': 'Feature-Topology-evidence-recovery'})
    expected = {**status['files'], 'job_manifest.json': None}
    with urllib.request.urlopen(request, timeout=90) as response:
        kept, _, count = verify_stream(response, status['archive_sha256'], expected,
                                       lambda name: name.endswith('.json'))
    transferred += count
    report = load_json(kept['job_manifest.json'])
    for key in ['run_id', 'job_id', 'complete', 'completed', 'expected', 'input_archive_sha256',
                'analysis_data_sha256', 'metric_provenance', 'files']:
        if report.get(key) != status.get(key):
            raise ValueError(f'Status/archive disagreement: {rid}/{key}')
    files = {f'runs/{rid}/{name}': value for name, value in input_files.items()}
    for name, payload in kept.items():
        if name.startswith(metric_prefix):
            files[f'runs/{rid}/metrics/{metric_options}/'+name[len(metric_prefix):]] = payload
    options = load_json(files[f'runs/{rid}/metrics/{metric_options}/options.json'])
    if options != index['metric_options']:
        raise ValueError(f'Metric options mismatch: {rid}')
    for checkpoint in required:
        name = f'runs/{rid}/metrics/{metric_options}/{checkpoint}.json'
        metric = load_json(files[name])
        if (metric.get('checkpoint_sha256') != record['files'][checkpoint+'.pt']
                or metric.get('gamma') != record['gamma'] or metric.get('seed') != record['seed']
                or len(metric.get('layers', [])) != config.get('depth', 4)):
            raise ValueError(f'Metric/checkpoint identity mismatch: {rid}/{checkpoint}')
    files[f'statuses/{status_name}'] = status_payload
    receipt = dict(schema='feature-topology.json-evidence-recovery.v1', run_id=rid,
                   input_archive_sha256=record['archive_sha256'], input_validation=input_method,
                   metric_archive=archive_name, metric_archive_sha256=status['archive_sha256'],
                   metric_provenance=status['metric_provenance'], required_checkpoints=required,
                   transferred_bytes=transferred,
                   retained_files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
                   scope='JSON analysis mirror; not tensor validation or a full results freeze')
    publish(root, files)
    publish(root, {f'receipts/{rid}.json': (json.dumps(receipt, indent=2)+'\n').encode()})
    return dict(run_id=rid, status='recovered_verified_json', transferred_bytes=transferred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--collector', required=True)
    parser.add_argument('--release-catalog', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-source', action='append', default=[])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    index = load_json(Path(args.index).read_bytes())
    collector = load_json(Path(args.collector).read_bytes())
    catalog = load_json(Path(args.release_catalog).read_bytes())
    records = [row for row in index['runs'] if row['run_id'] in collector['complete']]
    if len(records) != collector['expected']:
        raise ValueError('Collector/index coverage mismatch')
    if args.limit is not None:
        records = records[:args.limit]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # This mirror retains small JSON records only, never training checkpoints.
    # The upstream 2 GiB checkpoint reserve remains unchanged.
    if shutil.disk_usage(output).free < 256*1024**2:
        raise RuntimeError('Recovery requires 256 MiB free for JSON evidence')
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(recover_one, row, index, collector, catalog, output,
                                   args.local_source): row['run_id'] for row in records}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failure = dict(run_id=futures[future], error=str(error))
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    summary = dict(schema='feature-topology.json-evidence-recovery-summary.v1',
                   index_sha256=digest(args.index), collector_sha256=digest(args.collector),
                   expected=len(records), complete=len(records)-len(failures), failures=failures,
                   scope='Analysis mirror only; the full freeze gate is unchanged')
    (output/'recovery_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
