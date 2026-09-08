#!/usr/bin/env python3
"""Apply an additive, SHA256-checked update to an existing checkout.

Never deletes or replaces an existing different file. Does not commit or push.
The manifest provides integrity checking, not a cryptographic publisher signature.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate manifest key: {key}')
        result[key] = value
    return result


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if (not relative or p.is_absolute() or str(p) != relative or '\\' in relative
            or any(part in {'..', '.', '.git'} for part in p.parts)):
        raise ValueError(f'Unsafe bundle path: {relative}')
    candidate = root.joinpath(*p.parts)
    current = root
    for part in p.parts:
        current = current/part
        if current.is_symlink():
            raise ValueError(f'Refusing a symlinked payload/destination: {current}')
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return candidate


def apply_update(bundle: str | Path, repo: str | Path, *, dry_run: bool = False,
                 allow_different_base: bool = False) -> dict:
    bundle, repo = Path(bundle).resolve(), Path(repo).resolve()
    if not repo.is_dir():
        raise ValueError('Destination must be an existing repository directory')
    manifest = json.loads((bundle/'BUNDLE_MANIFEST.json').read_text(), object_pairs_hook=_object)
    if manifest.get('schema') != 'feature-topology.additive-bundle.v1':
        raise ValueError('Unsupported bundle manifest')
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                          check=True, text=True, capture_output=True).stdout.strip()
    top = subprocess.run(['git','-C',str(repo),'rev-parse','--show-toplevel'],
                         check=True,text=True,capture_output=True).stdout.strip()
    if Path(top).resolve() != repo:
        raise ValueError('Pass the checkout root, not a nested directory')
    if head != manifest['base_commit'] and not allow_different_base:
        raise ValueError(f"Expected base {manifest['base_commit']}, found {head}. "
                         'Review compatibility before using --allow-different-base.')
    pending, identical = [], []
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('No files in manifest')
    # Validate every source and destination BEFORE making any change.
    for name, expected in sorted(files.items()):
        source, destination = safe_path(bundle, name), safe_path(repo, name)
        payload = source.read_bytes()
        if sha256(payload) != expected:
            raise ValueError(f'Bundle hash mismatch: {name}')
        if destination.exists():
            if not destination.is_file() or sha256(destination.read_bytes()) != expected:
                raise ValueError(f'Existing different file would be overwritten: {name}')
            identical.append(name)
        else:
            for parent in destination.parents:
                if parent == repo:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f'Destination parent is not a directory: {parent}')
            pending.append((name, destination, payload))
    created = []
    if not dry_run:
        try:
            for name, destination, payload in pending:
                safe_path(repo, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Prepare complete bytes before an exclusive, non-replacing link.
                # A filesystem that does not support hard links fails safely.
                descriptor, temporary_name = tempfile.mkstemp(prefix='.research-update-', dir=destination.parent)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, 'wb') as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(temporary, destination)
                    created.append((destination, sha256(payload)))
                finally:
                    temporary.unlink(missing_ok=True)
        except BaseException:
            for path, expected in reversed(created):
                # These files were exclusively created by this invocation.
                # Preserve any file another process has subsequently changed.
                try:
                    if path.is_file() and sha256(path.read_bytes()) == expected:
                        path.unlink()
                except OSError:
                    pass
            raise
    return {'status':'dry_run' if dry_run else 'applied', 'base_commit':head,
            'added_files':[x[0] for x in pending], 'already_identical':identical,
            'overwritten_files':[], 'committed':False, 'pushed':False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo', help='Path to the existing feature-topology checkout')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-different-base', action='store_true',
                   help='Accept another HEAD after reviewing compatibility; does not allow overwrites')
    args = p.parse_args(argv)
    try:
        print(json.dumps(apply_update(Path(__file__).resolve().parents[1], args.repo,
                         dry_run=args.dry_run, allow_different_base=args.allow_different_base), indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'Update not applied: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
