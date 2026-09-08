"""Crash-safe single-file writes; concurrent writers publish complete files.

This is atomic last-writer-wins replacement, not a lock for an entire experiment.
Use disjoint run assignments when executing multiple training workers.
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import torch


def fingerprint(config):
    # Preserve existing run identifiers for backward compatibility.
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def atomic_write(path, writer):
    """Call writer(binary_stream), then atomically publish in the same directory.

    A unique temporary file prevents workers from replacing each other's .tmp
    file. A failed writer leaves the previous destination unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, value):
    # Validate/serialize before touching the destination; reject NaN/Infinity.
    payload = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(payload))


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 2*1024**3:
        raise RuntimeError("Checkpoint storage reserve reached (2 GiB); use an external results volume")
    atomic_write(path, lambda stream: torch.save(payload, stream))
