import hashlib
import json
import os
import shutil
from pathlib import Path
import torch


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temp, path)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 2*1024**3:
        raise RuntimeError("Checkpoint storage reserve reached (2 GiB); use an external results volume")
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
