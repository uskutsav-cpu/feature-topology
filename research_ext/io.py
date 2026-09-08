"""Strict JSON, content hashes, atomic writes and reproducibility records."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def original_fingerprint(value: Any) -> str:
    """Byte-compatible with the upstream config fingerprint, not `digest`."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-JSON number {value} in {path}")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=invalid, object_pairs_hook=pairs)


def atomic_json(path: str | Path, value: Any) -> None:
    """A unique temporary file avoids shared '.tmp' races between workers."""
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a measurement")
    number = float(value)
    return number if math.isfinite(number) else None


def environment(repo: str | Path = ".") -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args],
                                           stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    versions = {}
    for name in ["numpy", "scipy", "pandas", "torch", "scikit-learn", "ripser", "pytest"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    state = {"utc": utc_now(), "python": platform.python_version(),
             "platform": platform.platform(), "machine": platform.machine(),
             "packages": versions, "git_commit": git("rev-parse", "HEAD"),
             "git_status": git("status", "--porcelain"),
             "note": "Seeds do not guarantee cross-platform bitwise equivalence."}
    if versions["torch"]:
        import torch
        state["torch"] = {"cuda_available": torch.cuda.is_available(),
                          "cuda_version": torch.version.cuda,
                          "threads": torch.get_num_threads(),
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return state


@contextlib.contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Never automatically steal a lock: another computer may own it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Lock exists: {path}. Verify the owner stopped before removing it.") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "created": utc_now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)
