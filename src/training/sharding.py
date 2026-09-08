"""Stable work assignment: adding run directories never moves existing runs."""
import hashlib


def shard_for(run_id: str, shard_count: int) -> int:
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    return int(hashlib.sha256(run_id.encode("utf-8")).hexdigest(), 16) % shard_count


def select_shard(runs, shard_index: int = 0, shard_count: int = 1):
    shard_for("validation", shard_count)
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    return [run for run in sorted(runs) if shard_for(run.name, shard_count) == shard_index]
