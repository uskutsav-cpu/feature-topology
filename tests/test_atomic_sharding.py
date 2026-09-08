from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import pytest
from src.training.checkpoints import atomic_json, atomic_write, fingerprint
from src.training.sharding import shard_for, select_shard


def test_atomic_concurrent_writers(tmp_path):
    path = tmp_path/'result.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: atomic_json(path, {'index': i, 'payload': [i]*100}), range(64)))
    result = json.loads(path.read_text())
    assert result['payload'] == [result['index']]*100
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_writer_preserves_existing_file(tmp_path):
    path = tmp_path/'result.bin'; path.write_bytes(b'original')
    def fail(stream):
        stream.write(b'incomplete'); raise RuntimeError('simulated')
    with pytest.raises(RuntimeError, match='simulated'):
        atomic_write(path, fail)
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_json_rejects_nonfinite_before_write(tmp_path):
    path = tmp_path/'result.json'; atomic_json(path, {'ok': 1})
    with pytest.raises(ValueError):
        atomic_json(path, {'bad': float('nan')})
    assert json.loads(path.read_text()) == {'ok': 1}


def test_shards_are_disjoint_complete_and_stable():
    old = [Path(f'{i:016x}') for i in range(100)]
    new = old + [Path('0000_new'), Path('zzzz_new')]
    pieces = [set(select_shard(old, i, 4)) for i in range(4)]
    assert set.union(*pieces) == set(old)
    assert sum(map(len, pieces)) == len(old)
    for i in range(4):
        assert set(select_shard(new, i, 4)) & set(old) == pieces[i]


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_shard_count_validation(count):
    with pytest.raises(ValueError): shard_for('abc', count)


@pytest.mark.parametrize('index', [-1, 3, 0.5, True])
def test_shard_index_validation(index):
    with pytest.raises(ValueError): select_shard([Path('a')], index, 3)


def test_run_fingerprint_is_unchanged():
    import hashlib
    value = {'gamma': 1., 'seed': 2}
    assert fingerprint(value) == hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
