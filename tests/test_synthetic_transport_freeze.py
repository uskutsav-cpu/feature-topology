import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.validate_synthetic_transport import validate


ROOT = Path(__file__).resolve().parents[1]
REL = Path('results/synthetic_archive_transport_v1')


def copy_evidence(tmp_path):
    shutil.copytree(ROOT / REL, tmp_path / REL)
    audit = json.loads((tmp_path / REL / 'transport_audit.json').read_text())
    for relative in audit['bound_source_files_sha256']:
        destination = tmp_path / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
    return tmp_path / REL, audit


def test_real_transport_collection_binds_every_receipt_and_source():
    paths = validate(ROOT)
    assert len([p for p in paths if p.parent.name == 'receipts']) == 845
    assert ROOT / REL / 'primary_workflow.json' in paths
    assert ROOT / 'configs/completion/primary_inputs_v3.json' in paths


def test_missing_receipt_prevents_freeze(tmp_path):
    root, _ = copy_evidence(tmp_path)
    next((root / 'receipts').glob('*.json')).unlink()
    with pytest.raises(ValueError, match='coverage'):
        validate(tmp_path)


def test_rehashed_receipt_cannot_change_its_original_archive(tmp_path):
    root, audit = copy_evidence(tmp_path)
    rid = next(iter(audit['receipts_sha256']))
    receipt = root / 'receipts' / f'{rid}.json'
    value = json.loads(receipt.read_text())
    value['input_archive_sha256'] = '0' * 64
    receipt.write_text(json.dumps(value))
    audit['receipts_sha256'][rid] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    (root / 'transport_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='archive identity'):
        validate(tmp_path)


def test_source_observation_is_bound_even_without_changing_receipts(tmp_path):
    root, _ = copy_evidence(tmp_path)
    (root / 'primary_workflow.json').write_text('{}')
    with pytest.raises(ValueError, match='hash changed'):
        validate(tmp_path)


def test_transport_manifest_cannot_read_an_external_source(tmp_path):
    root, audit = copy_evidence(tmp_path)
    audit['bound_source_files_sha256']['../outside.json'] = '0' * 64
    (root / 'transport_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='escapes repository'):
        validate(tmp_path)
