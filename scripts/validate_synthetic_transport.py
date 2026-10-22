"""Bind retained transport receipts to the frozen production archive identities.

These receipts record previous archive/member checks. This validator neither
downloads archives nor replaces the full freeze's physical checkpoint checks.
"""
import hashlib
import json
from pathlib import Path


def validate(repo):
    repo = Path(repo).resolve()
    root = repo / 'results/synthetic_archive_transport_v1'
    paths = set()

    def read(path, expected=None, parse=True):
        path = Path(path)
        if not path.resolve().is_relative_to(repo):
            raise ValueError('Transport evidence escapes repository')
        payload = path.read_bytes()
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Transport evidence hash changed: {path.name}')
        paths.add(path)
        return json.loads(payload) if parse else None

    audit = read(root / 'transport_audit.json')
    if (audit.get('schema') != 'feature-topology.synthetic-archive-transport.v1'
            or audit.get('full_study_frozen') is not False
            or audit.get('profiles') != 845 or audit.get('primary') != 130
            or audit.get('other_profiles') != 715):
        raise ValueError('Transport audit scope or coverage changed')
    for relative, expected in audit['bound_source_files_sha256'].items():
        read(repo / relative, expected, parse=False)
    registry = read(repo / 'configs/completion/production_cohorts_v3.json')
    expected, cohorts = {}, {}
    for cohort in registry['cohorts']:
        index = read(repo / cohort['index'])
        collector = read(repo / f"results/completion/remote_collections/{cohort['name']}.json")
        for row in index['runs']:
            rid = row['run_id']
            if rid not in collector['complete']:
                continue
            if rid in expected:
                raise ValueError('Duplicate production archive identity')
            expected[rid] = (row['archive_sha256'], collector['complete'][rid])
            cohorts[rid] = cohort['name']
    receipts = audit['receipts_sha256']
    if (len(expected) != 845 or set(receipts) != set(expected)
            or audit.get('cohorts_by_run_id') != cohorts
            or {p.stem for p in (root / 'receipts').glob('*.json')} != set(expected)):
        raise ValueError('Transport receipt coverage changed')
    retained_count = 0
    for rid, (archive_sha, metric_archive) in expected.items():
        receipt = read(root / 'receipts' / f'{rid}.json', receipts[rid])
        if (receipt.get('schema') != 'feature-topology.json-evidence-recovery.v1'
                or receipt.get('run_id') != rid
                or receipt.get('input_validation') != 'full_archive_hash_and_members'
                or receipt.get('input_archive_sha256') != archive_sha
                or receipt.get('metric_archive') != metric_archive):
            raise ValueError(f'Transport receipt/archive identity changed: {rid}')
        retained_count += len(receipt['retained_files'])
    if retained_count != audit.get('retained_file_hashes_rechecked') or retained_count != 7719:
        raise ValueError('Transport retained-file accounting changed')
    for filename, run_id, source in [
            ('primary_workflow.json', 39540638096, 'dfa8e43d0f06231dcc5eddac38306b0bed405f3b'),
            ('other_profiles_workflow.json', 39482975307, '6b35f85ee55b04b963352b7de9a7be48c91a96f1')]:
        workflow = read(root / filename)
        if (workflow.get('databaseId') != run_id or workflow.get('headSha') != source
                or workflow.get('status') != 'completed' or workflow.get('conclusion') != 'success'):
            raise ValueError('Transport source workflow changed')
    return paths
