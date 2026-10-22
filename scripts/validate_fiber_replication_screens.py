"""Validate the complete fixed replication screen before any proof selection."""
from pathlib import Path
import re

import numpy as np

from scripts.recover_release_evidence import digest, load_json
from scripts.screen_fiber_replication import replication_records, source_files


def validate(repo, root):
    repo,root=Path(repo).resolve(),Path(root)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/fiber_replication_v1.json'
    protocol=load_json(protocol_path.read_bytes())
    planned={row['run_id']:row for row in replication_records(index,protocol)}
    collection=load_json((root/'collection.json').read_bytes())
    sources={path.relative_to(repo).as_posix():digest(path) for path in source_files(repo)}
    rows=collection.get('records',[])
    if (collection.get('schema')!='feature-topology.fiber-replication-screen-collection.v1'
            or collection.get('expected_models')!=36 or collection.get('complete_models')!=36
            or collection.get('planned_domains')!=144 or len(rows)!=36
            or {row['run_id'] for row in rows}!=set(planned)
            or collection.get('training_seeds')!=list(range(1,10))
            or collection.get('protocol_sha256')!=digest(protocol_path)
            or not re.fullmatch('[0-9a-f]{40}',collection.get('source_commit',''))):
        raise ValueError('Incomplete or changed replication screen grid')
    for row in rows:
        record=planned[row['run_id']]
        passed=row.get('empirical_screen_passed')
        maximum=row.get('empirical_max_derivative_error')
        readout=np.array([[float.fromhex(value) for value in line] for line in row['readout_hex']])
        intercept=np.array([float.fromhex(value) for value in row['intercept_hex']])
        if (load_json((root/(row['run_id']+'.json')).read_bytes())!=row
                or row.get('schema')!='feature-topology.fiber-replication-screen.v1'
                or row.get('gamma')!=record['gamma'] or row.get('seed')!=record['seed']
                or row.get('input_archive_sha256')!=record['archive_sha256']
                or row.get('input_index_sha256')!=digest(index_path)
                or row.get('analysis_data_sha256')!=index['analysis_data']['sha256']
                or row.get('protocol_sha256')!=digest(protocol_path) or row.get('source_sha256')!=sources
                or row.get('source_commit')!=collection['source_commit']
                or row.get('checkpoint_sha256')!=record['files'].get(row.get('checkpoint'))
                or type(row.get('selected_step')) is not int or row['selected_step']<0
                or not np.isfinite(row['actual_training_loss']) or not 0<=row['actual_training_loss']<=.1
                or row.get('samples')!=4096 or row.get('continuous_certificate') is not False
                or type(passed) is not bool or not np.isfinite(maximum) or maximum<0
                or passed!=bool(maximum<2/np.pi)
                or row.get('status')!=('eligible_fixed_witness' if passed else 'fixed_witness_screen_failed')
                or readout.shape!=(2,record['config'].get('width',256)) or not np.isfinite(readout).all()
                or intercept.shape!=(2,) or not np.isfinite(intercept).all()
                or not np.isfinite(row['ridge']) or row['ridge']<=0
                or not np.isfinite(row['fit_rmse']) or row['fit_rmse']<0):
            raise ValueError('Replication witness/source/screen binding changed')
    eligible=sum(row['empirical_screen_passed'] for row in rows)
    if collection.get('eligible_models')!=eligible or collection.get('screen_failed_models')!=36-eligible:
        raise ValueError('Replication screen status counts differ')
    return collection
