"""Summarize the v3 evidence ladder without conflating certification levels."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from research_ext.exact import verify_polygon_certificate
from research_ext.io import atomic_json, atomic_text
from scripts.freeze_results import exact_control_paths,validate_numeric_circle_quotient


def summarize_certification(repo,output):
    repo=Path(repo)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    exact_control_paths(repo)
    controls=json.loads((repo/'results/research_ext/exact_controls/control_summary.json').read_text())
    trained=json.loads((repo/'results/exact_trained_circles/manifest.json').read_text())
    exact_rows=[]
    for record in trained['records']:
        path=repo/'results/exact_trained_circles'/record['certificate_file']
        certificate=json.loads(path.read_text())
        if not verify_polygon_certificate(certificate):
            raise ValueError(f'Exact trained-circle replay failed: {path}')
        exact_rows.append(dict(run_id=record['run_id'],gamma=record['gamma'],seed=record['seed'],
                               layer=record['layer'],beta0=certificate['graph']['beta0'],
                               beta1=certificate['graph']['beta1'],
                               injective_on_polygon_subset=certificate['injective_on_polygon_subset'],
                               exact_collision_present=certificate['collision'] is not None,
                               evidence_level='rationally certified on declared polygon'))
    atomic_text(output/'trained_circle_exact_certificates.csv',pd.DataFrame(exact_rows).to_csv(index=False))
    numerical_rows=[]
    for record in json.loads((repo/'results/circle_quotient/manifest.json').read_text()):
        path=repo/'results/circle_quotient/runs'/record['run_id']/'quotient.json'
        report=validate_numeric_circle_quotient(json.loads(path.read_text()))
        for stage in ('initial','final'):
            for index,layer in enumerate(report[stage],1):
                numerical_rows.append(dict(run_id=record['run_id'],gamma=record['gamma'],seed=record['seed'],
                                           stage=stage,layer=index,beta0=layer['beta0'],beta1=layer['beta1'],
                                           nodes=layer['nodes'],edges=layer['edges'],
                                           tolerance=layer['tolerance'],
                                           evidence_level='numerically resolved finite polygon with LP intersections'))
    atomic_text(output/'trained_circle_numerical_diagnostics.csv',pd.DataFrame(numerical_rows).to_csv(index=False))
    formal=json.loads((repo/'results/completion/formal/formal_status.json').read_text())
    if formal.get('lean_verified') is not True:
        raise ValueError('Formal audit is not verified')
    result=dict(schema='feature-topology.certification-summary.v1',
                analytic_exact_controls=len(controls['controls']),
                rational_trained_layer_certificates=len(exact_rows),
                exact_trained_checkpoints=len({row['run_id'] for row in exact_rows}),
                numerical_polygon_layer_diagnostics=len(numerical_rows),
                lean_verified=True,
                exact_scope='Stored rational weights on the declared finite polygon or positive box only.',
                numerical_scope='Tolerance-dependent finite polygon/LP computation; not a smooth-continuum proof.',
                formal_scope=formal['scope'])
    atomic_json(output/'certification_scope.json',result)
    return result

