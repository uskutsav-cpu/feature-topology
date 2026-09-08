"""Validate completed replication artifacts and export auditable result tables."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.checkpoints import atomic_json


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to export')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='results/continuation')
    args = p.parse_args(); root = Path(args.root)
    study = json.loads((root/'study_config.json').read_text())
    records = json.loads((root/'diagnostics.json').read_text())
    expected = {(g, s) for g in study['gammas'] for s in study['seeds']}
    observed = {(r['gamma'], r['seed']) for r in records}
    if len(records) != len(observed) or observed != expected:
        raise RuntimeError('Diagnostics are duplicate or incomplete; refusing a completed-study summary')
    if len({r['code_sha256'] for r in records}) != 1:
        raise RuntimeError('Mixed numerical-code versions')
    if any(r['options'] != study['metrics'] for r in records):
        raise RuntimeError('Mixed metric resolutions')
    layer_rows, noise_rows, runs = [], [], []
    for r in sorted(records, key=lambda r: (r['gamma'], r['seed'])):
        runs.append({k: r[k] for k in ('run_id','gamma','seed','step','training_loss',
                                      'test_accuracy','weight_displacement','ntk_drift','checkpoint_sha256')})
        for v in r['layers']:
            layer_rows.append(dict(run_id=r['run_id'],gamma=r['gamma'],seed=r['seed'],layer=v['layer'],
                cka_drift=v['cka_drift'],scale=v['scale'],effective_rank=v['effective_rank'],
                local_raw_q01=v['jacobian']['q01'],local_normalized_q01=v['jacobian']['normalized_q01'],
                task_jacobian=v['jacobian']['task_norm'],nuisance_jacobian=v['jacobian']['nuisance_norm'],
                fiber_raw_q01=v['nuisance_fiber']['raw_q01'],fiber_normalized_q01=v['nuisance_fiber']['normalized_q01'],
                exact_float_equal_pairs=v['nuisance_fiber']['exact_float_equal_pairs'],
                linear_nuisance_cosine=v['probes']['linear']['angular_cosine'],
                mlp_nuisance_cosine=v['probes']['mlp']['angular_cosine'],
                mlp_nuisance_angular_mae=v['probes']['mlp']['angular_mae'],
                convergence_warnings=sum(p['convergence_warnings'] for p in v['probes'].values())))
            for n in v.get('noise_probe', []):
                noise_rows.append(dict(gamma=r['gamma'],seed=r['seed'],layer=v['layer'],**n))
    aggregate = []
    for gamma in study['gammas']:
        part = [v for v in layer_rows if v['gamma']==gamma and v['layer']==study['training']['depth']]
        rr = [v for v in runs if v['gamma']==gamma]
        row = dict(gamma=gamma,seeds=len(part),test_accuracy_mean=float(np.mean([v['test_accuracy'] for v in rr])))
        for metric in ('cka_drift','local_raw_q01','local_normalized_q01','fiber_raw_q01',
                       'fiber_normalized_q01','mlp_nuisance_cosine'):
            values = [v[metric] for v in part]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_sd'] = float(np.std(values,ddof=1))
        for level in (0., .001, .01, .1):
            values = [v['angular_cosine'] for v in noise_rows if v['gamma']==gamma and v['noise_level']==level]
            row[f'ridge_noise_{level:g}_mean'] = float(np.mean(values))
        aggregate.append(row)
    write_csv(root/'runs.csv',runs)
    write_csv(root/'layer_metrics.csv',layer_rows)
    write_csv(root/'noise_metrics.csv',noise_rows)
    write_csv(root/'gamma_summary.csv',aggregate)
    overview = dict(completed_runs=len(runs),gamma_levels=len(study['gammas']),seeds=study['seeds'],
        layer_records=len(layer_rows),noise_records=len(noise_rows),
        training_loss_range=[min(r['training_loss'] for r in runs),max(r['training_loss'] for r in runs)],
        exact_float_equal_pairs=sum(v['exact_float_equal_pairs'] for v in layer_rows),
        probe_convergence_warnings=sum(v['convergence_warnings'] for v in layer_rows),
        numerical_code_sha256=records[0]['code_sha256'],
        persistence_computed=False,interpretation=study['interpretation'])
    atomic_json(root/'overview.json',overview)
    print(json.dumps(overview,indent=2))
    print('gamma,clean_mlp_cosine,clean_ridge_cosine,noise_0.1_ridge_cosine,normalized_local_q01')
    for row in aggregate:
        print(','.join(str(row[k]) for k in ('gamma','mlp_nuisance_cosine_mean','ridge_noise_0_mean','ridge_noise_0.1_mean','local_normalized_q01_mean')))


if __name__ == '__main__': main()
