"""Grouped, nested held-out-seed comparison of the recorded diagnostic models.

All model choices are from strengthening_diagnostic_protocol_v1.json. This is an
additional research analysis and never substitutes for the full freeze gate.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import evaluation_plan
from research_ext.catalog import ENDPOINT_ONLY_FIELDS, _flatten
import research_ext.catalog as catalog_module


def grouped_folds(groups):
    groups = np.asarray(groups)
    for seed in sorted(set(groups.tolist())):
        train, test = np.flatnonzero(groups != seed), np.flatnonzero(groups == seed)
        if len(train) == 0 or len(test) == 0:
            raise ValueError('Grouped validation needs at least two nonempty seed groups')
        yield int(seed), train, test


def ridge_predict(design_train, numeric_train, y_train, design_test, numeric_test, alpha):
    """Fit centering/scaling and intercept exclusively on the supplied train set."""
    mean, scale = numeric_train.mean(axis=0), numeric_train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    x_train = np.column_stack([design_train, (numeric_train-mean)/scale])
    x_test = np.column_stack([design_test, (numeric_test-mean)/scale])
    center, offset = x_train.mean(axis=0), y_train.mean()
    centered = x_train-center
    coefficient = np.linalg.solve(centered.T @ centered+alpha*np.eye(x_train.shape[1]),
                                  centered.T @ (y_train-offset))
    return (x_test-center) @ coefficient+offset


def nested_predictions(design, numeric, outcome, groups, alphas):
    design, numeric, outcome, groups = map(np.asarray, [design, numeric, outcome, groups])
    if not all(np.isfinite(value).all() for value in [design, numeric, outcome]):
        raise ValueError('Undefined values prevent this comparison; no implicit imputation')
    predictions, folds = np.full(len(outcome), np.nan), []
    for seed, outer_train, outer_test in grouped_folds(groups):
        losses = []
        for alpha in alphas:
            errors = []
            for _, train, validation in grouped_folds(groups[outer_train]):
                train, validation = outer_train[train], outer_train[validation]
                prediction = ridge_predict(design[train], numeric[train], outcome[train],
                                           design[validation], numeric[validation], alpha)
                errors.append(float(np.mean((outcome[validation]-prediction)**2)))
            losses.append(float(np.mean(errors)))
        chosen = int(np.argmin(losses))  # Declared grid order breaks exact ties.
        prediction = ridge_predict(design[outer_train], numeric[outer_train], outcome[outer_train],
                                   design[outer_test], numeric[outer_test], alphas[chosen])
        predictions[outer_test] = prediction
        folds.append(dict(seed=seed, training_seeds=sorted(set(groups[outer_train].tolist())),
                          alpha=float(alphas[chosen]), inner_mse=losses,
                          mse=float(np.mean((outcome[outer_test]-prediction)**2)),
                          samples=len(outer_test)))
    return predictions, folds


def find_unique(root, name):
    # Each cohort may be a symlink to a separately downloaded immutable mirror.
    paths = list(Path(root).glob('*/runs/'+name))
    if len(paths) != 1:
        raise ValueError(f'Expected one {name} under {root}, found {len(paths)}')
    return paths[0]


def comparison_rows(repo, evidence, ood, variability, aliases, protocol):
    repo = Path(repo).resolve()
    plan = evaluation_plan(repo)
    outcomes = load_json((Path(ood)/'manifest.json').read_bytes())['runs']
    baselines = load_json((Path(variability)/'manifest.json').read_bytes())['runs']
    outcome_by_id, baseline_by_id = ({row['run_id']: row for row in records}
                                     for records in [outcomes, baselines])
    ids = {row['record']['run_id'] for row in plan}
    if len(outcomes) != len(ids) or len(baselines) != len(ids) or set(outcome_by_id) != ids or set(baseline_by_id) != ids:
        raise ValueError('Comparison requires every frozen cell exactly once')
    supplied_aliases = list(Path(aliases).rglob('*.json'))
    alias_paths = {path.stem: path for path in supplied_aliases}
    if len(supplied_aliases) != len(ids) or set(alias_paths) != ids:
        raise ValueError('Comparison requires an exact checkpoint alias audit for every cell')
    rows, hashes, missing = [], {}, []
    for cell in plan:
        record, rid = cell['record'], cell['record']['run_id']
        directory = find_unique(evidence, rid)
        config_path, summary_path = directory/'config.json', directory/'summary.json'
        config, summary = load_json(config_path.read_bytes()), load_json(summary_path.read_bytes())
        if config != record['config'] or digest(config_path) != record['files']['config.json'] or digest(summary_path) != record['files']['summary.json']:
            raise ValueError('Comparison training JSON differs from frozen index')
        eligible = [row for row in summary['history'] if row['training_loss'] <= .1]
        if not eligible:
            raise ValueError('Comparison cell did not reach common risk threshold')
        selected = min(eligible, key=lambda row: row['step'])
        result, baseline = outcome_by_id[rid], baseline_by_id[rid]
        for measured in [result, baseline]:
            if (measured['selected_step'] != selected['step']
                    or measured['actual_training_loss'] != selected['training_loss']
                    or measured['checkpoint_sha256'] != record['files'].get(measured['checkpoint'])):
                raise ValueError('Additional evaluation differs from frozen checkpoint selection')
        if result['checkpoint_sha256'] != baseline['checkpoint_sha256']:
            raise ValueError('Predictive and variability evaluations use different checkpoint weights')
        alias_path = alias_paths[rid]
        alias = load_json(alias_path.read_bytes())
        if (alias.get('schema') != 'feature-topology.checkpoint-alias-equivalence.v1'
                or alias.get('run_id') != rid or alias.get('input_index_sha256') != cell['index_sha256']
                or alias.get('input_archive_sha256') != record['archive_sha256']
                or alias.get('selected_step') != selected['step']
                or alias.get('selected_checkpoint_sha256') != result['checkpoint_sha256']
                or alias.get('final_checkpoint_sha256') != record['files']['final.pt']
                or alias.get('exact_model_state_equality') is not True
                or alias.get('auditor_sha256') != digest(Path(repo)/'scripts/remote_checkpoint_equivalence.py')
                or alias.get('frozen_reference_included') is not True):
            raise ValueError('Checkpoint alias audit does not bind the same complete predictor')
        equivalent = {alias['selected_checkpoint_sha256'], alias['final_checkpoint_sha256']}
        candidates = []
        for path in directory.glob('metrics/*/*.json'):
            if path.stem in {'options', 'execution'}:
                continue
            metric = load_json(path.read_bytes())
            if metric.get('step') == selected['step'] and metric.get('checkpoint_sha256') in equivalent:
                candidates.append((path, metric))
        if not candidates:
            raise ValueError('No original metric measurement at selected checkpoint')
        # Original profiles compute PH on final endpoints only. Preserve that
        # existing canonical alias rule; all other flattened measurements agree.
        candidates.sort(key=lambda pair: pair[0].stem != 'final')
        canonical_path, metric = candidates[0]
        canonical_layer = next(layer for layer in metric['layers'] if layer['layer'] == config['depth'])
        canonical = _flatten(canonical_layer, metric)
        for path, other in candidates[1:]:
            other_layer = next(layer for layer in other['layers'] if layer['layer'] == config['depth'])
            alternative = _flatten(other_layer, other)
            excluded = ENDPOINT_ONLY_FIELDS if 'final' in {path.stem, canonical_path.stem} else set()
            if {k:v for k,v in canonical.items() if k not in excluded} != {
                    k:v for k,v in alternative.items() if k not in excluded}:
                raise ValueError('Conflicting original metric checkpoint aliases')
        layers = [layer for layer in metric['layers'] if layer['layer'] == config['depth']]
        if len(layers) != 1 or baseline['layer'] != config['depth']:
            raise ValueError('Comparison requires the last hidden layer')
        row = dict(run_id=rid, seed=record['seed'], gamma=record['gamma'],
                   condition=cell['training_condition'], actual_training_loss=selected['training_loss'],
                   checkpoint_sha256=result['checkpoint_sha256'], **_flatten(layers[0], metric))
        row.update(baseline['metrics'])
        for name in protocol['outcomes']:
            environment = name.removesuffix('_minus_iid_accuracy')
            row[name] = result['environments'][environment]['accuracy']-result['environments']['iid']['accuracy']
        required = ['actual_training_loss', *protocol['comparison_models']['conventional'],
                    *protocol['comparison_models']['fiber_added'], *protocol['outcomes']]
        for name in required:
            if row.get(name) is None or not isinstance(row[name], (int, float)) or not np.isfinite(row[name]):
                missing.append(dict(run_id=rid, field=name, value=row.get(name)))
        rows.append(row)
        for path in [config_path, summary_path, alias_path, *(p for p, _ in candidates)]:
            resolved = path.resolve()
            key = str(resolved.relative_to(repo)) if resolved.is_relative_to(repo) else str(resolved)
            hashes[key] = digest(path)
    for root in [Path(ood), Path(variability)]:
        for path in root.rglob('*.json'):
            resolved = path.resolve()
            key = str(resolved.relative_to(repo)) if resolved.is_relative_to(repo) else str(resolved)
            hashes[key] = digest(path)
    return rows, hashes, missing


def run_comparison(rows, protocol):
    groups = np.array([row['seed'] for row in rows])
    cells = sorted({(row['condition'], row['gamma']) for row in rows})
    design = np.array([[float((row['condition'], row['gamma']) == cell) for cell in cells] for row in rows])
    models = {'design': ['actual_training_loss'],
              'conventional': ['actual_training_loss', *protocol['comparison_models']['conventional']],
              'fiber_added': ['actual_training_loss', *protocol['comparison_models']['conventional'],
                              *protocol['comparison_models']['fiber_added']]}
    scores, predictions, contrasts = {}, [], []
    for outcome in protocol['outcomes']:
        scores[outcome] = {}
        y = np.array([row[outcome] for row in rows])
        for model, columns in models.items():
            numeric = np.array([[row[column] for column in columns] for row in rows])
            prediction, folds = nested_predictions(design, numeric, y, groups,
                                                    protocol['estimation']['ridge_grid'])
            scores[outcome][model] = dict(folds=folds, mean_seed_mse=float(np.mean([f['mse'] for f in folds])))
            predictions.extend(dict(run_id=row['run_id'], seed=row['seed'], outcome=outcome,
                                    model=model, observed=float(value), predicted=float(estimate))
                               for row, value, estimate in zip(rows, y, prediction))
        conventional, fiber = [scores[outcome][model]['folds'] for model in ['conventional', 'fiber_added']]
        differences = np.array([a['mse']-b['mse'] for a, b in zip(conventional, fiber)])
        rng = np.random.default_rng(20261020)
        bootstrap = differences[rng.integers(0, len(differences), size=(2000, len(differences)))].mean(axis=1)
        tail = .05/(2*len(protocol['outcomes']))
        lower, upper = np.quantile(bootstrap, [tail, 1-tail])
        contrasts.append(dict(outcome=outcome, conventional_minus_fiber_mse=float(differences.mean()),
                              seed_differences={str(f['seed']): float(d) for f, d in zip(fiber, differences)},
                              pointwise_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                              family_adjusted_interval=[float(lower), float(upper)],
                              family_adjustment='Bonferroni percentile intervals for three outcomes',
                              incremental_benefit_supported=bool(lower > 0),
                              scope='Held-out seed predictive association; five independent seeds'))
    return dict(scores=scores, contrasts=contrasts, predictions=predictions,
                bootstrap_draws=2000, bootstrap_seed=20261020, model_columns=models,
                design_cells=[list(cell) for cell in cells], independent_seeds=sorted(set(groups.tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--ood', required=True)
    parser.add_argument('--variability', required=True)
    parser.add_argument('--aliases', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol_path = Path(args.repo)/'configs/strengthening_diagnostic_protocol_v1.json'
    protocol = load_json(protocol_path.read_bytes())
    rows, hashes, missing = comparison_rows(args.repo, args.evidence, args.ood, args.variability,
                                           args.aliases, protocol)
    result = dict(schema='feature-topology.diagnostic-comparison.v1', protocol_sha256=digest(protocol_path),
                  status='not_estimable' if missing else 'complete_additional_analysis',
                  cells=len(rows), missing=missing, input_sha256=hashes,
                  analysis_source_sha256={
                      'scripts/diagnostic_comparison.py':digest(Path(__file__)),
                      'research_ext/catalog.py':digest(Path(catalog_module.__file__))},
                  environment=dict(python=platform.python_version(),numpy=np.__version__),
                  scope='Additional source-bound analysis; full original-study freeze still required')
    if not missing:
        result.update(run_comparison(rows, protocol))
    files = {'comparison.json': (json.dumps(result, indent=2, allow_nan=False)+'\n').encode(),
             'comparison_rows.json': (json.dumps(rows, indent=2, allow_nan=False)+'\n').encode()}
    publish(Path(args.output), files)
    print(json.dumps({key: result[key] for key in ['status', 'cells', 'missing']}))


if __name__ == '__main__':
    main()
