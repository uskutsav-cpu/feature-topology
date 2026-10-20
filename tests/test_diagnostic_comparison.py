import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.collect_additional_evaluations import validate_cell
from scripts.diagnostic_comparison import grouped_folds, nested_predictions
from scripts.remote_nuisance_evaluation import evaluation_plan


def test_outer_seed_outcomes_cannot_influence_its_predictions_or_hyperparameter():
    rng = np.random.default_rng(39)
    groups = np.repeat(np.arange(5), 12)
    numeric = rng.normal(size=(60, 3))
    design = np.tile(np.eye(12), (5, 1))
    y = 2*numeric[:, 0]-.4*numeric[:, 1]+rng.normal(size=60)*.05
    original, folds = nested_predictions(design, numeric, y, groups, [.01, 1, 100])
    changed_y = y.copy()
    changed_y[groups == 3] += 10000
    changed, changed_folds = nested_predictions(design, numeric, changed_y, groups, [.01, 1, 100])
    assert np.array_equal(original[groups == 3], changed[groups == 3])
    a, b = [next(fold for fold in values if fold['seed'] == 3)
            for values in [folds, changed_folds]]
    assert a['alpha'] == b['alpha']
    assert a['inner_mse'] == b['inner_mse']
    assert 3 not in a['training_seeds']
    for seed, train, test in grouped_folds(groups):
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(groups[test]) == {seed}


def test_nested_estimator_recovers_a_known_predictive_control_on_unseen_seeds():
    rng = np.random.default_rng(320)
    groups = np.repeat(np.arange(5), 20)
    numeric = rng.normal(size=(100, 2))
    design = np.tile(np.eye(20), (5, 1))
    outcome = 3*numeric[:, 0]+np.tile(np.linspace(-1, 1, 20), 5)
    estimates, _ = nested_predictions(design, numeric, outcome, groups, [.0001, .01, 1])
    control, _ = nested_predictions(design, np.ones((100, 1)), outcome, groups, [.0001, .01, 1])
    assert np.mean((outcome-estimates)**2) < .001
    assert np.mean((outcome-control)**2) > 3


def test_comparison_does_not_impute_nonfinite_values():
    with pytest.raises(ValueError, match='no implicit imputation'):
        nested_predictions(np.ones((6, 1)), np.full((6, 1), np.nan), np.arange(6),
                           np.repeat(np.arange(3), 2), [.1])


def test_collection_rejects_a_rehashed_result_using_a_foreign_checkpoint():
    repo = Path(__file__).resolve().parents[1]
    row = evaluation_plan(repo)[0]
    record = row['record']
    checkpoint = next(name for name in record['files'] if name.startswith('step_'))
    result = dict(schema='feature-topology.nuisance-shift-evaluation.v1',
                  run_id=record['run_id'], seed=record['seed'], gamma=record['gamma'],
                  training_nuisance_condition=row['training_condition'], checkpoint=checkpoint,
                  checkpoint_sha256=record['files'][checkpoint], target_loss=.1, status='evaluated',
                  environments={name: dict(samples=5000, accuracy=.5, loss=.7)
                                for name in ['iid', 'concentrated', 'spurious', 'unseen']})
    receipt = dict(run_id=record['run_id'], input_index=row['index'], input_tag=row['input_tag'],
                   input_index_sha256=row['index_sha256'], input_archive_sha256=record['archive_sha256'],
                   source_member_sha256=record['files'], evaluation_sha256='hash-of-result')
    validate_cell(row, result, receipt, 'hash-of-result', 'ood', repo)
    corrupt = copy.deepcopy(result)
    corrupt['checkpoint_sha256'] = 'foreign-weights'
    with pytest.raises(ValueError, match='frozen sources'):
        validate_cell(row, corrupt, receipt, 'hash-of-result', 'ood', repo)
