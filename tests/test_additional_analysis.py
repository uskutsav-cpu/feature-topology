import copy
import pytest

pytest.importorskip('torch')
from scripts.analyze_additional_research import metric_row, summarize


def test_additional_figures_cannot_be_generated_from_an_original_only_freeze(tmp_path):
    with pytest.raises(ValueError, match='combined additional-research freeze'):
        summarize(tmp_path, {'ready': True, 'files': {}})


def test_model_summary_requires_the_same_first_stored_common_risk_checkpoint():
    record = dict(run_id='model', gamma=8., config={'depth': 4})
    witness = dict(selected_step=64, checkpoint_sha256='checkpoint', empirical_screen_passed=True)
    summary = {'history': [dict(step=0, training_loss=1.), dict(step=64, training_loss=.08),
                            dict(step=128, training_loss=.04)]}
    metric = dict(step=64, checkpoint_sha256='checkpoint', gamma=8., seed=0,
        layers=[dict(layer=4, scale=26., effective_rank=3.2,
            fiber_global_separation={'sampled_minimum': .9, 'normalized_sampled_minimum': .035},
            probes={'linear': {'angular_cosine': .999}})])
    row = metric_row(record, witness, summary, metric)
    assert row['achieved_training_loss'] == .08 and row['selected_step'] == 64
    changed = copy.deepcopy(metric)
    changed['checkpoint_sha256'] = 'different checkpoint'
    with pytest.raises(ValueError, match='selection mismatch'):
        metric_row(record, witness, summary, changed)
    changed = dict(witness, selected_step=128)
    with pytest.raises(ValueError, match='selection mismatch'):
        metric_row(record, changed, summary, metric)
