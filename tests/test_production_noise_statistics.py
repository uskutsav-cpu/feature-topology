from pathlib import Path
import copy
import json

import pytest

from scripts.analyze_production_noise import summarize_noise


def collection():
    return json.loads((Path(__file__).resolve().parents[1]/
        'results/production_noise_completion_v1/collection.json').read_text())


def test_noise_views_share_models_without_inflating_training_seed_counts():
    result=summarize_noise(collection())
    assert len(result['rows'])==1220
    assert len(result['seed_confidence_intervals'])==440
    assert len(result['relevance_paired_contrasts'])==280
    assert sum(row['main_view'] for row in result['rows'])==520
    assert sum(row['relevance_view'] for row in result['rows'])==840
    assert sum(row['main_view'] and row['relevance_view'] for row in result['rows'])==140
    assert {row['n_seeds'] for row in result['seed_confidence_intervals'] if row['view']=='main'}=={10}
    assert {row['n_seeds'] for row in result['seed_confidence_intervals'] if row['view']=='relevance'}=={5}
    for row in result['relevance_paired_contrasts']:
        assert row['paired_seeds']==list(range(5))
        assert row['left_only']==row['right_only']==[]
        baseline=next(x for x in result['seed_confidence_intervals'] if x['view']=='relevance'
            and x['gamma']==row['gamma'] and x['noise_level']==row['noise_level']
            and x['metric']==row['metric'] and x['relevance']==0)
        other=next(x for x in result['seed_confidence_intervals'] if x['view']=='relevance'
            and x['gamma']==row['gamma'] and x['noise_level']==row['noise_level']
            and x['metric']==row['metric'] and x['relevance']==row['relevance_right'])
        assert row['mean']==pytest.approx(other['mean']-baseline['mean'])


def test_unreached_noise_selection_is_retained_without_substituting_a_final_probe():
    value=copy.deepcopy(collection());model=value['records'][0]
    model.update(status='unreached_common_risk',noise_rows=[],checkpoint=None,checkpoint_sha256=None,
        selected_step=None,actual_training_loss=None,constant_training_representation=None)
    value.update(evaluated_models=304,unreached_models=1)
    result=summarize_noise(value)
    row=next(x for x in result['rows'] if x['run_id']==model['run_id'])
    assert row['noise_level'] is None and row['angular_cosine'] is None
    assert row['status']=='unreached_common_risk'
    cell=next(x for x in result['seed_confidence_intervals'] if x['view']=='main'
        and x['gamma']==model['gamma'] and x['noise_level']==0 and x['metric']=='angular_cosine')
    assert cell['n_seeds']==9
