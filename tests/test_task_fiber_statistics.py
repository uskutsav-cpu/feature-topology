"""Synthetic effects test pairing and replicate accounting, not trained results."""
from scripts.analyze_task_preserving_fibers import summarize_task_fibers
import pytest


def synthetic_collection():
    records=[]
    for level in [0.,.05,.1,.25,.5,1.]:
        for gamma in [.125,.5,1.,4.,16.,64.,128.]:
            for seed in range(5):
                families=[]
                for task in [True,False]:
                    value=seed/100+gamma/1000+(level if task else 0)
                    norms=dict(minimum=value,normalized_minimum=value,normalized_q01=value)
                    families.append(dict(family='fixed_u_task_preserving' if task else 'fixed_theta_original_factor_control',
                        centered_representation_rms=1.,local_per_phi=norms,
                        local_per_unit_flat_torus_tangent=norms,local_per_unit_embedded_input_tangent=norms,
                        dense_pairs=dict(phi_arc=norms,flat_torus_ambient=norms,
                            exact_float_collision_pairs=0,tolerance_collision_pairs=0),
                        adversarial_search={'best_candidate':{'normalized_phi_arc_ratio':value}},classifier_grid_accuracy=.75))
                records.append(dict(run_id=f'synthetic-{level}-{gamma}-{seed}',gamma=gamma,seed=seed,relevance=level,
                    status='evaluated',checkpoint_sha256='synthetic',selected_step=64,actual_training_loss=.08,families=families))
    return dict(complete_models=210,evaluated_models=210,unreached_models=0,records=records,scope='Synthetic test effects only')


def test_task_fiber_statistics_keep_families_paired_and_do_not_multiply_seed_count():
    result=summarize_task_fibers(synthetic_collection())
    assert len(result['rows'])==420
    assert len(result['seed_confidence_intervals'])==1260
    assert len(result['family_paired_contrasts'])==630
    assert len(result['relevance_paired_contrasts'])==1050
    assert {r['n_seeds'] for r in result['seed_confidence_intervals']}=={5}
    for row in result['family_paired_contrasts']:
        assert row['paired_seeds']==list(range(5))
        if row['metric']=='local_unit_torus_normalized_minimum':
            assert row['mean']==pytest.approx(row['relevance'])
    for row in result['relevance_paired_contrasts']:
        assert row['paired_seeds']==list(range(5))
        if row['metric']=='local_unit_torus_normalized_minimum':
            assert row['mean']==pytest.approx(row['relevance_right'] if row['family']=='fixed_u_task_preserving' else 0)


def test_unreached_task_fiber_risk_remains_in_accounting_and_does_not_become_a_probe():
    value=synthetic_collection();model=value['records'][0]
    model.update(status='unreached_common_risk',families=[],selected_step=None,actual_training_loss=None)
    value.update(evaluated_models=209,unreached_models=1)
    result=summarize_task_fibers(value)
    rows=[r for r in result['rows'] if r['run_id']==model['run_id']]
    assert len(rows)==2 and all(r['dense_phi_minimum'] is None for r in rows)
    cell=next(r for r in result['seed_confidence_intervals'] if r['relevance']==0 and r['gamma']==.125
        and r['family']=='fixed_u_task_preserving' and r['metric']=='local_unit_torus_normalized_minimum')
    assert cell['n_seeds']==4
