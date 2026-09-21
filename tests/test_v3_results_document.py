import pandas as pd

from scripts.v3_results_document import endpoint_effects, threshold_audit


def test_endpoint_effects_pair_training_seeds_and_cover_every_metric(tmp_path):
    rows = []
    for gamma in (0.125, 128.0):
        for seed in (0, 1, 2):
            row = {"seed": seed, "gamma": gamma, "layer": 4}
            from research_ext.report import METRICS
            row.update({metric: gamma + seed for metric in METRICS})
            rows.append(row)
    path = tmp_path / "matched_risk.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = endpoint_effects(path, repeats=20)
    assert set(result.metric) == set(METRICS)
    assert set(result.delta_n_seeds) == {3}
    assert all(abs(value - 127.875) < 1e-12 for value in result.delta_mean)


def test_threshold_audit_never_upgrades_sampled_margin_to_exact_claim():
    ci = pd.DataFrame([
        {"layer": 4, "metric": "cka_drift", "gamma": 1.,
         "lower": .06, "upper": .2},
        {"layer": 4, "metric": "fiber_global_normalized_minimum", "gamma": 1.,
         "lower": .002, "upper": .004},
        {"layer": 4, "metric": "fiber_local_normalized_minimum", "gamma": 2.,
         "lower": .0001, "upper": .0009},
    ])
    exact = pd.DataFrame([{"exact_collision_present": False}])
    result = threshold_audit(ci, exact, .05, .001)
    assert result["deformation_with_positive_sampled_global_margin_gammas"] == [1.]
    assert result["near_singular_cells"] == [
        {"gamma": 2., "metric": "fiber_local_normalized_minimum"}]
    assert result["exact_continuum_task_quotient_established"] is False
