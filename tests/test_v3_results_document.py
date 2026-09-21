import json

import pandas as pd

from scripts.v3_results_document import endpoint_effects, render, threshold_audit


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


def test_render_writes_bound_scientific_report_without_upgrading_claims(tmp_path):
    from research_ext.report import METRICS
    config = tmp_path / "configs"
    config.mkdir()
    (config / "analysis_plan_v2.json").write_text(
        '{"descriptive_thresholds":{"geometry_deformed":0.05,'
        '"near_singular_normalized_margin":0.001}}')
    output = tmp_path / "results/final_analysis"
    primary = output / "main/main"
    certification_dir = output / "certification"
    primary.mkdir(parents=True)
    certification_dir.mkdir()
    matched = []
    for gamma in (.125, 128.):
        for seed in range(3):
            row = {"seed": seed, "gamma": gamma, "layer": 4}
            row.update({metric: gamma / 128 + seed / 100 for metric in METRICS})
            matched.append(row)
    pd.DataFrame(matched).to_csv(primary / "matched_risk.csv", index=False)
    pd.DataFrame([
        {"layer": 4, "metric": "cka_drift", "gamma": 128.,
         "lower": .1, "upper": .2},
        {"layer": 4, "metric": "fiber_global_normalized_minimum", "gamma": 128.,
         "lower": .01, "upper": .02},
        {"layer": 4, "metric": "fiber_local_normalized_minimum", "gamma": 128.,
         "lower": .01, "upper": .02},
    ]).to_csv(primary / "seed_bootstrap_ci.csv", index=False)
    pd.DataFrame([{"exact_collision_present": False}]).to_csv(
        certification_dir / "trained_circle_exact_certificates.csv", index=False)
    relevance_dir = output / "relevance_dependence"
    ood_dir = output / "ood_evaluation"
    image_dir = output / "images"
    relevance_dir.mkdir()
    ood_dir.mkdir()
    image_dir.mkdir()
    relevance = []
    for metric in METRICS:
        relevance.append({
            "gamma": 128., "layer": 4, "metric": metric,
            "comparison": "baseline", "relevance_left": 0.,
            "relevance_right": 1., "n_seeds": 3, "mean": .1,
            "lower": .05, "upper": .15,
        })
    (relevance_dir / "relevance_paired_contrasts.json").write_text(
        json.dumps(relevance))
    pd.DataFrame([
        {"training_condition": "iid", "gamma": 128.,
         "evaluation_environment": environment, "metric": metric,
         "n_seeds": 3, "mean": -.1 if metric == "accuracy" else .1,
         "lower": -.15 if metric == "accuracy" else .05,
         "upper": -.05 if metric == "accuracy" else .15}
        for environment in ("concentrated", "spurious", "unseen")
        for metric in ("accuracy", "loss")
    ]).to_csv(ood_dir / "ood_environment_shift_confidence_intervals.csv", index=False)
    pd.DataFrame([
        {"study": "cifar10", "metric": metric, "layer": layer,
         "gamma": 128., "reference_gamma": .125, "n_seeds": 3,
         "mean": .1, "lower": .05, "upper": .15}
        for metric, layer in (("test_accuracy", 0), ("cka_drift", 4),
                              ("effective_rank", 4), ("test_loss", 0))
    ]).to_csv(image_dir / "image_paired_contrasts.csv", index=False)
    certification = {
        "analytic_exact_controls": 7, "rational_trained_layer_certificates": 40,
        "exact_scope": "finite polygon", "numerical_scope": "sampled polygon",
        "formal_scope": "abstract graph layer",
    }
    width = {
        "terminology": "crossover", "phase_transition_language_allowed": False,
        "scaling": {"bootstrap_shrink_support": 0.,
                    "bootstrap_stable_center_support": .989,
                    "largest_to_smallest_transition_width_ratio": 3.334},
    }
    audit = render(tmp_path, output, "f" * 64, [{}],
                   {"relevance_levels": [0., 1.]}, {"runs": 140},
                   certification, width)
    text = (output / "V3_RESULTS.md").read_text()
    assert audit["exact_continuum_task_quotient_established"] is False
    assert "**crossover**" in text
    assert "broadened rather than sharpened" in text
    assert "Controlled image validation" in text
    assert (output / "headline_endpoint_effects.csv").is_file()
    assert (output / "headline_relevance_effects.csv").is_file()
    assert (output / "headline_ood_effects.csv").is_file()
    assert (output / "headline_image_effects.csv").is_file()
    assert (output / "claim_audit.json").is_file()


def test_image_headline_uses_maximal_estimable_gamma_per_metric_layer(tmp_path):
    from research_ext.report import METRICS
    from scripts.v3_results_document import _write_headline_tables

    output = tmp_path
    (output / "relevance_dependence").mkdir()
    (output / "ood_evaluation").mkdir()
    (output / "images").mkdir()
    (output / "relevance_dependence/relevance_paired_contrasts.json").write_text(
        json.dumps([
            {"gamma": 128., "layer": 4, "metric": metric,
             "comparison": "baseline", "relevance_left": 0.,
             "relevance_right": 1., "n_seeds": 2, "mean": 0.,
             "lower": 0., "upper": 0.}
            for metric in METRICS
        ]))
    pd.DataFrame([
        {"training_condition": "iid", "gamma": 128.,
         "evaluation_environment": "unseen", "metric": "accuracy",
         "n_seeds": 2, "mean": 0., "lower": 0., "upper": 0.},
    ]).to_csv(output / "ood_evaluation/ood_environment_shift_confidence_intervals.csv",
              index=False)
    pd.DataFrame([
        {"study": "cifar10", "metric": "test_accuracy", "layer": 0,
         "gamma": gamma, "reference_gamma": .125, "n_seeds": 2,
         "mean": 0., "lower": 0., "upper": 0.}
        for gamma in (64., 128.)
    ] + [
        {"study": "cifar10", "metric": "cka_drift", "layer": 4,
         "gamma": 64., "reference_gamma": .125, "n_seeds": 2,
         "mean": 0., "lower": 0., "upper": 0.}
    ]).to_csv(output / "images/image_paired_contrasts.csv", index=False)
    _, _, images = _write_headline_tables(output)
    assert images.set_index("metric").loc["test_accuracy", "gamma"] == 128.
    assert images.set_index("metric").loc["cka_drift", "gamma"] == 64.
