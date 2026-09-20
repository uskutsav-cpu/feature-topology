import numpy as np
import pandas as pd

from research_ext.finite_width import evaluate_gate, fit_width_curve, select_gate_rows


def specification(repeats=80):
    return {
        "primary_metric": "fiber_global_normalized_minimum",
        "expected_direction": "decreasing",
        "widths": [64, 128, 256, 512, 1024],
        "gammas": [.125, .5, 1., 4., 16., 64., 128.],
        "seeds": [0, 1, 2, 3, 4],
        "bootstrap_repeats": repeats,
        "gate": {
            "minimum_delta_aicc_each_width": 6.,
            "maximum_width_spearman_rho": -.8,
            "maximum_largest_to_smallest_transition_width_ratio": .75,
            "minimum_bootstrap_shrink_support": .9,
            "maximum_largest_width_center_drift_log2": 1.,
            "minimum_bootstrap_stable_center_support": .9,
            "require_all_centers_interior": True,
        },
        "scope": "test",
    }


def frame(shrinking=True):
    spec = specification()
    rows = []
    x = np.log2(spec["gammas"])
    rng = np.random.default_rng(7)
    for wi, width in enumerate(spec["widths"]):
        scale = (.9-.13*wi) if shrinking else .9
        center = 2.0+.08*wi
        for seed in spec["seeds"]:
            y = 1.-.8/(1.+np.exp(-(x-center)/scale))
            y += rng.normal(0, .002, len(x))
            for gamma, value in zip(spec["gammas"], y):
                rows.append(dict(width=width, gamma=gamma, seed=seed,
                                 fiber_global_normalized_minimum=value))
    return pd.DataFrame(rows)


def test_fit_recovers_monotone_center_and_width():
    gamma = np.array([.125, .5, 1., 4., 16., 64., 128.])
    x = np.log2(gamma)
    values = 1.-.8/(1.+np.exp(-(x-2.)/.7))
    fit = fit_width_curve(gamma, values, "decreasing")
    assert fit["status"] == "ok"
    assert abs(fit["transition"]["center_log2_gamma"]-2.) < 1e-4
    assert abs(fit["transition"]["scale_log2_gamma"]-.7) < 1e-4


def test_gate_allows_sharpening_stable_transition():
    result = evaluate_gate(frame(shrinking=True), specification())
    assert result["status"] == "complete"
    assert result["phase_transition_language_allowed"]
    assert all(result["clauses"].values())


def test_gate_falls_back_to_crossover_without_sharpening():
    result = evaluate_gate(frame(shrinking=False), specification())
    assert result["status"] == "complete"
    assert not result["phase_transition_language_allowed"]
    assert result["terminology"] == "crossover"


def test_gate_fails_closed_on_missing_cell():
    result = evaluate_gate(frame().iloc[:-1], specification(repeats=2))
    assert result["status"] == "incomplete"
    assert result["observed_cells"] == 174
    assert not result["phase_transition_language_allowed"]


def test_gate_ignores_prespecified_out_of_design_gamma_cells():
    data = frame()
    extras = data[(data.width == 256) & (data.gamma == .125)].copy()
    extras["gamma"] = .25
    result = evaluate_gate(pd.concat([data, extras], ignore_index=True), specification(repeats=5))
    assert result["status"] == "complete"
    assert len(result["ignored_out_of_design_cells"]) == 5


def test_metric_layer_numbering_is_one_indexed():
    # Production metric rows label hidden layers 1..depth, so the frozen
    # "last_hidden" selector must use equality rather than depth-1.
    shared=dict(width=64,profile='ablation',manifold='torus',swap=False,
                relevance=0.,nuisance_condition='iid',depth=4)
    rows=pd.DataFrame([{**shared,'layer':3},{**shared,'layer':4}])
    assert select_gate_rows(rows,'primary','ablation').layer.tolist()==[4]
