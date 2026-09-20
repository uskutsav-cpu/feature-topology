import json
import pytest
from src.training.train import train
from src.training.checkpoints import fingerprint
from scripts.compute_metrics import analysis_condition, compute
from scripts import compute_metrics


def test_centered_initial_metrics_reused_across_gamma(tmp_path):
    root = tmp_path/"results"/"experiment"/"runs"
    base = dict(lr=.01, seed=0, width=8, depth=1, n_train=64,
                n_validation=32, n_grid=25, max_steps=2)
    options = dict(jacobian_points=25, ntk_points=4, ph_size=20, ph_repeats=1,
                   ph_maxdim=1, probe_train=80, probe_test=40, probe_iterations=10,
                   all_checkpoints=False)
    paths = []
    for gamma in [1., 2.]:
        result = train({**base, "gamma":gamma}, root)
        path = root/result["run_id"]
        compute(path, options)
        paths.append(path/"metrics"/fingerprint(options)/"step_0000000.json")
    first, second = [json.loads(p.read_text()) for p in paths]
    assert "initial_metric_reused" in second
    assert first["layers"] == second["layers"]
    assert first["layers"][0]["fiber_local_margin"]["samples"] == 25
    assert first["layers"][0]["fiber_global_separation"]["pairs"] == 50
    assert "normalized_sampled_minimum" in first["layers"][0]["fiber_global_separation"]
    assert first["layers"][0]["fiber_empirical_regime"] in {
        "preserved", "deformed", "near-singular", "sampled-collision"
    }
    assert second["gamma"] == 2.
    # A fully populated cache must still validate the checkpoint bytes.
    compute(path, options)
    with (path/"final.pt").open("ab") as stream:
        stream.write(b"changed checkpoint bytes")
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        compute(path, options)


def test_analysis_condition_distinguishes_nuisance_environments():
    assert analysis_condition({})["nuisance_condition"] == "iid"
    assert analysis_condition({"nuisance_condition": "spurious"})["nuisance_condition"] == "spurious"


def test_endpoint_ph_schedule_skips_intermediate_checkpoints(tmp_path):
    config = dict(lr=.001, seed=0, gamma=1., width=8, depth=1, n_train=64,
                  n_validation=32, n_grid=25, max_steps=2, target_loss=1e-8)
    result = train(config, tmp_path/"runs")
    run = tmp_path/"runs"/result["run_id"]
    options = dict(jacobian_points=25, ntk_points=4, ph_size=20, ph_repeats=1,
                   ph_maxdim=1, probe_train=80, probe_test=40, probe_iterations=10,
                   all_checkpoints=True, ph_schedule="endpoints")
    compute(run, options)
    directory = run/"metrics"/fingerprint(options)
    middle = json.loads((directory/"step_0000001.json").read_text())
    assert middle["layers"][0]["persistence"] == []
    assert not (directory/"step_0000001_layer1_ph.npz").exists()
    final = json.loads((directory/"final.json").read_text())
    assert len(final["layers"][0]["persistence"]) == 1


def test_baseline_uses_saved_initial_weights_not_local_rng(tmp_path,monkeypatch):
    config=dict(lr=.01,seed=0,gamma=1.,width=8,depth=1,n_train=64,
                n_validation=32,n_grid=25,max_steps=1)
    result=train(config,tmp_path/'runs')
    run=tmp_path/'runs'/result['run_id']
    original=compute_metrics.build
    monkeypatch.setattr(compute_metrics,'build',lambda c,s:original(c,s+111))
    options=dict(jacobian_points=25,ntk_points=4,ph_size=20,ph_repeats=1,
                 ph_maxdim=1,probe_train=80,probe_test=40,probe_iterations=10,all_checkpoints=False)
    compute(run,options)
    row=json.loads((run/'metrics'/fingerprint(options)/'step_0000000.json').read_text())
    assert row['ntk_drift']==pytest.approx(0.,abs=1e-7)
    assert row['layers'][0]['cka_drift']==pytest.approx(0.,abs=1e-7)
