import json
from src.training.train import train
from src.training.checkpoints import fingerprint
from scripts.compute_metrics import compute


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
    assert second["gamma"] == 2.
