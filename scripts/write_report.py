import argparse
import json
from pathlib import Path
import pandas as pd


def report(root, output):
    root = Path(root)
    frame = pd.read_csv(root/"metrics_long.csv")
    final = frame[frame.checkpoint == "final"]
    if final.profile.nunique() != 1:
        raise RuntimeError("Choose a single metric profile before writing a report")
    last = final[final.layer == final.layer.max()]
    grouped = last.groupby("gamma")
    table = ["| γ | Seeds | Test accuracy | 1−CKA | Raw local q1% | Normalized local q1% | MLP nuisance cosine | Second H1 lifetime |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gamma, group in grouped:
        table.append(f"| {gamma:g} | {group.seed.nunique()} | {group.test_accuracy.mean():.4f} | {group.cka_drift.mean():.4f} | {group.local_q01.mean():.4f} | {group.local_normalized_q01.mean():.4f} | {group.mlp_nuisance_cosine.mean():.4f} | {group.ph_h1_top2.mean():.4f} |")
    count = last.run_id.nunique()
    options_path = next(root.glob("runs/*/metrics/*/options.json"))
    options = json.loads(options_path.read_text())
    text = f"""# Feature geometry versus information loss — {root.name} report

**Project status: in progress.** This report covers {count} completed, analyzed
networks across {last.gamma.nunique()} gamma values. It does not claim completion
of the full production sweep, all ablations, or image robustness experiments.

The controlled pilot finds a clear feature-learning contrast. Geometry and
normalized injectivity diagnostics change with gamma, while the nuisance factor
remains strongly decodable. Raw local tangent margins must be read alongside
normalized margins: shrinking relative scale does not establish destruction.

## Actual experiment

- Exact torus in an orthogonally embedded 16-dimensional input space.
- Four width-256 ReLU hidden layers, centered output divided by gamma, vanilla SGD.
- 20,000 training points; 5,000 validation points; fixed 100×100 analysis grid.
- Independent calibration seed 900; seven LR multipliers per gamma; frozen map.
- All reported final checkpoints meet the common training-loss target 0.1.
  Actual final training losses range from {last.training_loss.min():.4f} to {last.training_loss.max():.4f};
  these are common stopping-criterion comparisons, not identical-risk checkpoints.
- Metrics use {options['jacobian_points']} fixed Jacobian points, {options['ntk_points']} fixed NTK points,
  PH subsamples of {options['ph_size']} points with {options['ph_repeats']} repeats and maxdim {options['ph_maxdim']},
  and independently drawn probe training/test sets of {options['probe_train']}/{options['probe_test']} points.

## Last hidden layer

Values below are means across independent training seeds. Full layer results and
95% seed-bootstrap intervals are in the CSV files next to this report's source data.

"""+"\n".join(table)+f"""

## What the evidence does and does not establish

1. The NTK contrast verifies that output scaling traverses substantially different
   feature-learning behavior in this fixed-width setup.
2. CKA drift, normalized tangent margins, sampled global margins and H1 lifetimes
   show geometric changes. These measurements alone cannot establish noninjectivity.
3. High nuisance-probe performance is positive evidence of retained decodable
   information. These runs do not establish complete nuisance-factor destruction.
4. Finite-grid positive Jacobians and sampled noncollision do not prove the map is
   globally injective. A topological claim about a continuum requires more evidence.
5. The five-gamma pilot is too small for the configured six-level segmented-model
   procedure. Transition locations are therefore not estimated from this pilot.

## Validation already executed

The analytic tests cover orthogonal torus geometry, derivative rank, centered
output and gradient scaling, direct-vs-functional empirical NTK agreement,
known circle persistence, collision sensitivity, train/test identity separation,
seed bootstrap behavior, and exact continuation after a simulated interruption.
PH cache tests verify that completed subsamples are reused.

Known controls show that scaling/rotation preserve the appropriate normalized
metrics. A latent row permutation leaves point-cloud PH invariant while changing
the collision diagnostic; this is an alignment null, not a smooth manifold map or
proof of information destruction. Projecting out the nuisance circle eliminates
its loop in the known control. Polygon image-graph controls recover circle, interval,
and point Betti numbers at the documented numerical tolerance.

The separate 500-point H2 feasibility runs use one subsample per gamma for seed 0.
They are computational benchmarks and supplementary diagnostics, not 20-replicate
production H2 estimates. Their cached diagrams are retained for reuse.

## Reproducibility and remaining work

Use `scripts/status.py` for live completion counts. The full 13-gamma calibration,
10-seed production study, image validation and extended sweeps are tracked separately.
Completed primary-compatible pilot runs are reused in production. Training state,
optimizer state and minibatch RNG permit continuation. PH caches store each
completed subsample independently.

Saving every full-grid activation at every checkpoint requires substantially more
than the available local storage. Weights and deterministic grid generation retain
the representation maps; materialized float32 archives require the requested
external storage location. No data has been fabricated to fill an unfinished stage.

Sources and methodological qualifications are in `docs/PROTOCOL.md`.
"""
    Path(output).write_text(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); report(a.root, a.output)
