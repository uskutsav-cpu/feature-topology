# Feature geometry versus information loss — pilot report

**Project status: in progress.** This report covers 15 completed, analyzed
networks across 5 gamma values. It does not claim completion
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
  Actual final training losses range from 0.0439 to 0.0996;
  these are common stopping-criterion comparisons, not identical-risk checkpoints.
- Metrics use 1000 fixed Jacobian points, 32 fixed NTK points,
  PH subsamples of 150 points with 3 repeats and maxdim 1,
  and independently drawn probe training/test sets of 2000/1000 points.

## Last hidden layer

Values below are means across independent training seeds. Full layer results and
95% seed-bootstrap intervals are in the CSV files next to this report's source data.

| γ | Seeds | Test accuracy | 1−CKA | Raw local q1% | Normalized local q1% | MLP nuisance cosine | Second H1 lifetime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03125 | 3 | 0.9863 | 0.0010 | 2.5780 | 0.8614 | 0.9990 | 0.5068 |
| 0.5 | 3 | 0.9750 | 0.2315 | 2.4641 | 0.5469 | 0.9988 | 0.3560 |
| 1 | 3 | 0.9670 | 0.3103 | 2.4322 | 0.3488 | 0.9991 | 0.2087 |
| 16 | 3 | 0.9753 | 0.3684 | 1.9580 | 0.0413 | 0.9984 | 0.0282 |
| 128 | 3 | 0.9720 | 0.3610 | 2.5710 | 0.0108 | 0.9946 | 0.0160 |

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
