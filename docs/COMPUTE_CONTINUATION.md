# Feature-topology: computational continuation

**Date:** 2026-09-08  
**Base repository:** `uskutsav-cpu/feature-topology`  
**Base commit:** `d6c3f34b41feab45d7d963228c5c3e327098203b`  
**Status:** code changes and computations completed locally; remote write rejected with HTTP 403, `Resource not accessible by integration`. No remote commit or push is claimed.

This is an engineering and computation record, not a manuscript. It preserves the original pilot and production datasets unchanged.

## Completed work

- 39 full-size networks trained and converged: all 13 gamma values, three new seeds (100, 101, 102).
- 39 final-checkpoint analyses, comprising 156 hidden-layer records and 156 noise-probe records.
- 24 known-map analytic controls: four amplification levels × two map types × three seeds.
- 41 tests passed in the available core/regression suite; one real-Ripser test skipped and one original PH test deselected because Ripser is unavailable. Other original image/quotient tests were not reconstructed or executed. No claim of a fully executed upstream test suite or CI run.
- All 39 final checkpoint SHA-256 values verified against both the manifest and analysis records; code and option provenance verified.

## Corrections to the execution pipeline

### Terminal restart semantics

An interruption after saving a target-reaching recovery checkpoint but before saving the final summary could cause an additional optimizer update on restart. Two reproductions failed against the original code (terminal steps 0 and 4 resumed to steps 1 and 6). The patched loop recognizes a target already reached at the saved step, restores a missing terminal checkpoint when necessary, and finalizes without another update. Bitwise final-state regression tests pass. Configuration mismatches in recovery checkpoints are rejected.

### Concurrent output writes

JSON, PyTorch checkpoints, persistent-homology cache files, and initial NTK cache arrays now use uniquely named temporary files in the destination directory followed by atomic replacement. Failed writes preserve the previous destination. This prevents shared temporary-filename collisions. It is an atomic last-writer-wins strategy, **not a run-level ownership lock**; do not deliberately launch two training workers for the same run.

### Stable metric sharding

Worker assignment is based on SHA-256 of the run ID rather than a run's position in a changing directory list. Adding new runs no longer moves existing runs between shards. The existing production script uses this selector.

### Explicit PH dependency handling

Non-PH utilities can be imported without Ripser. Previously completed compatible PH caches can still be read; an uncached request without the real backend raises a clear error. No replacement diagrams or synthetic PH outputs were generated. PH I/O unit tests use a labelled stub only to test file handling, not to generate research evidence.

### Continuation safeguards

The independent runner refuses to reuse a result directory with a changed study configuration or numerical environment. Limited reruns preserve the completed run manifest. Missing final checkpoints are reported rather than counted as complete. Analysis records bind checkpoint bytes, options, and numerical source hashes. The summary step rejects incomplete grids, duplicates, and mixed analysis profiles. Analysis `--limit` deliberately writes a partial top-level diagnostic summary; use the full analysis command before producing the complete summary (per-run diagnostic caches remain reusable).

## Exact experiment and scope

The training design retains the original 16-dimensional orthogonal torus embedding, four width-256 ReLU hidden layers, four theta-sector classes, centered output `(f_theta - f_initial)/gamma`, vanilla SGD, 20,000 training samples, 5,000 validation samples, and the fixed 100×100 analysis grid. Maximum updates: 8192; target training loss: 0.1; evaluation interval: 128, with the original additional checkpoint schedule.

Learning rates are copied from the frozen upstream calibration, not retuned on new results. Calibration ID: `d94b0112a7aa3e46`; source blob: `70c1445cb52138c2f90987a5c805bf3b5c826f4c`.

**New seeds are 100, 101, 102.** These 39 runs do not complete or replace the original 13×10 production sweep. They are a separate CPU replication, stored under `results/continuation`, not `results/main`.

Numerical environment: Python 3.13.5; PyTorch 2.10.0+cpu; NumPy 2.3.5; SciPy 1.17.0; scikit-learn 1.8.0. The original snapshot used a different platform/PyTorch version. Do not merge results silently across environments.

Final losses span **0.037553 to 0.099734**. This is a common stopping-threshold comparison, **not identical-risk matching**. Only final representations are analyzed in this continuation.

Metric resolution is explicitly reduced: 1600 grid points, 256 tangent-Jacobian points, 16 empirical NTK points, 2000 probe-training and 1000 probe-test samples, 200 probe iterations, 512 nuisance-fiber pairs. All four hidden layers are measured. This profile does **not** compute PH, the original random-pair global margin/kNN collision profile, image validation, or exact quotient homology.

## Added diagnostics

**Factor-controlled pairs:** hold theta fixed and shift phi by pi. Compare representation separation in raw units and after division by representation RMS. This is more specific to nuisance retention than mixing arbitrary task and nuisance differences. No floating-point equal pair was observed in the sampled pairs; absence of sampled collisions is not a global-injectivity proof.

**Finite-noise ridge probes:** circular targets are `(cos(phi), sin(phi))`. Independent isotropic Gaussian noise is added to training/test representations before fitting a training-only StandardScaler and ridge probe. For level epsilon, each coordinate's noise SD is `epsilon * training_RMS / sqrt(width)`. Therefore the expected Euclidean noise norm is epsilon times training representation RMS. One Gaussian draw is paired across levels. The probe is refitted at each noise level. This is noise on hidden representations, not noise on pixels or SGD training, and it is not a clean-trained-probe distribution-shift test.

## Measured results

Means across three independent training seeds; scores below are **angular cosines, not percentage accuracies**.

| γ | Clean ridge cosine | Noisy ridge cosine, 10% RMS | Clean MLP cosine | Normalized local q1% |
|---:|---:|---:|---:|---:|
| 0.03125 | 0.999867 | 0.999599 | 0.999114 | 0.872186 |
| 1 | 0.999799 | 0.998465 | 0.999292 | 0.374958 |
| 16 | 0.998864 | 0.948002 | 0.998343 | 0.050134 |
| 32 | 0.997651 | 0.819330 | 0.997673 | 0.028298 |
| 64 | 0.997987 | 0.785003 | 0.998154 | 0.023882 |
| 128 | 0.994562 | 0.434828 | 0.990909 | 0.011006 |

Clean nuisance decoding stays strong, including at gamma 128. In contrast, the same ridge-probe protocol with a 10%-RMS noise norm declines substantially at large gamma. This is evidence of **noise-sensitive recoverability under this intervention**, not proof that the network has deleted the nuisance variable.

The analytic controls make the distinction explicit. For every finite A>0, `(A*cos(theta), A*sin(theta), cos(phi), sin(phi))` is injective on the product of circles. Amplifying the task coordinates can nevertheless reduce normalized tangent margins and make nuisance decoding fragile at fixed relative noise. Projecting out both phi coordinates is the separate known noninjective control. These constructed examples validate interpretation; they are not claimed as novel theorems or discoveries.

All nonlinear probe fits in these 39 runs completed without recorded convergence warnings. Three seeds and one noise draw per seed are not enough for a universal claim. The standard-deviation error bars in `noise_sensitivity.png` are descriptive seed variability, not confidence intervals.

## What remains unproven or unfinished

- The original 130-run production sweep and all-checkpoint production metrics are not finished by this continuation.
- No new persistent-homology analysis was executed; the dependency is unavailable here. No H1/H2 conclusion is inferred from the new runs.
- A small normalized margin, failed probe, or strong noise sensitivity is not a certificate of noninjectivity. Positive finite-grid margins and successful probes do not certify global injectivity either.
- We analyze the trainable branch's hidden states. The centered model also has a frozen reference branch; statements about information in the complete two-branch model require separate analysis.
- No trained circle-quotient study, rotated-image study, dSprites/CIFAR experiment, width/depth sweep, or continuum theorem was completed in this continuation.
- The added noise experiment is exploratory. Its apparent effect should be checked with more noise seeds, nonlinear noisy probes, task probes under the same noise, matched-risk checkpoints, and secondary datasets before making broader robustness claims.

## Reproduce and continue

Use a separate result directory on a different numerical environment; do not mix or overwrite the supplied records. Commands from the repository root:

```sh
python scripts/run_continuation.py --output results/continuation_new_environment
python scripts/analyze_continuation.py --root results/continuation_new_environment
python scripts/summarize_continuation.py --root results/continuation_new_environment
python scripts/run_information_controls.py --output results/continuation_new_environment/information_controls.json
python scripts/plot_continuation.py --root results/continuation_new_environment
```

The available core/regression suite, with explicit absent-PH handling in this environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_resume.py tests/test_core.py tests/test_terminal_resume.py tests/test_atomic_sharding.py tests/test_fibers.py tests/test_persistence_io.py tests/test_continuation_runner.py -q -k 'not known_circle_ph'
```

With all upstream dependencies installed, run the entire original suite as well. Stop existing workers before replacing their source files. Do not force-push or overwrite newer local changes.

## Artifact guide

- `manifest.json`: run status and checkpoint hashes.
- `diagnostics.json`: complete final-profile numerical records.
- `runs.csv`, `layer_metrics.csv`, `noise_metrics.csv`: analysis-ready measured data.
- `gamma_summary.csv`, `overview.json`: derived summaries with explicit scope.
- `information_controls.json`: all 24 analytic controls.
- `environment.json`, `study_config.json`, `software_versions.json`: provenance.
- `tests.txt`, `regression_before.txt`, `verification.json`: validation evidence.
- The lightweight update package contains text results, source changes and a guarded commit/apply script. The separate compute archive contains the continuation workspace and all saved model checkpoints; it is not a full clone of the original repository.

## Primary references

1. Upstream frozen experiment: https://github.com/uskutsav-cpu/feature-topology/tree/d6c3f34b41feab45d7d963228c5c3e327098203b
2. PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness
3. Python temporary-file APIs: https://docs.python.org/3/library/tempfile.html
4. GitHub REST permission troubleshooting: https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api

**Publication state:** no manuscript generated; no remote code write succeeded in this session.
