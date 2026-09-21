# Completion work, 13 September 2026

This is an execution record. The complete study is **not frozen or finished**.

The archived original workspace was ahead of the GitHub snapshot. Its results
were recovered into a separate writable workspace, leaving the archive read-only.
The original Python environment is available: Python 3.13 and PyTorch 2.14.0,
NumPy 2.5.3 and Ripser 0.6.15. The separate 39-run CPU continuation remains a
separate experiment and does not replace any primary seed.

## Verified recovery

- All 130 requested primary gamma/seed pairs have converged summaries and final
  checkpoints. Configurations, run fingerprints, final steps, finite tensors,
  target-loss statuses and checkpoint SHA-256 values were checked.
- The archive contained 170/175 width entries, 35/35 rotated-digit production
  runs with metrics, and 20 trained circle networks with numerical quotient results.
- Held-out classification was replayed from all 35 digit checkpoints; all matched
  the saved metrics within the recorded tolerances. The digit and circle source
  checkpoints are included in this update.
- Shared baseline runs also account for 35 depth entries and 35 relevance entries.
- Recovered primary metrics include H1-only initial/final profiles. These do not
  satisfy the full H2 and all-checkpoint production protocol.

The five remaining width runs and the full cylinder and swapped-factor training
sweeps have subsequently completed. Depth, relevance and small-network training
continue with their own frozen condition-specific calibrations. Run
`scripts/completion_inventory.py` for current artifact-backed counts; process exit
alone is not treated as scientific completion.

## Exact and formal evidence

`scripts/certify_trained_circles.py` exported the exact stored finite weights from
20 trained circle networks, computed 40 certificates (both hidden layers), and
replayed each certificate independently with rational arithmetic. The explicit
domain is the ten-vertex rational polygon recorded in every certificate. This
is neither a smooth-circle nor a smooth-torus certificate and does not describe
IEEE floating-point inference semantics.

All 40 certificates have an injective map on their stated polygon domain and
image-graph counts (1 component, 1 cycle). This is a restricted-domain result;
it does not establish global injectivity for the synthetic torus study.

The generated Lean example now checks a 38-vertex graph from one of these trained
network certificates. An executable component-label merging routine replaced
the costly all-pairs reachability calculation for graph counts. Connected,
disconnected, reordered-edge and invalid-edge examples are kernel checked.
All 17 audited declarations compile without axioms. Lean verifies the supplied
abstract graph and logical lemmas; the Python network-to-graph construction is
not a Lean theorem. The build log and source hashes are in
`results/completion/formal/`.

## Running computation

```sh
python scripts/complete_synthetic_training.py
python scripts/compute_metrics.py --runs results/main/runs --all-checkpoints
python scripts/run_dsprites.py --device mps
python scripts/complete_cifar.py
```

The synthetic driver has a single-writer lock and per-study logs. The metrics job
uses the full 10,000-point Jacobian, 128-point NTK, and 500-point, 20-repeat H2
profile. New metric rows bind checkpoint bytes and the metric code/environment.
Caches without that provenance are retained but rejected by this new runner.
Use a separate result workspace when changing metric code or environment.

The dSprites runner verifies the official archive's Git blob identity. Its fixed
design uses all scales and orientations at four declared positions per axis.
Shape/scale/position identities are partitioned before their orientations, with
6,960 training and 2,280 validation/test images each. Images remain 64×64.
Separate orientation probes use symmetry multipliers 4, 2 and 1 for square,
ellipse and heart; their auxiliary sector accuracy is not shape accuracy.
The GPU pooling implementation uses the same adaptive bins as PyTorch and is
tested against reference values and gradients. This is a planned 35-run study,
not yet a completed experiment.

Linear CKA keeps its original definition. If a learned image representation is
exactly constant, its centered Gram norm is zero and CKA is mathematically
undefined rather than a finite drift score. Such a result is retained as null
with an explicit zero-variance status and replayed diagnostics; it is never
imputed. Final numeric CKA summaries exclude only these declared undefined
seed-level values, report their actual replicate count, and list every exclusion
separately. Effective rank and PH continue to record the collapse itself.

The initial explicit-bin fallback was replaced during calibration with a
separable linear implementation and a head-only training forward pass. CPU and
MPS value/gradient checks passed. Provisional fallback calibration files are
preserved under their distinct configuration IDs; production uses
`separable_adaptive_mean_v2` and a separately selected rate map.

The CIFAR queue downloads and validates both official datasets, then waits for
the dSprites manifest before using the GPU. Both CIFAR10 and CIFAR100 retain the
seven-gamma/five-seed scope and the original calibration and maximum-update
budgets. On macOS Python installations without configured certificate roots,
set `SSL_CERT_FILE=/etc/ssl/cert.pem`; TLS verification stays enabled.

## Freeze, analysis and release

Analysis choices are recorded in `configs/completion/analysis_plan.json`. They
were recorded after pilot and recovered H1-only results, so are exploratory,
not a preregistration. No parameter is tuned to force nuisance loss.

```sh
python scripts/completion_inventory.py --output results/completion/inventory.json --check-tensors
python scripts/freeze_results.py --check-only --output results/completion/readiness.json
# Only when readiness passes:
python scripts/freeze_results.py --output results/frozen_manifest.json
python scripts/final_analysis.py --manifest results/frozen_manifest.json
python scripts/pack_release.py --manifest results/frozen_manifest.json \
  --analysis-manifest results/final_analysis/analysis_manifest.json \
  --output release-assets
```

When retaining all compressed parts would exceed local free space, first create
the final GitHub release at the already-pushed frozen tag, then use the bounded-
space publication mode:

```sh
python scripts/pack_release.py --manifest results/frozen_manifest.json \
  --analysis-manifest results/final_analysis/analysis_manifest.json \
  --output release-assets --upload-release production-v3-frozen
```

Each part is uploaded without clobbering, checked against GitHub's server-side
SHA-256 digest, and only then removed locally. Source results, checkpoints,
caches, the package index, and the frozen manifests are never deleted.

The freeze requires every design, production trajectory, image study, trained
certificate and formal audit. Final statistics/figures require verified frozen
inputs. The release packer creates deterministic gzip/tar parts of at most 1 GiB,
with checksums, and refuses changed frozen inputs or conflicting remote assets.
Final image-study synthesis
and the paper remain dependent on completed experiments and the results freeze.

GitHub Actions checks Python tests and the Lean build/audit. Successful CI is an
engineering check, not evidence that long-running studies have completed.

## Hosted production metrics

The `Production metrics batch` workflow evaluates up to 24 explicitly selected
primary runs per batch, with at most eight standard Ubuntu workers concurrently.
It is gated to public repositories. The first batch is a one-run validation;
larger batches follow only after its artifact checks pass.

`configs/completion/primary_inputs.json` binds all 130 input archives and every
checkpoint member to SHA-256 values. The shared analysis grid and probe arrays
are also frozen by checksum. Metrics load the saved initialization checkpoint,
so cross-platform RNG differences cannot silently change the baseline. Hosted
production uses Python 3.13.5 and the pinned dependency lock; host and source
provenance accompany the outputs. Earlier local full-profile diagnostics remain
separate from this production cohort.

Workers stop computation before the hosted job limit, package all completed
metric rows plus partial PH caches, and upload a complete/partial status manifest.
Compatible previous artifacts can be resumed. Training-input and in-progress
metric prereleases are explicitly not the final research release.

The CIFAR runner also now checks resume configurations, final artifacts and the
frozen calibration protocol, resumes terminal checkpoints without another update,
and detaches Jacobians before NumPy export. These changes were made before the
queued CIFAR studies started.

### Hosted CIFAR continuation

The original single-worker MPS path was stopped only after one complete
2,000-step calibration trial had been written. That trial and the next cell's
configuration are retained unchanged under
`results/cifar10/calibration_mps_partial_2026_09_20`; their hashes and the reason
for the handoff are recorded in `results/completion/cifar_mps_handoff.json`.
They are not mixed into the hosted CPU learning-rate selection.

The replacement hosted path reconstructs the same seven gammas, seven
calibration multipliers, five production seeds, selection rule, ResNet, training
budgets, and image metrics from the original runner. Each cell is bound to a
fixed source tag and an official CIFAR archive SHA-256, saves the existing
500-step resume checkpoint, and publishes an immutable checksum manifest to a
work-in-progress GitHub release. Collection validates the entire archive before
additive installation and refuses a conflicting local file. Terminal optimizer
resumes present in early hosted archives are validated but not duplicated into
the local frozen-results tree; later workers omit that redundant terminal-only
state while still retaining every partial resume needed for recovery. Final
checkpoints, summaries, metrics, representations, and provenance are installed.
This is an infrastructure change, not a protocol or analysis change.

The final freeze requires complete 49-cell calibration and 35-cell production
ledgers for both CIFAR-10 and CIFAR-100, a single source commit per result tag,
source-specification agreement with the local dataset-provenance record, and the
existing full CPU held-out replay of all 70 production checkpoints.
