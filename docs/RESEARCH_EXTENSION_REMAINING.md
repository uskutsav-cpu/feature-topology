# What remains before calling the research finished

The source base audited here was commit
`d6c3f34b41feab45d7d963228c5c3e327098203b`. At the inspection time, the committed
main manifest contained 47 converged entries toward the 13-gamma × 10-seed design.
That is a snapshot, not a live view of processes on another machine. Work done
locally but not pushed is not visible in that count.

## Required empirical work

Resume the original calibrated primary training with its frozen rate map and
pilot gate. Check all 130 requested run outcomes and preserve divergence/budget
exhaustion as outcomes. Then compute the production-resolution metrics at all
stored checkpoints. A successful process exit is not enough: inspect the coverage
and missing-artifact reports. Main studies, all ablation conditions and full
image studies have not been run by this extension's delivery validation.

Run width, depth, factor-relevance, swapped-label, cylinder and small-network
ablations through the separate plan. Existing condition-specific calibrations
must be retained. The wrapper does not add missing scientific controls or repair
unseen problems in the original training/metric algorithms.

The `images` plan wraps the existing offline rotated-digits study. The `cifar`
plan wraps the existing CIFAR10 and CIFAR100 studies and expects datasets already
present: it does not silently download them. Those upstream image runners expose
CPU and Apple MPS, not CUDA. The wrapper does not add CUDA support. dSprites still
has no completed, validated study in this deliverable. A loader/config is not a
finished experiment.

High-resolution persistent homology (including the full H2 protocol), calibration
robustness and image results still require their actual runs and inspection. The
validation environment had no Ripser installed. The tiny smoke test deliberately
omits persistence and nonlinear-probe measurements rather than fabricating them.

## Required formal work

Install the pinned Lean toolchain and execute `check-formal`. Resolve any build
errors. Audit the actual statements and their assumptions, not just the absence
of placeholders. Extend the proof from an abstract finite graph to the exact
piecewise-affine network-to-graph construction before claiming a formally
verified neural-network topology result. Smooth-torus conclusions need additional
analysis; the current polygon/box certificates do not provide them.

## Required interpretation and paper work

Freeze the analysis choices before inspecting new production outcomes where
possible. Choices introduced after seeing the pilot must be labeled accordingly.
Report the full exclusion/nonconvergence pattern and distinguish common-threshold
from identical-risk comparisons. A nuisance factor that remains recoverable is
an acceptable outcome, not a failed experiment to be retuned away.

Review novelty and related work, select defensible claims, write the paper and
obtain mentor review. This package does not claim publication readiness, novel
formal results, universal nuisance destruction, or a proven phase transition.

## Operational cautions

Do not run the new workflow concurrently with old workers on the same result
directories. The new workflow lock coordinates new workflow invocations only; it
cannot control already-running old scripts on another machine. Stop or wait for
those workers before starting a competing writer. Do not delete an existing lock
until its owner has actually stopped. Changing budgets/configuration changes
experiment identity; this package does not silently modify frozen runs.

Existing cached metrics remain governed by the original compute script's cache
semantics. New wrapper hashes detect changes between wrapper executions, but do
not retrospectively prove the correctness of older cached computations. Inspect
cache provenance before reusing results after a scientific implementation change.
