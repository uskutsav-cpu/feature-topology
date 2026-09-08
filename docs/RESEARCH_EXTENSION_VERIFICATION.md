# Verification boundaries

This extension adds separate numerical, exact-arithmetic and formal-source layers.
They must not be described as interchangeable verification.

## 1. Empirical pipeline

`research_ext/catalog.py` reads recorded JSON outputs. It does not reconstruct
all numerical metrics or validate binary checkpoint contents. It joins training
loss only at the exact logged step, separates model/task conditions and metric
profiles, and deduplicates final/step aliases and copied runs. Original artifacts
are not changed. The legacy `scripts/summarize.py` still exists unchanged; use
`scripts/research.py analyze` for the new audited analysis.

A common loss threshold means the first available **measured** checkpoint at or
below that loss. It does not mean identical loss across networks. Unmeasured
checkpoints and missing outcomes are not interpolated. The coverage audit records
missing runs/profiles and distinguishes finished training from convergence.

Bootstrap resampling units are independent training seeds. Gamma contrasts retain
seed pairing. Conditional hinge intervals and model winner frequencies are
exploratory; neither establishes a phase transition. Across-gamma mean residuals
are not independent observations, so AICc is labeled descriptive. The script
also reports leave-one-gamma-out prediction errors. Seven gamma levels and two
complete seeds are the minimum for this descriptive model fitting, not a claim
of sufficient power. A numerical residual floor prevents roundoff favoring
needlessly complicated fits for constant controls.

Checkpoint threshold events are optional, user-defined and recorded in the
analysis configuration. The default has **no event thresholds**. Events denote
observed or confirmed episodes, not the first event on an unobserved continuous
trajectory. Missing measurements break consecutive confirmation. Unconfirmed
hits do not get asserted event order. Event ordering never establishes causation.

## 2. Noise robustness

`noise-checkpoint` fits a ridge decoder to independent clean training samples,
then tests it with paired Gaussian noise realizations. Center and RMS are fitted
on training representations only. Noise is scaled so its expected RMS norm is
`epsilon * training representation RMS`. This tests a particular decoder's
fragility. Low performance does not prove that no better decoder exists.
Noise repetitions are not additional independent training seeds.

The direct checkpoint command currently supports the upstream torus MLP, not
CNNs, CIFAR models or nonperiodic cylinder coordinates. When relevance is nonzero,
the selected factor need not be truly task-irrelevant; relevance is recorded.

## 3. Exact rational Python certificates

`exact.py` propagates straight polygon edges through small affine/ReLU networks,
subdividing at every rational activation breakpoint. Pairwise image intersections
include interior crossings, collinear overlaps and degenerate segments. No
floating-point tolerance merges nearby points. Image edges are subdivided and
connected-component/cycle counts computed. Exact collisions are checked against
the original input points and a separate exact forward evaluation.

The domain is the **union of the specified straight input edges**. The certificate
is not for the torus between samples, nor for arbitrary points inside the
polygon. A polygon may approximate a smooth circle but the approximation is not
certified here. Input vertices need not be a simple polygon; injectivity is about
distinct points of the geometric subset, not multiple parametrizations of the
same input point.

Box certificates are sufficient tests. A network is propagated to one affine
map only when all activation signs can be fixed on the entire supplied box.
An exact left inverse `B A = I` certifies injectivity of that affine map. A box
crossing an activation boundary, or a rank-deficient ambient matrix, produces an
**inconclusive** result. It is not automatically labeled noninjective. Rank on
the full ambient input is not the same as rank restricted to torus tangents.

Checkpoint export uses restricted `torch.load(weights_only=True)` and hashes the
source checkpoint. It supports only the audited consecutive Linear/ReLU MLP
architecture. It explicitly exports raw hidden representations or the raw head,
not centered/scaled logits. IEEE weights become their exact stored rational
values. The resulting theorem target is ideal arithmetic at those values, not
rounding/error behavior during floating-point inference.

The replay checker recomputes a certificate using the Python implementation.
Exact arithmetic removes roundoff, **not the possibility of programming bugs**.
Tests include known controls, close disjoint segments, high-dimensional crossings,
tampered certificates, constant maps and finite-network input collisions. This
is not an independently proved Python-to-Lean soundness bridge.

## 4. Lean

The delivered Lean source has no proof placeholders or custom axioms. It contains
logical injectivity/recovery lemmas, explicit control collisions, and an executable
finite-graph checker. Generated graph examples use `by decide`, not a compiler-
trusted native decision tactic. `check-formal` must successfully build the pinned
project and audit all 14 named theorem dependencies before it reports success.
Only the three ordinary logical axioms `propext`, `Classical.choice`, `Quot.sound`
are permitted. Source scans are additional checks, not substitutes for building.

**Lean was not available in the delivery environment. Compilation and kernel
verification were not executed.** The bundled status says `lean_verified: false`.
Syntax/library compatibility may still require repair on the pinned compiler.

Even after a successful build, the graph theorem checks the supplied abstract
graph. It does not prove that Python extracted the correct image graph from a
neural network. It does not prove the general correspondence between cycle rank
and singular homology, or global injectivity of a trained smooth-torus map.
Those are separate remaining formalization tasks.

## 5. Controls versus discoveries

Coordinatewise ReLU on the boundary of `[-1,1]^2` illustrates the distinction:
its image still has one graph cycle, but `(-1,1)` and `(0,1)` have the same image.
This exact analytic control shows why unchanged cycle counts do not certify
injectivity. It is not presented as a novel empirical discovery.

## Primary methodology references

- Lean axiom/dependency auditing: https://lean-lang.org/doc/reference/latest/Axioms/
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- Paired bootstrap semantics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Repository protocol at the audited commit: https://github.com/uskutsav-cpu/feature-topology/blob/d6c3f34b41feab45d7d963228c5c3e327098203b/docs/PROTOCOL.md
