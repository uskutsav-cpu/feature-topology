# Additional diagnostic comparison: observed result

The recorded strengthening comparison does **not** establish incremental
predictive value for the fiber diagnostics. The project must not claim superior
OOD prediction, a validated diagnostic advantage, or high novelty on that basis.

This comparison covers all 140 frozen cells: four training conditions, seven
gamma values, and five training seeds. Every original metric, nuisance evaluation,
and VCI baseline is joined to the same first saved checkpoint at training loss
at most 0.1. There are no missing predictors or outcomes. Where numbered and
final checkpoint files have different serialized hashes, a separate audit checks
exact equality of every inference-state tensor, including the frozen reference
branch. Step numbers alone are never treated as sufficient identity evidence.

## Held-out prediction

Each outer test fold holds out an entire training seed. Its gamma and condition
cells are never used for model fitting, standardization, or regularization
selection. Inner validation also holds out complete seeds. The comparison was
specified in `configs/strengthening_diagnostic_protocol_v1.json` before inspecting
the new shift outcomes; earlier study results were already known.

| Shift accuracy minus IID accuracy | Design MSE | Conventional MSE | Fiber-added MSE |
|---|---:|---:|---:|
| Concentrated | 0.000478069 | 0.000515680 | 0.000545522 |
| Spurious | 0.001094904 | 0.001549166 | 0.002691410 |
| Unseen | 0.000440053 | 0.000453648 | 0.000455355 |

The design model includes condition-by-gamma indicators and actual achieved
training loss. The conventional model adds geometry, decoding, persistence, VCI,
and a within/between covariance trace ratio. The fiber model adds all five
declared fiber diagnostics. All three models use the same nested ridge procedure.

Conventional-minus-fiber MSE is negative for every outcome, meaning that adding
fiber measurements raises the mean held-out error. The family-adjusted bootstrap
intervals are:

- Concentrated: -0.000072917 to 0.000011015.
- Spurious: -0.002866779 to -0.000100621.
- Unseen: -0.000029623 to 0.000021603.

The spurious comparison is worse in all five held-out seeds. The bootstrap
interval is also negative, but five independent seeds give limited precision;
these intervals are not a substitute for a larger independent replication.
The design-only model has the smallest mean error for all three outcomes.
The result is a boundary on this proposed predictive claim, not evidence that
fiber diagnostics have no possible use in any other setting.

## What the evidence can support

The primary study's raw sampled fiber separation and clean nuisance decoding
remain strong while normalized geometry and persistent homology contract. This
supports separating metric change, nuisance decodability, and actual point
identification. The broader distinction has close prior work, as recorded in
the novelty audit. The current comparison does not supply a new predictive
advantage to distinguish the project from that literature.

Spurious training produces a large predictive failure under the frozen spurious
test intervention even though the task coordinates and true label rule are
unchanged. This is a useful controlled result. It does not imply that fiber
collapse caused the failure or that preserving fibers guarantees robust prediction.

The exact small-network certificates retain their rational polygon domains.
The formal lemmas retain their stated logical/graph scope. Neither is a proof of
injectivity of the trained full torus networks.

## Completion status

This additional analysis is source bound and complete. The original-study full
freeze and final synthesis remain required. The seven CIFAR continuations are
still running, and full physical inputs are being verified in an isolated copy
of the recovered external-SSD workspace. Manuscript work remains deferred.

Machine-readable inputs and all seed-level scores are retained under
`results/additional_diagnostic_comparison`, `results/additional_variability`,
`results/ood_evaluation`, and `results/checkpoint_aliases`.
