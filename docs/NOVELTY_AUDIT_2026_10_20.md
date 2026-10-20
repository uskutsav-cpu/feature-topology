# Research novelty audit, 2026-10-20

The user requests completion and strengthening of the project before manuscript
work. High novelty remains a scientific claim to demonstrate. Engineering
completeness, seed counts, formal compilation, and a large experiment grid do
not establish novelty by themselves.

## Closest results that must be distinguished

| Primary source inspected | Established overlap | Consequence for this project |
|---|---|---|
| [Atanasov et al., ICLR 2025](https://arxiv.org/abs/2410.04642v3) | Output scaling controls feature-learning strength and interacts with the optimal learning rate. | Neither a gamma sweep nor condition-specific calibration is a new contribution. |
| [Yeom et al., ICML 2026](https://arxiv.org/html/2602.00827v2) | Finite-training-risk generalization can favor intermediate FLS; two-layer theory explains over-alignment versus over-fitting. | An intermediate optimum or worse accuracy at extreme FLS is insufficient as the main novelty claim. |
| [Beshkov, TMLR 2026](https://arxiv.org/html/2502.01360v4) | Quotient/overlap computations distinguish identification from metric persistent homology; local rank and region overlap are different noninjectivity mechanisms. | The broad geometry-versus-topology distinction and a small overlap implementation are prior art. |
| [Goldfeld et al., ICML 2019](https://proceedings.mlr.press/v97/goldfeld19a.html) | Noisy-representation information and deterministic geometric clustering are different quantities. | Noise-sensitive decoding is not by itself a new information-destruction result. |
| [Puthawala et al., JMLR 2022](https://www.jmlr.org/papers/v23/21-0282.html) | Global injectivity of ReLU networks and inverse stability already have substantial theory. | Elementary lower-Lipschitz injectivity arguments are supporting lemmas, not a standalone theoretical advance. |
| [Xu and Liu, ICML 2023](https://proceedings.mlr.press/v202/xu23k.html) | Variability Collapse Index is invariant to invertible linear transformations and studied as an indicator of transferability. | A comparison restricted to CKA and effective rank would omit a particularly relevant baseline. |
| [Tirer et al., ICML 2023](https://proceedings.mlr.press/v202/tirer23a.html) | Near-collapse theory studies nonzero residual within-class variability. | Small within-class spread does not establish complete nuisance deletion. |

The statements above summarize the inspected versions. They are not assertions
that an exhaustive literature search proves the absence of every related result.

## Candidate contribution requiring evidence

The most defensible research question is whether a **task-conditioned map of
fiber survival**, measured at calibrated common-risk checkpoints, adds useful
information about nuisance-shift behavior beyond conventional geometry,
decodability, and variability-collapse measures. The relevance interventions,
manifold controls, width/depth controls, and restricted exact certificates can
make that question more specific and testable than a generic compression claim.

The existing width result is a negative finite-width finding: its frozen gate
returns crossover. It must not be relabeled a phase transition to strengthen
the perceived novelty.

## Evidence still needed

1. Reconcile all archived numerical inputs and finish the seven CIFAR cells.
2. Complete source-bound predictive evaluation of all 140 frozen nuisance-shift
   cells using the existing checkpoint and data rules.
3. Compare the fiber diagnostics against relevant conventional and VCI baselines
   with held-out training seeds, keeping fitting and evaluation separate.
4. Explain any apparent compression with raw/normalized quantities and the
   existing amplification controls before asserting actual identification.
5. Report whether the incremental diagnostic claim succeeds, fails, or remains
   uncertain. New experiments must address a concrete unresolved explanation.

The additional comparison is specified in
`configs/strengthening_diagnostic_protocol_v1.json`. It is prospective for that
new comparison, but it is not a preregistration of the original project: pilot
results, the continuation study, and the finite-width result were already known.

## Completed additional comparison

All 140 cells have been evaluated, including the VCI baseline and an exact
checkpoint-alias tensor audit. The recorded nested held-out-seed comparison
finds no incremental predictive benefit from adding the fiber diagnostics on
any of its three outcomes. Spurious-shift prediction is worse in all five
held-out seeds. The design-only baseline has the lowest mean error for every
outcome. See `docs/ADDITIONAL_DIAGNOSTIC_FINDINGS.md` and the retained numerical
results. This finding is preserved; it does not establish the proposed diagnostic
novelty, and the protocol is not revised to force a positive result.
