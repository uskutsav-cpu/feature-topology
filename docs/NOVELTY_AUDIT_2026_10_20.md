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
| [Shen, 2018](https://arxiv.org/html/1811.10304v1), section 5 | Uses quotient topology to describe task-specific nuisance invariance in representation learning. | Task-conditioned nuisance quotients are not a new conceptual framework here. |
| [Lin and Kriegeskorte, PNAS 2024](https://arxiv.org/abs/2309.11028v3) | Separates geometry-sensitive and topology-sensitive descriptions of representations, with simulation and fMRI comparisons. | A broad proposal to study both representation geometry and topology is already established. |
| [Dahl, Entropy 2026](https://www.mdpi.com/1099-4300/28/7/796) | Compares category compression with target and nuisance information in synthetic settings and pretrained visual representations. | The general distinction between compression and target preservation is prior art; category-map entropy is a different object from exact fiber injectivity. |
| [Zhan et al., ICLR 2026](https://arxiv.org/html/2604.06701v1), section 3.1 | Characterizes injectivity on compact domains using separation at every positive distance threshold and uses a sampled separation regularizer for autoencoders. | Pairwise separation and the distinction between local differential constraints, sampled noncollision, and global injectivity are not new. The present audit concerns a posteriori certificates on restricted domains of unchanged classifier representations, not a new injectivity training regularizer. |
| [Puthawala et al., ICML 2022](https://proceedings.mlr.press/v162/puthawala22a.html) | Studies manifold and density approximation with injective flows, including embedding-dependent topological obstructions. | Manifold embeddings and topology in injective network architectures are established; these are different from auditing ordinary trained representations. |
| [Shaw et al., TAG-DS 2026](https://proceedings.mlr.press/v334/shaw26a.html) | Studies layer-wise persistent homology and a compression/reconstruction tradeoff in predictive-coding networks, with matched MLP comparisons. | Layer-wise PH and model-capacity comparisons are existing empirical approaches; the restricted continuum evidence must supply the more specific distinction here. |

Two additional primary sources directly constrain the paired classifier claim:

- [Jacobsen, Smeulders and Oyallon, ICLR 2018](https://arxiv.org/abs/1802.07088)
  retains invertibility up to the final class projection in i-RevNet and
  demonstrates ImageNet classification alongside information retention and
  geometric contraction. Correct classification without hidden information
  deletion is already established. These paired certificates cannot be presented
  as the first demonstration of that general possibility.
- [Rath and Condurache, AISTATS 2023](https://proceedings.mlr.press/v206/rath23a.html)
  enforces guaranteed invariances using equivariant representations followed by
  invariant operations. Distinguishing changing intermediate representations
  from invariant outputs is established; this ordinary MLP audit introduces no
  new invariant architecture.

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

## Evidence checklist and present outcome

All 845 synthetic profiles and the 140-cell predictive/baseline comparison are
now recovered and checked. Seven unchanged CIFAR continuations and the final
physical-input freeze remain pending. Raw versus normalized quantities and
amplification controls remain necessary for interpreting compression.

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

## Continuous-fiber strengthening: contribution still to assess

The remaining empirical candidate is a source-bound comparison of sampled
geometric contraction with rigorous retention on explicitly specified continuous
nuisance circles in the same saved models. The full-torus attempt was
inconclusive. Narrower protocols retain every attempt and declare their scope;
they do not replace the original study or validate its failed predictive claim.
All sixteen V4 cases now pass source-bound generation and separate-process
hosted replay. They cover four seed-zero models at four fixed task coordinates,
so they cannot establish a population-wide training-seed result. All sixteen
additional local replays also pass. The final combined physical-input freeze
and complete research synthesis remain required.

The separately recorded complete-classifier check now yields sixteen positive
correct-class certificates and sixteen successful separate-process Linux
replays on exactly the same retained domains. This establishes a paired
retention/decision property for the four saved ordinary MLP models. It strengthens
the particular a posteriori empirical audit; the general coexistence of
information-preserving hidden representations and successful classification was
already established by i-RevNet. The additional Mac replay uses explicitly weaker
derived margins after a small enclosure mismatch; the original failure and
the limits of same-implementation replay remain part of the evidence.

The separation inequality is an elementary perturbation argument. Global ReLU
injectivity and inverse stability are established, as are the DeepZ affine
relaxation and activation-vertex bounding principles used by the implementation.
The current [TorchLean verification documentation](https://lean-dojo.github.io/TorchLean/blueprint/Verification-and-Certificates/Neural-Network-Verification/)
also describes interval directional derivatives and distinguishes executable
artifact acceptance from semantic soundness theorems. A derivative verifier or
successful Lean compilation alone cannot be presented as the novelty here.

These sources narrow the defensible claim to the particular calibrated empirical
study, its negative diagnostic comparison, and any successfully checked
domain-specific retention evidence. Whether that combination is a sufficiently
substantial advance remains a scientific judgment after final synthesis. The
search does not establish an exhaustive absence of related work, and the project
is not currently labeled highly novel or ready for journal submission.
