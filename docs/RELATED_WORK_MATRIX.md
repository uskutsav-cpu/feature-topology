# Novelty audit and related-work matrix

Audit date: 2026-09-19.  Claims below are scoped to the cited versions and will
be rechecked before submission.

| Closest work | What it establishes | What it does not establish | This project's incremental claim |
|---|---|---|---|
| [Atanasov et al., *The Optimization Landscape of SGD Across the Feature Learning Strength*](https://arxiv.org/abs/2410.04642) | Maps lazy, rich, and ultra-rich optimization regimes; shows the optimal learning rate scales approximately as \(\gamma^2\) in the lazy regime and \(\gamma^{2/L}\) in the rich regime. | Does not distinguish nuisance deformation, near-singularity, and exact fiber identification. | Calibrate learning rate independently for every \(\gamma\), compare at matched risk, and diagnose fiber survival rather than attributing topology to \(\gamma\) alone. |
| [Yeom et al., *Over-Alignment vs Over-Fitting*](https://arxiv.org/abs/2602.00827) | Gives theory and experiments for an intermediate optimal feature-learning strength; identifies ultra-rich over-alignment as a generalization failure. | Does not identify which nuisance equivalence classes are preserved or collapsed, nor connect the optimum to a quotient regime. | Test whether the generalization optimum coincides with preservation, compression, sampled collision, or exact restricted quotienting under nuisance shift. |
| [Beshkov & Einevoll, *A Quotient Homology Theory of Representation in Neural Networks*](https://openreview.net/forum?id=RluspxztzS) | Defines rank/overlap decompositions for piecewise-linear networks and relates representation homology to a quotient under stated convexity conditions; avoids reliance on an external metric. | Does not organize feature-learning-strength sweeps by task-defined nuisance fibers or matched-risk \((\gamma,\lambda)\) interventions. | Use task quotients as the target object and a fiber-survival hierarchy as a front end to exact restricted overlap/graph certificates.  We do not claim to replace or rediscover quotient homology. |
| [Naitzat, Zhitnikov & Lim, *Topology of Deep Neural Networks*](https://jmlr.org/papers/v21/20-345.html) | Empirically tracks layerwise persistent-homology simplification and contrasts ReLU with smooth activations. | PH simplification does not certify which input points were identified and can change under severe anisotropic deformation. | Ground-truth controls show explicitly when PH, global collisions, and nuisance information disagree; PH becomes supporting rather than decisive evidence. |
| [Montanari & Wang, *Phase Transitions for Feature Learning in Neural Networks*](https://arxiv.org/abs/2602.01434) | Derives a high-dimensional sample-ratio threshold for two-layer feature learning in a multi-index model and associates it with Hessian spectral change. | Does not study a \(\gamma\)-driven nuisance-fiber quotient transition in finite networks. | Reserve "phase transition" for a finite-width scaling result with shrinking transition width and controlled pseudo-critical drift; otherwise report a crossover. |
| [Kornblith et al., *Similarity of Neural Network Representations Revisited*](https://proceedings.mlr.press/v97/kornblith19a.html) | Establishes CKA as a useful representation-similarity measure with invariances suited to comparing learned features. | Similarity is not injectivity, recoverability, or equality of quotient relations. | Use \(1-\mathrm{CKA}\) only as the geometric order parameter \(G\), alongside \(L,M,Q,D_F,R\). |
| [Shen, *A Differential Topological View of Challenges in Learning with Feedforward Neural Networks*](https://arxiv.org/abs/1811.10304) | Proposes quotient-topological language for task nuisance factors in deep representations. | Does not provide the present matched-risk FLS experiment or the local/global/collision diagnostic separation. | Operationalize the task quotient with computable fiber-specific margins, exact controls, and predictive consequences. |

## Claim-by-claim novelty contract

- **Fiber-survival hierarchy.** Closest prior work: Beshkov's quotient/overlap
  decomposition plus differential diagnostics used broadly in representation
  geometry.  Added value: an explicitly task-fibered ordering of local margin,
  global lower-Lipschitz separation, collision, and quotient equality, with known
  counterexamples at every diagnostic boundary.
- **Feature-learning regime map.** Closest prior work: Atanasov's FLS/learning-rate
  landscape.  Added value: matched-risk, independently calibrated
  \((\gamma,\lambda)\) interventions where \(\lambda\) changes whether nuisance
  identification is statistically permissible.
- **Generalization consequence.** Closest prior work: Yeom's optimum-FLS
  generalization theory.  Added value: test whether over-alignment corresponds to
  a measured fiber-survival regime under IID, nuisance shift, spurious
  correlation, and unseen nuisance values.
- **Topology claim.** Closest prior work: Naitzat's PH simplification and Beshkov's
  quotient homology.  Added value: exact constructed controls and restricted
  rational certificates that prevent anisotropy or PH from being mislabeled as
  quotient formation.

These are target contributions, not presumed discoveries.  Null results narrow
the claims rather than triggering post-hoc retuning.
