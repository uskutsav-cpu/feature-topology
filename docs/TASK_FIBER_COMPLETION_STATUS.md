# Task-preserving relevance-fiber completion

The hosted campaign 39535434835 at source
7df70ce8422964953cfedebe18d44f916a6bfe40 finished successfully on all eight
shards. The unchanged calculations cover all 210 specified models: six
relevance levels, seven gamma values and five paired training seeds. Each model
has a separate-process numerical replay on the same Linux runner. Collector
workflow 39535923452 at source b1050881dbecb9d801c94528717f428c7ca8f857
passed the strict mathematical/metadata collection and every execution receipt.

The complete collection is `results/task_preserving_fiber_completion_hosted_v1`:
210 evaluations, 210 replay receipts, 210 execution receipts, eight shard
manifests, two stage manifests, collection manifest and execution audit. The
downloaded GitHub artifact 12317841674 matched its authenticated ZIP SHA-256
`ef64024b187bd73ffd7365201393d180bcc4892b4fa3a2b958feeec96cb4ce7c`.
All 638 per-file hashes in the execution audit matched after download.

All models reached the first-stored-loss-at-or-below-0.1 selection rule. No final
checkpoint replaced an unreached risk. The paired families yield 420 reported
views and 385 distinct calculations: the 35 lambda-zero controls reuse exactly
the same calculation as their task-preserving counterpart. These are five
training seeds, not 420 independent replicates. All original Mac outputs remain
separate and are not claimed cross-platform replayed.

## Descriptive component findings before the full-study freeze

The fixed-u family preserves u = theta + lambda sin(phi), with the mixed
derivative J_phi - lambda cos(phi) J_theta and explicit unit-tangent scaling.
The fixed-theta original-factor family remains a separate control. No exact
floating equality was found on either declared grid. This finite observation
does not certify real-valued injectivity or absence of continuum collisions.

The table gives means across the five original training seeds on the
task-preserving family. It is a component inspection, not the final locked-runtime
statistics or evidence that the full study is ready.

| Lambda | Gamma | Representation RMS | Raw unit-tangent minimum | Raw grid phi-arc minimum | Grid phi-arc minimum / RMS |
|---|---|---|---|---|---|
| 0 | 0.125 | 3.26195 | 2.39919 | 1.11803 | 0.343301 |
| 0 | 128 | 240.59021 | 2.40947 | 1.40100 | 0.005899 |
| 1 | 0.125 | 3.35401 | 2.47645 | 1.23515 | 0.368041 |
| 1 | 128 | 225.35734 | 3.33207 | 1.97258 | 0.008800 |

Thus the endpoint means show pronounced normalized contraction while the raw
sampled minima remain positive. A normalized decrease cannot be equated with
identification or information destruction. Every intermediate gamma and
relevance level is retained in the collection and will appear in the final
tables. The original fixed-theta measurements are not overwritten or relabeled.

The fixed-budget adversarial search is retained in full, including cases where
it does not improve the grid minimum. Neither its best candidate nor the grid
minimum is a continuum lower bound; both are upper bounds on an infimum over
their larger domain. No search budget was expanded after inspecting outcomes.

Actual selected training losses range from about 0.03496 to 0.09968. The
selection rule supplies a common threshold, not identical achieved risk. Final
summaries must retain these losses, all raw scales and both fiber families.
Pointwise seed-bootstrap intervals and paired contrasts remain part of the
freeze-gated final analysis. This completion does not change the negative
five-seed predictive comparison, establish causal OOD effects or guarantee
high novelty.

After the SSD was remounted on 2026-10-21, the physical-source component audit
also passed: the 210 hosted records and replay/execution receipts match the
actual selected checkpoints, configs, summaries and original probe archives.
All 18 protected mathematical source files remain unchanged. This check does
not claim a fresh Mac numerical replay or cross-platform numerical agreement.

The original and combined full physical-input freezes, remaining CIFAR work
and final analyses remain required. Manuscript preparation is still deferred.
