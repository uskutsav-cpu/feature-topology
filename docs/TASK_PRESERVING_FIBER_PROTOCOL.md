# Task-preserving relevance-fiber completion

`configs/task_preserving_fiber_completion_v1.json` records the design before any
new saved-model fixed-u or adversarial-search outcome. This is a post-hoc
semantic completion of the original v2 scope. Earlier original-coordinate
metrics, noise results and the seed-zero/replication certificates were known.
It is not an original preregistration or an independent discovery cohort.

For the periodic task, u = theta + lambda sin(phi). The task-preserving circle
therefore holds u fixed with theta(phi) = u - lambda sin(phi). Its representation
derivative is J_phi - lambda cos(phi) J_theta. The original fixed-theta phi
sweeps remain a separate control; at positive lambda they can change the label.
Neither family equals the complete four-class target-label fiber.

Use all 210 original relevance-view models, first stored loss <= 0.1, last
current hidden layer, original embedding, no retraining. Compare the two
families on the recorded 100 by 100 half-offset grid. Report raw and normalized
local derivatives, exhaustive finite-grid within-circle separation and
collision counts, centered full-classifier grid accuracy, and the recorded
16-start, 64-update adversarial pair search. Normalize unit tangents and
latent distances explicitly; never infer the mixed derivative from stored
scalar factor norms. Lambda-zero families are exactly the same computation.

Search minimizes a separated-pair ratio on an exact same-base parameterization.
Its best candidate is an upper bound on the true infimum. Failure to find a
collision is not proof of injectivity. Grid points and search starts do not
increase the number of training replicates. Summaries use five paired training
seeds and 2000 pointwise bootstrap resamples, preserving all requested outcomes.

Repeat each numerical evaluation separately on the same fixed runtime and
retain source/checkpoint bindings and replay receipts. This is numerical
reproducibility of the same implementation, not a new real-arithmetic
certificate, an independent verifier, or a theorem in Lean. Do not modify the
original v2 plan, original production metrics, original comparison or prior
certificate procedures to incorporate these results.
