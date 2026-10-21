# Production noisy-decoding completion

The protocol was recorded at 50b9f6a before new production noise outcomes.
Worker source 91b23d9d84e7fee8008534762b45c16ccf45936f evaluated all 305
specified saved models and repeated each evaluation in a separate invocation
on the same host. All 305 reached the first-stored-loss-at-or-below-0.1
selection rule; no final checkpoint was substituted for an unreached risk.
The full collection is `results/production_noise_completion_v1`.

A physical component audit checked every selected checkpoint, its config and
training summary, and all six original probe archives. It also rechecked all
worker/protocol source bytes against the recorded source commit. The 130-model
main view and 210-model relevance view share 35 models evaluated once. Canonical
primary paths include 15 existing pilot aliases; their original member hashes
and selection rules remain checked.

The unchanged legacy method adds relative noise to both probe-training and
held-out representations and refits the ridge probe at each of the four
levels. It is not a clean-trained probe tested only under corruption. Final
tables preserve every model, selected loss, representation scale and noise
level. Main intervals use ten training seeds, relevance intervals five paired
seeds, with 2000 pointwise percentile bootstrap resamples. Noise levels and
probe draws do not increase the number of training replicates.

The first generation invocation stopped after 130 models because the local
Python certificate bundle was unavailable for the next input download. The
failed operational logs are retained in the task workspace. Setting
`SSL_CERT_FILE` to the installed pip-vendored certifi bundle restored normal
certificate validation; unchanged code and parameters then resumed. No
scientific result or verifier rule was changed to resolve transport.

Strict collection and the component audit pass. The full original/combined
freeze, final analysis and figure checks remain required. No failed noisy
decoder proves information destruction, and at positive lambda the decoded
phi coordinate is partly task relevant. These results do not retroactively
alter the retained negative predictive comparison.
