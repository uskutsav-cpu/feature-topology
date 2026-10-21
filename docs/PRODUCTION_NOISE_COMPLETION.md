# Completing the planned noisy-decoding component

The immutable v2 analysis plan defines D_F to include clean and relative-noise
decoding. The original production records contain clean probes. The earlier
39-run continuation includes noise diagnostics at a reduced profile and its own
final checkpoints; it does not replace the requested production risk selection.
This completion fills that specific scope gap while preserving the old results,
training configurations, production profiles, and v2 plan.

The recorded specification is `configs/production_noise_completion_v1.json`.
It includes all 130 main models and all 210 relevance-view models, with their 35
shared lambda-zero models evaluated once: 305 unique checkpoints or explicitly
unreached-risk records. Use the same first stored loss <=0.1 rule and each
cohort's source-bound frozen 5000/2000 probe split.

The existing `src.metrics.fibers.noise_decodability` method uses relative levels
0, 0.001, 0.01 and 0.1. It adds noise to both training and test features, fits
scaling on the noisy training features, and refits a ridge decoder at each level.
The result measures this level-adapted decoder. It is different from the optional
`research_ext.noise` clean-trained-decoder fragility command. Noise magnitude is
relative to centered clean training RMS; directions are paired across levels.

All production noise outcomes are new evaluations of unchanged saved weights.
Earlier continuation results and the main clean/predictive outcomes were already
known, so this is a transparent completion of a missing planned component rather
than a new original preregistration. The previously negative predictive comparison
is unchanged. At positive lambda, phi is a partly task-relevant coordinate; these
measurements are not mislabeled as task-preserving nuisance fibers.

The same-host replay, source binding and seed-level summaries are required before
the additional research freeze can pass. Ten main seeds and five relevance seeds
are the statistical units. Noise levels, probe samples and resampling repetitions
do not create more independent training seeds. Decoder failure does not prove
that all information is absent.
