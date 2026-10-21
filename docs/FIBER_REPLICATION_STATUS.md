# Independent-seed replication status

The protocol was committed as fa9371ecd22f4d58845bd102b9ce3d4b3cb176b1 before
constructing any new-seed witnesses. Screening implementation
0e5a63b6a1995a047b7cd0e1ad2f6710aa6bce3a then evaluated every one of the 36
planned models. Its complete source-bound collection is retained in
`results/fiber_replication_screens_v1`.

| Gamma | Planned replication seeds | Fixed-readout screens passed | Screens failed |
|---|---:|---:|---:|
| 0.03125 | 9 | 1 | 8 |
| 0.5 | 9 | 1 | 8 |
| 1 | 9 | 3 | 6 |
| 8 | 9 | 2 | 7 |
| Total | 36 model configurations | 7 | 29 |

The nine independent seed identifiers are shared across gamma configurations.
The 36 configurations are not 36 distinct independent training seeds. All 29
failed readouts remain present. These failures show that this particular fixed
witness construction is not universally eligible in the new cohort; they do not
establish noninjectivity, nuisance deletion, or incorrect classifier decisions.

The eligible models are gamma 0.03125/seed 1; gamma 0.5/seed 3; gamma 1/seeds
2, 3, 5; and gamma 8/seeds 8, 9. Each has four prospective retention attempts.
Every one of the 144 planned domains receives a complete-classifier decision
attempt, including the models whose retention readouts failed the screen.

Workflow 39512823384 completed successfully at source
1531aa945dfa85eaf23d16d0d86c6a37a6a3fc10. All 144 domain records and their
separate-process replay receipts are collected in
`results/fiber_replication_verification_v1`. A component audit against all 36
physical saved checkpoints and the original embedding passed. This is not the
full-study freeze.

| Gamma | Planned domains | Retention and correct-class certificates | Models with all four joint certificates | Correct-class certificates |
|---|---:|---:|---:|---:|
| 0.03125 | 36 | 3 | 0 | 36 |
| 0.5 | 36 | 4 | 1 | 36 |
| 1 | 36 | 12 | 3 | 36 |
| 8 | 36 | 8 | 2 | 36 |
| Total | 144 | 27 | 6 configurations | 144 |

One eligible retention attempt, gamma 0.03125/seed 1/theta 5pi/4, rejected
the fixed witness. It is retained alongside all 116 domains with failed
readout screens. Neither outcome proves noninjectivity. Six configurations
with all four joint certificates are not six independent seeds: seed 3 is
shared by the gamma 0.5 and gamma 1 configurations. All 144 complete-classifier
checks certify the correct class; none produces a certified counterexample.

The discovery cohort's four gammas were selected using seed-zero eligibility,
so its successes cannot be compared with the replication proportion as if both
cohorts were unselected random samples. The accepted bounds apply only to the
specified real-valued saved-weight circles. They do not establish all-theta or
full-torus behavior, independent-verifier agreement, or causal OOD robustness.
