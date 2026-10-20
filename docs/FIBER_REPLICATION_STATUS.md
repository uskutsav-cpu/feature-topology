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

Verification and separate-process replay remain pending. No joint replication
success is inferred from a passed numerical screen. The discovery cohort's four
gammas were selected using seed-zero eligibility, so its four successes cannot
be compared with the new screen proportion as if both cohorts were unselected
random samples. The original seed-zero findings remain valid at their stated
scope; their generality still needs the completed replication outcomes.
