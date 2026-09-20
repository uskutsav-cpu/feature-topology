# Nuisance-shift protocol

The predictive-consequence study crosses the seven representative feature-learning
strengths, five paired seeds, and four training environments in
`configs/sweeps/ood.json`.  Every non-IID training condition receives an
optimization-only learning-rate calibration.  The IID cells reuse compatible
primary runs; no topology or test outcome enters calibration.

The environments change only the nuisance angle distribution for the pure
\(Y=f(\theta)\) task:

- `iid`: both angles are uniform.
- `concentrated`: training \(\phi\sim\mathrm{vonMises}(0,8)\); shifted test
  \(\phi\sim\mathrm{vonMises}(\pi,8)\).
- `spurious`: training \(\phi\) is concentrated around the angular sector
  associated with the theta class; test reverses that association by \(\pi\).
- `unseen`: training observes \(\phi\in[0,3\pi/2)\), while test uses the disjoint
  arc \([3\pi/2,2\pi)\).

Every checkpoint is selected by the frozen matched-risk rule: the first stored
step at or below training loss 0.1. `scripts/evaluate_nuisance_shift.py` evaluates
that one checkpoint on all four test environments with fixed independent seeds
and 5,000 examples per environment.  It binds every row to the checkpoint's
SHA-256.  Accuracy/loss support predictive claims only; they are not evidence of
injectivity or quotient formation.

The study is incomplete until the live freeze gate verifies all 140 training
conditions, their production representation metrics, and all 140 four-environment
evaluations.  Failed or target-unreached runs remain recorded outcomes.
