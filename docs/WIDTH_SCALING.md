# Finite-width terminology gate

`configs/width_scaling_gate_v1.json` is the prospective, machine-readable rule
for deciding whether the empirical result may be called a phase transition. The
primary order parameter is the dense within-fiber normalized separation at the
last hidden layer and the comparison is made at the first stored checkpoint at
or below training loss 0.1.

For each of five widths, the gate compares a monotone four-parameter logistic
curve in \(\log_2\gamma\) against a quadratic smooth null using AICc. Its
transition width is the fitted 10%--90% interval. Seed trajectories are paired
across gamma and width; the bootstrap resamples seed identities jointly.

The stronger phrase is allowed only when every frozen clause passes: the
transition model is decisively preferred at every width, transition widths
shrink systematically and materially, at least 90% of paired bootstraps support
that shrinkage, the two largest-width pseudo-critical points differ by at most
one log2-gamma unit with at least 90% bootstrap support, and all fitted centers
are interior. Any failed clause deterministically yields `crossover`.

This gate controls terminology, not study completion. A complete negative result
is valid and freezeable. Even a passed gate applies only to the stated empirical
order parameter; it neither proves a thermodynamic limit nor certifies exact
quotient formation.

The runner selects the production metric profile by default, discovers nested
ablation condition directories, ignores gamma cells outside the frozen design,
and fails closed when any of the 175 requested width/gamma/seed cells is absent.
