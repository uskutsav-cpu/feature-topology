# Tighter activation bounds: engineering extension

The first full-torus verification attempt is retained under
`results/continuous_fiber_verification_v1`. All four eligible witnesses returned
inconclusive at the depth limit. No trained continuum certificate resulted.
Success of the hosted workflow means the attempt executed successfully, not
that an injectivity statement was established.

The elementary interval enclosure lost too much dependency information. For
example, on the lower-left depth-16 parameter box of the gamma=0.03125 witness,
its derivative-error upper bound was about 164.6, despite a numerical whole-grid
maximum near 0.598. This is an enclosure problem, not evidence of a collision.

`continuous_fiber_affine_verifier.py` keeps four input-coordinate symbols through
affine layers and separates the remaining error into intervals. It uses an
established centered ReLU relaxation. If l<0<u, let

    alpha = u/(u-l), beta = -alpha*l/2.

For every y in [l,u], ReLU(y) lies in alpha*y+beta+[-beta,beta]. This follows by
checking the two affine pieces on [l,0] and [0,u]. Positive/negative stable
neurons retain the exact affine transform or zero. Every arithmetic operation,
including relaxation parameters and remainder propagation, uses Arb balls.
Interval masks still enclose the actual ReLU derivatives; the final separation
criterion is unchanged.

This relaxation is prior art, not a claimed theoretical contribution:
[DeepZ, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html).
Related affine/interval abstraction and refinement are also established in
[DeepPoly, POPL 2019](https://doi.org/10.1145/3290354).

On that same diagnostic box, the affine-input/remainder implementation reduces
the bound to about 0.162. This demonstrates a useful local improvement only.
It does not establish complete torus coverage or a trained continuum certificate.
The new implementation passes complete-circle replay, a folded-ReLU negative
control, and an affine cancellation control.

A further verification attempt needs a separately recorded protocol, complete
domain coverage, source-bound results, and replay. If whole-torus verification
is computationally impractical, certifying entire smooth circles at explicitly
fixed task coordinates would be a narrower result. Such a result must not be
described as covering every task coordinate or as proving full-torus injectivity.
