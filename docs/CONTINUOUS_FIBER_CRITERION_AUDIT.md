# Scope and mathematical checks for the circle criterion

This document supplements the earlier, source-bound
`CONTINUOUS_FIBER_CRITERION.md`; that file is preserved because executed workers
bind its bytes. These are elementary supporting arguments, not claims of new
injectivity theory.

## The theorem actually used

Fix one theta. Let h be the resulting periodic Lipschitz curve, let A be its
fixed two-row readout, and let s(phi) = (cos(phi), sin(phi)). If the absolutely
continuous residual r = A h - s satisfies ||r'|| <= epsilon almost everywhere,
where epsilon < 2/pi, then for every pair of angles

    ||h(phi_1)-h(phi_2)|| >= k ||s(phi_1)-s(phi_2)||,
    k = (1-epsilon*pi/2)/||A||_F > 0.

Indeed, integrate r' along the shorter circular arc of length delta <= pi.
Its chord c satisfies delta <= pi*c/2. Thus ||A delta_h|| >= c-epsilon*delta,
and ||A delta_h|| <= ||A||_2 ||delta_h|| <= ||A||_F ||delta_h||. These inequalities
give the displayed bound. An affine readout intercept cancels in differences,
so the fitted intercept need not enter the derivative computation.

Finite ReLU networks restricted to a smooth compact circle are Lipschitz and
therefore absolutely continuous in the angle parameter. A derivative bound
almost everywhere suffices, including when the curve crosses activation
boundaries. Uncertain derivative masks must contain the possible derivatives;
selecting one convenient activation pattern would not be sound.

Arb enclosures for the Frobenius norm yield a conservative positive factor.
V4 rounds epsilon upwards and the resulting factor downwards to exact dyadic
bounds. A replay proves the inequalities against those exact bounds, rather
than relying on approximate decimal display intervals.

## Why the strict threshold cannot be weakened for general curves

Let a = 2/pi, t(phi) = arccos(cos(phi)), and

    h(phi) = (cos(phi) + a*t(phi), sin(phi)),  A = identity.

This is a periodic Lipschitz curve. Its residual has ||r'|| = a almost
everywhere, but h(0) = h(pi) = (1,0). It is therefore not injective at equality.
Moreover, on both smooth halves its speed is at least 1-a > 0, since

    ||h'||^2 >= 1 + a^2 - 2a = (1-a)^2.

Thus a positive local speed almost everywhere does not prevent a global
collision. The example establishes sharpness of the strict residual threshold
over this general class of curves and witnesses. It is not a construction of
that curve as one of the trained ReLU models, nor a new optimality claim about
ReLU verification algorithms.

## What these certificates do and do not identify

The implemented domains fix theta at one of four exact class-sector midpoints
and include every phi on that particular circle. They do not include every
theta. A fixed-theta nuisance orbit is smaller than a target-label fiber: the
four-class target also identifies different theta values within each sector.
Injectivity on the declared nuisance orbit therefore does not prove a theorem
about the complete target-label fiber, a task quotient, or the entire torus.

The conclusion concerns the hidden representation of the real-valued
continuation with saved dyadic coefficients. It does not concern the classifier
head's invariance, correctness under distribution shift, information after
observation noise, or the native floating-point program. The same implementation
replays every leaf in a separate process. It is not an independent implementation
or a Lean soundness proof of the network encoding.

There is one additional qualitative interpretation under the primary study's
independent uniform latent law. A certified positive separation at a single
theta excludes complete exact almost-sure phi erasure by the jointly continuous
hidden representation: continuity extends a positive pair difference to an
open set of angle triples, which has positive product measure. The proof is in
`THEORY.md`. This does not extend the circle injectivity certificate to every
theta or establish unconditional/noisy phi recoverability. Exact finite-valued
class-center collapse also faces a separate connectedness obstruction under a
full-support continuum law; that observation is structural and not a claimed
empirical contribution.
