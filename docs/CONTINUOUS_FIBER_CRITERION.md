# Continuous-circle retention: candidate certificate

This is a supporting sufficient condition and a proposed verification route.
The inequality below is an elementary perturbation argument. Neither theoretical
novelty nor a successful trained-network certificate is asserted here.

Let h(theta,phi) be a representation on the product of two circles. For each fixed
theta, assume the periodic curve phi -> h(theta,phi) is Lipschitz. This includes
the real-valued continuation of a finite ReLU network restricted to the torus.
Let A be a fixed two-row linear readout, s(phi)=(cos(phi),sin(phi)), and
r_theta(phi)=A h(theta,phi)-s(phi). Suppose, for **every** theta, the absolutely
continuous residual obeys ||d r_theta/d phi|| <= epsilon almost everywhere, with
epsilon < 2/pi.

For two nuisance angles with shorter arc distance delta in [0,pi], integration
along that shorter arc gives

    ||r_theta(phi_1)-r_theta(phi_2)|| <= epsilon delta.

Their unit-circle chord has length c=2 sin(delta/2), and delta <= (pi/2)c.
The reverse triangle inequality therefore gives

    ||A[h(theta,phi_1)-h(theta,phi_2)]||
      >= c-epsilon delta >= (1-epsilon pi/2)c.

Since ||A v|| <= ||A||_2 ||v||, each nuisance circle has the lower separation
bound (1-epsilon pi/2)c/||A||_2. It follows that each individual nuisance circle
is mapped injectively. The positive criterion also ensures A is nonzero.

This condition does not compare points on different theta fibers. It does not
certify injectivity of the full torus, output invariance to nuisance, predictive
robustness, or preservation under observation noise.

## What an executable certificate would require

A fitted readout and a dense derivative grid are only a feasibility screen.
A genuine certificate needs rigorous arithmetic enclosures for the derivative
residual on every box of a complete torus cover, with every uncertain activation
handled conservatively. All weights, embedding coefficients, and the readout
must be bound to exact saved coefficients. Unverified boxes or exhausted budgets
must return inconclusive. An independent replay must check arithmetic bounds and
coverage, not just a claimed maximum.

The exploratory screen is specified before execution in
`configs/continuous_fiber_feasibility_v1.json`. It uses all 13 primary gamma values
at seed zero; failed witnesses remain recorded. Even a successful numerical
screen is explicitly labeled `continuous_certificate: false`.

## Relation to prior work

Global injectivity and inverse stability of ReLU networks are established topics:
[Puthawala et al., JMLR 2022](https://www.jmlr.org/papers/v23/21-0282.html).
Bounded-domain injectivity and reconstruction are also studied using frame
theory: [Injectivity of ReLU Layers, 2025](https://link.springer.com/article/10.1007/s44439-025-00003-6).
General exact injectivity verification has complexity barriers:
[Froese, Grillo, and Skutella, COLT 2025](https://arxiv.org/abs/2405.19805v3).
This proposed restriction to declared nuisance circles is not a claim to solve
the general problem or bypass its complexity.

For a possible verifier, [FLINT's Arb matrices](https://flintlib.org/doc/arb_mat.html)
offer rigorous error bounds for operations without the `approx` prefix, and
[python-flint](https://github.com/flintlib/python-flint) exposes ball arithmetic
in Python. Library guarantees do not establish correctness of the surrounding
network encoding, domain cover, or verifier implementation; these need their own
audit and controls. This route would use an explicit arithmetic trust boundary,
separate from the existing Lean graph and logical lemmas.
