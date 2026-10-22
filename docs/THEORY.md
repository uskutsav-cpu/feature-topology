# Fiber survival: statements and proofs

This note fixes the assumptions behind the paper's theoretical claims.  It does
not promote finite numerical evidence into a continuum theorem.

## Proposition 1: global separation implies fiberwise injectivity

Let \((F,d_F)\) be a metric space and \(h:B\times F\to\mathbb R^d\).  Suppose
there is \(m>0\) such that for every \(b\in B\) and all \(f_1,f_2\in F\),

\[
\|h(b,f_1)-h(b,f_2)\|\ge m d_F(f_1,f_2).
\]

Then \(h_b:f\mapsto h(b,f)\) is injective for every \(b\).  Moreover, its inverse
on its image is \(1/m\)-Lipschitz.

**Proof.**  If \(h_b(f_1)=h_b(f_2)\), the left side is zero, so
\(m d_F(f_1,f_2)\le0\).  Since \(m>0\) and \(d_F\) is a metric,
\(f_1=f_2\).  Rearranging the same inequality gives the inverse-Lipschitz
bound. \(\square\)

The sampled statistic in code is not the premise of this proposition: it is an
estimate over a declared grid.  An analytic bound, interval bound, or exact
restricted certificate is needed for the theorem.

## Proposition 2: local regularity does not imply global injectivity

On \(T^2=S^1\times S^1\), define

\[
h(\theta,\phi)=(\cos\theta,\sin\theta,\cos2\phi,\sin2\phi).
\]

Then \(\|Dh|_{T_\phi F}\|=2\) everywhere, yet
\(h(\theta,\phi)=h(\theta,\phi+\pi)\).  Thus \(L(h)=2>0\) while \(h\) is
two-to-one on each nuisance fiber and \(M(h)=0\).

**Proof.**  Differentiating the last two coordinates gives
\((-2\sin2\phi,2\cos2\phi)\), whose norm is two.  Periodicity gives equality at
\(\phi\) and \(\phi+\pi\), which are distinct on \(S^1\). \(\square\)

This counterexample is implemented as `double_cover` in
`src/data/ground_truth.py` and tested independently of neural training.

## Proposition 3: task-information loss is exactly conditional information

Let \(Z=h(X)\) be deterministic and assume the relevant entropies are finite.
Then

\[
H(Y\mid Z)-H(Y\mid X)=I(Y;X\mid Z)\ge0.
\]

**Proof.**  Because \(Z\) is a function of \(X\),
\(H(Y\mid X,Z)=H(Y\mid X)\).  Substitute this into
\(I(Y;X\mid Z)=H(Y\mid Z)-H(Y\mid X,Z)\). \(\square\)

Merging a pure nuisance orbit can therefore have zero information cost.  Strict
positivity requires that the representation lose target-relevant information on
a set of positive probability; an isolated collision between two zero-mass
points does not establish positive conditional mutual information.
This identity does not imply that finite probes recover all retained information.

For the sector-classification experiment, the quotient that removes only the
nuisance circle retains the continuous task coordinate.  It is not the minimal
target-law quotient, which also identifies theta values within each label
sector.  The manuscript distinguishes these two equivalence relations.

## Theorem 4: feature change need not destroy nuisance information

Let \(X=(B,F)\), with \(B\in\mathbb R^p\), \(F\in\mathbb R^q\), and consider a
block representation

\[
h_{A,C}(B,F)=(AB,CF).
\]

Let the predictor be \(g_{u,A,C}(B,F)=u^TAB\), and let the empirical objective
be any differentiable supervised loss

\[
\mathcal L(A,C,u)=\frac1n\sum_i \ell(u^TAb_i,y_i).
\]

Under gradient flow (or ordinary gradient descent without regularization on
\(C\)), \(C_t=C_0\) for all time.  If \(C_0\) has full column rank, the nuisance
factor remains linearly recoverable throughout training, even when \(A_t\) and
the task geometry change arbitrarily far from initialization.

**Proof.**  The objective is independent of \(C\), hence
\(\nabla_C\mathcal L=0\) identically and the update of \(C\) is zero.  Full column
rank gives a left inverse \(C_0^\dagger C_0=I_q\), so applying the left inverse
to the nuisance block recovers \(F\).  No corresponding bound restricts the
gradient or displacement of \(A\). \(\square\)

The theorem is deliberately modest but mechanistic: supervised feature learning
can strongly reorganize task directions without any mathematical necessity to
erase an unused factor.  Whether coupled nonlinear networks preserve, compress,
or identify their nuisance fibers is therefore an empirical question, not a
consequence of "strong feature learning" alone.

## Interpretation check: exact almost-sure nuisance erasure

These elementary continuity observations support interpretation; they are not
claimed as new theory or included in the seventeen-lemma Lean audit.

Let h be jointly continuous on the primary torus, with independent uniform
latent angles Theta and Phi. Suppose one rigorously checked fixed-theta circle
contains two angles phi_1, phi_2 with different hidden representations. Then
there is no measurable g for which h(Theta,Phi)=g(Theta) almost surely.

**Proof.** Define the continuous nonnegative function

    D(theta,phi,phi') = ||h(theta,phi)-h(theta,phi')||.

If h=g(Theta) almost surely, Fubini's theorem implies D=0 almost everywhere under
the product law of (Theta,Phi,Phi'). That product measure has full support.
A continuous nonnegative function that is zero almost everywhere under a
full-support measure is zero everywhere: any positive value would have an open
positive-measure neighborhood. This contradicts the checked pair. Equivalently,
a positive D at that pair persists on an open set of triples, so a positive
measure of theta values has nonconstant conditional hidden representations.

This excludes **complete exact** removal of phi from that hidden representation
under the specified product law. It does not prove fiberwise injectivity for
almost every theta, unconditional recoverability of phi without theta, a
quantitative information/noise bound, or predictor robustness. The sixteen
circle certificates still have their declared point-theta injectivity scope.

## Interpretation check: finite class centers on a connected continuum

Let M be connected, h:M->R^d continuous, and the data law have full support.
If h takes values in a finite set almost surely, it takes values in that set
everywhere: the preimage of the complement is open and must be empty. But a
continuous image of a connected space is connected, and a finite subset of
Euclidean space is connected only when it has one point. Thus h is constant.

Consequently, an exact finite-valued label-only hidden representation cannot
also be sufficient for multiple positive-probability classes under these
continuum assumptions. This is a structural observation, not an empirical
discovery. It does not rule out an S1 nuisance-orbit quotient, approximate class
clustering, or classification by a discontinuous decision rule applied to
continuous logits. A finite training point cloud is not a connected continuum,
so the same obstruction does not apply to exact collapse of its sampled classes.

## Exact control family

The executable controls are:

| map | local margin | global margin | injective on each fiber? |
|---|---:|---:|---|
| product-circle isometry | \(1\) | \(2/\pi\) | yes |
| radius/twist deformation | \(e^{-2}\) | \(2e^{-2}/\pi\) | yes |
| \(10^{-4}\)-scaled nuisance circle | \(10^{-4}\) | \(2\cdot10^{-4}/\pi\) | yes |
| nuisance double cover | \(2\) | \(0\) | no |
| task quotient | \(0\) | \(0\) | no |

The computed figure and JSON are under `results/ground_truth_controls/`.  They
show that CKA, local margins, persistent homology, and clean probe accuracy each
fail to distinguish at least one pair of regimes; the hierarchy succeeds because
it does not ask any single diagnostic to stand in for identification.
