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

Merging a pure nuisance orbit can therefore have zero information cost, while
merging points with different target conditionals makes the right side positive.
This identity does not imply that finite probes recover all retained information.

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
