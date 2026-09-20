# Scientific claim and contribution contract

## Question

Let the data manifold be a product \(\mathcal M=B\times F\), with task-relevant
base \(B\) and nuisance fiber \(F\), and let a learned layer be
\(h_\gamma:\mathcal M\to\mathbb R^d\).  This project asks:

> As feature-learning strength \(\gamma\) increases, when does the representation
> merely deform the nuisance fiber, when does the fiber become numerically
> near-singular, and when are distinct nuisance states actually identified?

For the primary synthetic study, \(B=S^1_\theta\), \(F=S^1_\phi\), and
\(\mathcal M=T^2\).  For a deterministic task \(Y=f(\theta)\), define
\(x\sim_Yx'\) when the conditional target law is identical.  The task quotient is
\(\mathcal Q_Y=\mathcal M/\!\sim_Y\); in the pure-nuisance case it is
\(T^2/S^1_\phi\cong S^1_\theta\).

The representation induces a second equivalence relation,
\(x\sim_hx'\iff h(x)=h(x')\).  Task sufficiency requires that \(\sim_h\) never
merge task-distinct points.  Exact task quotienting requires
\(\sim_h=\sim_Y\).  A small distance, a small Jacobian, failed decoding, or a
persistent-homology change is not by itself equality of these relations.

## Fiber-survival hierarchy

For a smooth map and a one-dimensional fiber, the primary quantities are:

\[
L(h)=\inf_{x\in\mathcal M}\sigma_{\min}(Dh_x|_{T_xF}),
\qquad
M(h)=\inf_b\inf_{f_1\ne f_2}
\frac{\|h(b,f_1)-h(b,f_2)\|}{d_F(f_1,f_2)},
\]

and the fiber collision relation

\[
C_h(b)=\{(f_1,f_2):f_1\ne f_2,\ h(b,f_1)=h(b,f_2)\}.
\]

They define four conceptually different regimes:

1. **Preserved:** geometry is close to the reference and \(L,M\) stay bounded away from zero.
2. **Deformed:** geometry changes substantially while \(L,M>0\) and no collision is established.
3. **Near-singular:** \(L\) or \(M\) is small, but no collision is established.
4. **Quotiented:** distinct nuisance states are identified; exact quotienting additionally requires the induced equivalence relation to match the task equivalence relation.

Finite-grid measurements are reported as sampled evidence.  The words
"injective" and "quotiented" are reserved for analytic controls, exact restricted
certificates, or statements whose domain and proof status are explicit.

## Headline contributions under test

1. A fiber-survival hierarchy that separates geometry, differential
   degeneration, global separation, and actual equivalence-class formation.
2. A theorem/counterexample package showing that a positive global margin gives
   fiberwise injectivity, while an everywhere-positive local margin does not.
3. Matched-risk \((\gamma,\lambda)\) experiments testing how nuisance relevance
   changes the cost and prevalence of compression or quotienting.
4. A link from the diagnosed regime to nuisance-shift generalization, tested with
   width scaling, exact restricted certificates, and factor-controlled images.

Contributions 1–2 now have analytic controls and executable tests.  Contributions
3–4 remain hypotheses until the frozen production protocol passes.  The project
will use "crossover" or "regime change" unless finite-width scaling supports the
stronger phrase "phase transition."
