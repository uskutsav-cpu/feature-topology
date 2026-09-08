# Protocol and interpretation

H1: increasing output-scale feature-learning strength increases geometric deformation.
H2: geometric deformation can occur without loss of injectivity.
H3: some strong-learning settings may collapse an injectivity margin or latent factor.
H4: geometric untangling and information destruction are distinct phenomena.
H3 is an empirical question, never an acceptance criterion.

The primary experiment uses a 16-dimensional orthogonal embedding of a product
of two circles, four theta classes, four width-256 ReLU hidden layers, and vanilla
SGD. Training, validation, analysis-grid, and probe/test draws have distinct fixed
seeds; the embedding is shared. Initialization and minibatch indices are paired
across gamma. Depth in the LR reference exponent counts the output affine map.

The default output is `(f_theta(x) - f_initial(x))/gamma`. The initial branch is
frozen. This avoids changing initial logits across gamma and follows the centered
setup in Atanasov. Set `centered=false` for the literal uncentered brief variant.
This implementation uses Kaiming-initialized standard parameters, not a claim of
exact muP reproduction. Architecture changes require separate LR calibration.
Empirical NTK and representation drift must verify that gamma spans regimes.

Calibration uses independent seeds, stable optimization, reaching a fixed training
loss, and step count. Stable trials that miss the target remain explicitly marked.
The selected map is frozen by configuration fingerprint. Test accuracy and latent
metrics are not consulted during calibration. Main seeds never retune the map.

Matched-risk comparisons use the first stored checkpoint reaching the requested
loss, report its actual loss and step, and exclude runs that never reach it.
Accuracy crossings are detected at evaluation checkpoints (not every SGD step).
This discretization must be stated in the report. No interpolation is a substitute
for actually reaching a target risk.

Geometric metrics, raw and scale-normalized tangent margins, sampled global
margins, nearest-neighbor collisions, held-out linear/MLP probes, and PH are
reported separately. Positive sampled tangent singular values do not prove global
injectivity; small margins do not prove exact collisions. ReLU derivatives at
activation boundaries use PyTorch's convention. Finite probes cannot certify
absence of all decodable information. The NTK uses the trace over class outputs,
accumulated by parameter tensor to bound memory, with a fixed point subset.

PH uses RMS-normalized representations and fixed, paired subsample indices across
checkpoints; raw scale is retained separately. H1 bar lifetimes are diagnostics,
not estimates of exact Betti numbers without a stated filtration scale. H2 resource
limits and any reduced pilot subsample sizes must be reported. Subsamples are not
independent training seeds and cannot inflate the number of statistical replicates.

Nulls include scaling, rotation, a latent alignment row permutation (preserves
the point cloud exactly), and a covariance-matched Gaussian cloud (only approximate
geometry matching). The latter must have matching errors measured; it does not
guarantee matching norms or pairwise distance histograms.

For fractional lambda, `theta + lambda*phi` is coordinate-dependent across the
circle seam. The default well-defined periodic alternative is
`theta + lambda*sin(phi)`; `relevance_mode=literal` preserves the requested formula
for an explicitly seam-dependent sensitivity analysis. These are different tasks.

The small quotient routine resolves ReLU breakpoints on a polygonal circle and
uses numerical LP intersections and union-find image-graph Betti numbers. It is
not an exact smooth torus result or a reproduction of all Beshkov theorems.

References:
- Atanasov et al., https://arxiv.org/abs/2410.04642
- Yeom et al., https://arxiv.org/abs/2602.00827
- Beshkov, https://openreview.net/forum?id=RluspxztzS
- PyTorch NTK tutorial, https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
- Ripser API, https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html
