# Paired hidden-retention and correct-decision evidence

The recorded sixteen-domain classifier study is complete. All sixteen domains
have positive correct-class certificates and successful separate-process Linux
replays. There are zero certified target-class counterexamples and zero
inconclusive attempts. The source-checked collection is
`results/nuisance_decision_verification_v1/collection.json`.

Each classifier result binds the same model checkpoint, input embedding and
fixed theta as its prior hidden-retention certificate. The complete centered
predictor, including both MLP branches and both heads, therefore keeps the
correct top-one class for every nuisance angle on the specified circle while
the current branch's last hidden representation is injective there.

| Gamma | Selected stored step | Certified circles | Smallest hosted scaled-logit lower margin across four circles (approximate) |
|---|---:|---:|---:|
| 0.03125 | 64 | 4 | 0.0213982 |
| 0.5 | 32 | 4 | 0.0120690 |
| 1 | 128 | 4 | 0.227484 |
| 8 | 128 | 4 | 0.353199 |

These are conservative lower bounds, not estimates of the exact minimum
logit margin or independent statistical observations. Each model uses seed zero.
The four theta values are pi/4, 3pi/4, 5pi/4 and 7pi/4. Every circle covers all
phi, with periodic endpoints. No every-theta or full-torus correctness follows.
Correct decisions do not require constant logits.

All sixteen additional Mac replays pass for separately stored weaker derived
margins in `results/nuisance_decision_local_replays_v2`. The deterministic
derivation halves the original positive numerator, rounding down at denominator
2^64. It was recorded after the first Mac replay failed to reproduce a stronger
hosted enclosure by approximately 3.5e-16. The original result and failure remain
unchanged. Local acceptance validates the weaker bound and the same strict
correct-class property; it does not validate the original hosted numerical
margin on the Mac. See `NUISANCE_DECISION_LOCAL_REPLAY_INCIDENT.md`.

The result concerns the real-valued continuation of exact saved float32
coefficients. Same-implementation interval replay is not an independent
verifier, a machine-floating-point guarantee, or a Lean proof of network
semantics. The component integration check passes against all four physical
checkpoints and the original embedding. That check is not the original or
combined full-study freeze, which still requires CIFAR completion.

The scientific implication is specific: a correct invariant decision on these
circles coexists with retained hidden nuisance variation. Hidden retention alone
does not tell us whether a classifier uses that variation, and a correct decision
does not imply its hidden representation has collapsed to the target quotient.
This does not explain or reverse the negative five-seed predictive comparison.

The general possibility of information-preserving classification is established
prior art, including [i-RevNet](https://arxiv.org/abs/1802.07088). The candidate
contribution is this source-bound a posteriori audit of ordinary trained models,
combined with the calibrated geometry and negative diagnostic comparison.
High novelty is not established merely by the sixteen successful certificates.
