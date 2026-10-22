# Additional Mac numerical-margin replay, 2026-10-20

Workflow 39509448706 produced sixteen positive complete-classifier circle
certificates and sixteen successful separate-process Linux replays, all bound
to source commit 8dac5ed77de82ea7fa43a0745579402028935031. These files remain
unchanged in `results/nuisance_decision_verification_v1`.

The first additional Mac replay failed on gamma 0.03125, theta pi/4. On leaf
011000111 against competitor class 3, its lower enclosure was approximately
0.0241369042917824189, below the hosted declared margin of approximately
0.0241369042917827705. The difference was about 3.516e-16. Every recomputed
local leaf bound was still positive. The original numerical-margin claim did
not pass this additional local check and is not reported as having passed it.
The diagnostic record and original local replayer bytes are retained under
`results/nuisance_decision_local_attempts_v1`.

Rigorous enclosures need not be identical across builds or execution platforms.
The exact implementation cause of this small enclosure difference has not been
isolated. This incident does not establish that the Linux enclosure is unsound,
nor does positive Mac classification justify accepting its stronger numerical
margin without replay.

The separately recorded V2 local rule constructs a weaker claim by taking
floor(n/2)/2^64 from the original positive margin n/2^64. It requires this new
margin to remain strictly positive. It changes no network, input, domain,
candidate, binary cover or source binding. There is no adaptive margin search.
Negative point claims, if present, are copied unchanged. The original result
digest, full derived claim, new claim digest, verifier digest and platform are
saved separately. The local core recomputes every leaf and accepts only if it
proves the derived bound. Failure still prevents local acceptance.

The combined gate requires the exact deterministic derivation and all sixteen
local receipts. It explicitly records that local validation concerns the weaker
derived margins and does not validate the hosted numerical margin on the Mac.
All executions use the same Arb verifier implementation; they are not independent
semantic soundness proofs or Lean proofs. The scientific correct-class property
is unchanged because both accepted margins are strictly positive.
