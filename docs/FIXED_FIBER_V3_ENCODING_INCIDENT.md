# V3 replay incident and exact proof bounds

Workflow 39493171088 generated sixteen positive candidates for the previously
declared sixteen smooth circles. Thirteen obtained successful separate-process
replay receipts. Three failed with `Replayed separation factor mismatch`:
gamma 0.03125 at theta = 3 pi/4, gamma 0.5 at theta = 5 pi/4, and gamma 1 at
theta = 7 pi/4. Those three candidates were not accepted as certificates.
The complete attempts, thirteen receipts, and failure logs are retained in
`results/fixed_fiber_verification_v3_attempts`.

V3 serialized a 128-bit Arb separation enclosure as a decimal string, parsed it
again, and required the parsed interval to contain a recomputed interval. For
each failure the intervals overlap and their displayed midpoints agree, but the
containment test fails at approximately 1e-39. Overlap does not satisfy the
declared acceptance rule; this diagnosis does not retroactively certify the
three failed cases. V3 source and artifacts remain unchanged.

V4 uses exactly encoded dyadic inequalities. After computing the maximum of all
leaf residual upper bounds, it rounds strictly upwards to a multiple of 2^-20
and adds an extra unit. This exact epsilon must remain below 2/pi. The resulting
positive chord separation factor is rounded strictly downwards to a multiple of
2^-64 with an extra unit subtracted. Canonical integer strings encode the two
fractions. Replay recomputes every leaf and requires its upper bound to be at
most the exact epsilon, then requires the exact positive factor to be at most
the rigorous recomputed factor. Approximate display strings have no proof role.

V4 additionally requires each generated leaf to lie below 2/pi - 4*2^-20, leaving
room for outward rounding. This strengthens the leaf acceptance condition. It
does not relax the mathematical criterion, alter the checkpoint, refit a witness,
change the domain, or increase the budgets. All sixteen cases are attempted
again. A successful run would establish only the declared fixed-theta circles
in the real-valued continuation of saved weights. The trusted arithmetic and
same-implementation replay boundary remain as recorded in the protocol.

The subsequent V4 workflow 39498048922 completed all sixteen cases with positive
certificates and successful separate-process replay receipts. The source-checked
collection is retained in `results/fixed_fiber_verification_v4`. This is a new
execution under V4; the three rejected V3 replay candidates remain unaccepted.
