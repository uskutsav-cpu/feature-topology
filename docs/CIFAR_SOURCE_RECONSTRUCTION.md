# Recovering historical CIFAR provenance without the SSD

The original local queue-controller JSON files were present in the external
validation workspace but were not tracked in the repository. Their exact bytes
are currently unavailable. The new source ledgers explicitly reconstruct public
execution provenance; they do not claim to recover those original file identities.

`scripts/reconstruct_cifar_provenance.py` verifies the immutable release tag's
source commit, downloads status manifests, checks their authenticated GitHub
asset digests and validates the typed cell/configuration/host identities. It
retains the source-filtered workflow inventory and every job response, including
failed and cancelled executions. Assigned cells are linked to terminal workflow
metadata. A source with active or unresolved worker assignments cannot receive
a complete reconstructed ledger.

Decimal job-name displays are not exact learning-rate evidence. In the original
gamma-128 incident the matrix transport rounded a learning rate. The builder
therefore resolves decimal names through uniquely matching typed archived
source-status cells. Hexadecimal job fields retain their exact binary value.
Ambiguous or missing typed identities remain unresolved; no LR is guessed.

Three historical source components have been reconstructed and their evidence
bindings checked:

| Component | Source commit | Terminal attempts | Bound observation/API/status files |
|---|---|---|---|
| CIFAR-10 original | d06123ebdc4a23e8d4bec976e2fe92793b9795a0 | 7 | 117 |
| CIFAR-100 original | b20f3c77bd620389dbeb542f0b48789f754a2a8e | 7 | 134 |
| CIFAR-10 exact-LR correction | 8816fc099e3d5135f191597ff50764ad87d80af0 | 2 | 13 |

The retained files are under
`results/completion/cifar_source_reconstruction_20261021`. Counts describe
provenance coverage, not successful training or complete production cohorts.
The quarantined rounded-LR results and their original audit remain separate
from the corrected production cells.

The full freezer now checks every reconstructed observation and its API/status
evidence hashes, the exact terminal attempt list and the absence of unresolved
or active assignments. These files are included in the frozen input set rather
than relying only on a derived controller summary. Existing authored ledgers
without a reconstruction field retain their original validation path.

The seven source-06d5f68 continuations remain active and must finish unchanged.
Their terminal source ledgers will be reconstructed separately after completion.
Production archives must still pass the original archive/tensor checks; the
multi-source production reports must cover all 35 cells per dataset; all 70
held-out evaluations must pass. The source reconstruction does not bypass any
of those requirements or the original/combined full-study freeze.
