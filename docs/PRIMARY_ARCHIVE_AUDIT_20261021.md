# Primary archive transport audit

The external validation drive is temporarily unavailable. The earlier analysis
mirror contains 56 primary receipts whose training inputs were checked using
existing config/summary member hashes rather than downloading each complete
input archive. The remaining synthetic profiles already have full transport
receipts. Original physical checkpoint audits are a separate requirement.

The workflow `primary_archive_audit_20261021.yml` runs the unchanged streaming
verifier on all 130 primary profiles, with no local-source shortcut or reused
receipts. It verifies every compressed input/metric archive and every member
against the recorded identities while retaining only JSON evidence. It preserves
failures and does not overwrite the earlier receipts. This makes the new audit
independent of the absent drive without changing training or analysis settings.

Successful transport verification does not validate tensor semantics, replace
the original dSprites or MPS handoff files, establish the complete research
freeze, or demonstrate scientific novelty. Completion must be established from
the actual workflow output and retained receipts, not this execution plan.

## Observed result

[Workflow 39540638096](https://github.com/uskutsav-cpu/feature-topology/actions/runs/39540638096)
completed successfully from source
`dfa8e43d0f06231dcc5eddac38306b0bed405f3b`. All 130 primary profiles have full
archive-and-member receipts with no failures. The authenticated downloaded ZIP
has SHA-256 `210fd45faea78bf4851ff0274d90a0d3ea0f41b61e3749e9963396a4cdf15c0d`.

The new receipts cover all 56 previously JSON-only primary checks. Combined
with the 715 earlier hosted ablation receipts, the exact declared set of 845
profiles is covered. Every receipt's input archive identity was checked against
the source inventory, and all 7,719 retained file hashes were rechecked against
the available JSON mirrors. The 845 receipts, workflow observations and audit
summary are retained separately in `results/synthetic_archive_transport_v1`.
Earlier attempts and receipts remain untouched. The physical-input freeze and
original-file requirements above are still pending.

The full results-freeze gate now validates this collection against the current
cohort indices and exact collector archive names, and binds all 845 receipts
and their source observations into the freeze. Missing receipts, rewritten
archive identities, changed observation bytes and paths outside the repository
are rejected. This ensures the transport evidence survives into the final
research package while preserving the separate physical-input checks.
