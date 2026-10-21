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
