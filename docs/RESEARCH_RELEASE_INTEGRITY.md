# Release integrity audit

Two delivery gaps were corrected before creating a final full-study package.
The old release builder reformatted the frozen JSON while storing its original
checksum in the release index and analysis receipts. Equivalent JSON dictionaries
can have different byte hashes. The package now includes the exact source
manifest bytes, and reconstruction tests check both analysis checksum references.

The combined freeze also retains its original-study manifest as a hashed input
and records `original_freeze_path` together with its checksum. A complete
additional-analysis release requires that original manifest and all of its
unchanged input identities to be present in the combined freeze. This preserves
the original/full-gate provenance after restoration rather than leaving a checksum
whose referenced file is unavailable.

The release builder accepts both the original final-analysis manifest and the
additional summary. It checks the latter's figures, replication tables, consumed
input ledger, combined-freeze reference and cohort completeness. Missing tables,
changed figures, output paths outside the analysis directory, and inconsistent
original-study lineage prevent packaging. Inputs are hashed during the actual
archive read, so a change after the initial validation cannot silently enter the
archive under its old identity. Release part size must be a positive integer;
the reserved root `frozen_manifest.json` cannot collide with an input artifact.

The regression fixtures reconstruct both analysis sets and the exact original
and combined freeze references. They also exercise source changes after initial
validation and coherent-but-incomplete provenance edits. A synthetic layout
fixture checks byte-identical repeated PNG/SVG output in the locked runtime.
These are software and delivery checks, not new experimental findings or a
successful original/combined study freeze. Final scientific outputs still need
their own complete input audit, visual inspection and actual package verification.
