# Exact catalog input accounting

Sixteen recovered primary runs contain both pilot and production metric folders.
The production-profile filter selected the correct scientific rows, but the
report's former input ledger globbed every JSON file after analysis. It therefore
listed thirty unselected pilot metric records that the filtered analysis never
read. Conversely, the catalog reads every discovered profile's options to report
profile resolution, including the options of an unselected pilot profile.
Those audit inputs also need a freeze binding.

The catalog now records the exact bytes of each JSON read, using the same strict
duplicate-key and non-JSON-number checks as the existing parser. Reports retain
that read ledger. The original readiness pass binds all JSON inputs it actually
audits, including legacy profiles, in addition to the required production
checkpoints and numerical arrays. Analysis profile selection remains explicit.

A preflight on the real 130-run primary collection checks 4,356 selected metric
rows against the previous committed reader. Every scientific row is identical.
The filtered analysis records 1,625 actual JSON reads; the unfiltered readiness
catalog reads 1,655. Every filtered read is covered by the readiness bindings.
Regression tests exercise coexisting profiles, metadata-only reads, exact byte
hashes, changed inputs, and the unchanged full-readiness requirement. This is a
component preflight, not a declaration that the study is frozen.

Because the comparison records its catalog-source digest, a fresh source-bound
snapshot is required after this reader change even though its numerical
flattening function is unchanged. `additional_diagnostic_comparison_portable_v2`
retains that new snapshot; the preceding snapshots remain available. The
refreshed 140-cell snapshot has identical input rows, input hashes, and numerical
comparison outputs, including all three negative incremental-benefit findings.
Only its catalog-source digest changes. The predictive protocol and the full
original-study gate continue to determine analysis eligibility.

The full physical-input check was also rerun after the accounting change. It
still reports exactly four CIFAR-related gaps: two incomplete image cohorts,
the final held-out replay report, and the canonical source ledger. Its retained
report is `results/completion/neural_networks_preparation/full_readiness_input_ledger_20261020.json`.
