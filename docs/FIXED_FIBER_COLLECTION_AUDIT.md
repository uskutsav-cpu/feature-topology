# Collection portability and replication units

The hosted V4 results and their sixteen successful replay receipts are unchanged.
Two derived collection issues were corrected before the combined results freeze.

First, collecting nested hosted artifact folders and recollecting the flattened
canonical directory initially produced different record orders. Strict comparison
therefore blocked the local preflight before arithmetic replay. Records now have
a canonical gamma/theta ordering. A regression test checks that nested-to-flat
collection preserves the complete manifest, and a tamper test rejects an inflated
separation factor even if its accompanying result digest is updated.

Second, the first derived manifest used `independent_trained_models` for a count
of four saved models. That descriptor was incorrect: the gamma conditions share
training seed zero and do not supply four independent seed replicates. The
canonical field is now `trained_models`, with `training_seeds: [0]`. The original
derived metadata is retained as `collection.original_metadata.json` (SHA-256
`d64401baafae39bcfed9d984fdb7100f2107a645166c5d36f380c9fb2de30fc0`).
Every proof record and result/replay digest is identical before and after this
descriptor correction. The historical metadata must not be used to count
statistical replicates.

Sixteen certified circles are four declared theta domains in each of four
gamma-conditioned models. Additional executions on another host check numerical
and packaging reproducibility; they add no new trained models or independent
statistical seeds and do not constitute an independently implemented verifier.
