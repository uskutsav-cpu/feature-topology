# Completing the project without the external SSD

The external validation volume became unavailable after all 210 initial Mac
task-fiber evaluations completed and before a successful replay receipt was
issued. Those outputs are not claimed replayed. Their source identity,
generation-stage checksum and original metadata failure remain recorded in
`results/task_preserving_replay_attempts_v1`. The unchanged Mac files remain
on that unavailable volume; this campaign does not overwrite them.

The user requested continued work without the SSD. The separate execution plan
`configs/task_fiber_hosted_execution_v1.json` pins the SHA-256 identities of
all 18 mathematical source files from b320817, plus the metadata-only V2 replay
script. Eight deterministic shards regenerate all 210 models' evaluations from
immutable archived checkpoints and original analysis archives. Four shards run
concurrently. Every input archive/member is verified before extraction, and the
selected checkpoint and canonical source path must match the already checked
production-noise source binding. No training is performed.

Each model's generation and replay run in separate Python processes on the same
hosted Linux runner with two threads. Source/checkpoint identities, actual
workflow run, shard, orchestration code and result/replay checksums are retained
in per-model execution receipts. Scientific settings, search budget, checkpoint
selection, metrics and numerical tolerances are unchanged. The distinct final
collection path is `results/task_preserving_fiber_completion_hosted_v1`.

These are regenerated numerical outcomes on Linux, not cross-platform replay
agreement with the unavailable Mac outputs, and not additional training-seed
replicates. Hosted completion, strict collection and the full physical-input
freeze remain required. The committed replication and 305-model noise results
remain available in GitHub independently of the SSD. Remaining CIFAR work also
continues on its existing immutable-source GitHub workflows.
