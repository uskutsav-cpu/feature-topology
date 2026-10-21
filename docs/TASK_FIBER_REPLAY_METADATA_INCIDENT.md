# Task-fiber replay runtime metadata incident

The original worker at b3208174c08468a4aac65b8459684f5a70fb79d9 completed all
210 saved-model evaluations. Its first separate replay stopped before issuing
any receipt because the strict comparator included the in-memory runtime
metadata. PyTorch exposes its version as a `TorchVersion` subclass of string;
JSON serialization restores an ordinary string with the same version text.
The comparator rejected that Python type difference in `result.host.torch`.

The first model's mathematical fields were recomputed and passed the original
numeric comparison separately from metadata. The original failure log,
source/result identities and diagnostic are retained in
`results/task_preserving_replay_attempts_v1`. This diagnoses the first failed
record; it does not claim that the other 209 evaluations have replayed.

A separate V2 replay script checks canonical JSON runtime identity exactly,
then compares unchanged mathematical fields with the unchanged relative 1e-10
and absolute 1e-12 tolerances. It invokes the original evaluation function and
keeps every generation result, checkpoint, grid, search setting and scope
unchanged. Its source identity and policy are bound in separate V2 receipts.
The original worker and original failed attempt remain available.

This corrects a metadata serialization bug in the replay integration. It does
not strengthen mathematical bounds, accept a changed numerical result, or
provide independent-verifier agreement. All 210 V2 numerical replays must
still pass before the component can be collected or frozen.
