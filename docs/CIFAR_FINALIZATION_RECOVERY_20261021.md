# CIFAR continuation and collection recovery, 2026-10-21

The second continuation wave retained all outcomes. CIFAR-10 seeds 0 and 4 and
CIFAR-100 seeds 2 and 3 produced checked final archives. CIFAR-100 seeds 0, 1
and 4 reached the unchanged 19,000-second execution timeout in workflow
39530446041. Each uploaded an incomplete status and a resume archive. All three
new archive/member identities were independently checked before dispatching
workflow 39696195884 on the same immutable source tag and commit
06d5f685538e31542bbbd23a2c32cfde3c239aaf. Learning rates, seeds, total step
budgets and scientific metrics remain unchanged. This is another execution
attempt within the existing training budget, not a new experimental cohort.

The initial local finalizer stopped during CIFAR-10 calibration collection:
30/49 archives were accepted and 19 transfers failed with connection resets.
The report is preserved in the original execution directory. The separate
hosted finalizer 39539510307 accepted its calibration/CIFAR-10 components and
stopped when the three CIFAR-100 final archives were still absent. Neither
failure is counted as completion or a successful full-study freeze.

The operational finalizer now accepts an explicitly recorded continuation
workflow and a new in-repository output namespace. It still rejects a workflow
whose head differs from the fixed scientific source. Collection may retry up to
three times when every reported invalid item is a recognized network transport
error, preserving each attempt report. Content-validation errors and unfinished
training remain failures. All original source/configuration, archive, member,
calibration-map, source-ledger and held-out replay checks are unchanged.

The new local namespace is
`results/completion/local_cifar_finalization_wave3_20261021`. Earlier reports
remain in `results/completion/cloud_cifar_finalization_20261021`. Successful
calibration transport receipts may be reused only after the unchanged collector
rechecks their source/status identities and installed file hashes. No final
archive, replay tolerance, scientific failure, or risk threshold is replaced.
