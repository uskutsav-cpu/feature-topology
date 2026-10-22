# Verify the written research package

After the original and combined freezes, final analyses and figure inspections
pass, `pack_release.py` creates the multipart archive and `release_parts.json`.
The separate `verify_release_package.py` command reads the resulting package
back without extracting checkpoints or joining large parts on disk.

Record the SHA-256 of the exact combined freeze and release index at build time.
Use those trusted digests for verification. Deriving the expected digest from
the downloaded copy alone would not establish its identity.

For locally retained parts:

```sh
python scripts/verify_release_package.py \
  --index /path/to/release_parts.json \
  --index-sha256 BUILD_INDEX_SHA256 \
  --freeze-sha256 COMBINED_FREEZE_SHA256 \
  --parts-directory /path/to/package \
  --output /path/to/release_package_audit.json
```

For parts already uploaded to an existing GitHub release, use
`--release-tag TAG --repository uskutsav-cpu/feature-topology` instead of
`--parts-directory`. This reads GitHub's current asset identities, checks them
against the trusted index, then streams and hashes each body. It does not
publish, replace or remove release assets.

The audit checks numeric part order, every part's exact byte count and digest,
every regular archive member and its hash, the exact frozen-manifest bytes,
the complete member set and inclusion of every frozen input. It drains bytes
after the tar end marker so the last part and compressed trailer are included.
Paths outside the package root, duplicate members and links are rejected.

The freeze also includes the Python test sources. The restored project's
`pyproject.toml` points pytest at `tests`, and the packaged workflows invoke
that suite; shipping workflows without their tests would leave those checks
unusable after restoration. Bytecode caches remain excluded.

The implementation has small multipart corruption/omission controls and a test
that reads the actual packer's output. These tests establish the audit's
behavior on fixtures. The full research package must still be built and audited
from the actual frozen study before completion can be claimed.

A read-dependency audit of the full restored suite passed all 440 tests and
recorded 2,424 parent-process input paths. It identified two legacy CIFAR
component snapshots used by the cohort regression test but superseded in the
final research ledger. Exact copies now live in `tests/fixtures/cifar`, with
origin paths and byte hashes. The test's cell-set and exact-learning-rate
assertions remain unchanged. The original scientific collection reports are
preserved at their historical paths. The final package still requires its
actual complete freeze, read-back audit and restored-suite check; this input
trace alone is not that verification.
