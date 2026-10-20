# Reproducing the additional research

Run from the repository root, using the locked environment. The optional
continuous-circle verifier also needs `requirements.certification.txt`.
Checkpoints and arrays come from the immutable GitHub release indices; workers
verify their SHA-256 identities. A network connection is required to recover
uncached release inputs.

## One specified continuous nuisance circle

```sh
python scripts/remote_rational_fiber_verifier.py --repo . --gamma 8 --theta-numerator 1 --output work/circle-gamma8-theta1
python scripts/remote_rational_fiber_verifier.py --repo . --gamma 8 --theta-numerator 1 --output work/circle-gamma8-theta1 --replay
```

The second invocation is a separate process that reloads the source inputs and
recomputes every leaf of a positive result. A candidate JSON alone is not an
accepted certificate. Inconclusive or rejected witnesses retain their status.
The recorded sixteen-case matrix is in
`.github/workflows/fixed_fiber_verifier_v4.yml`; the fixed domain, witnesses,
budgets, and exact-bound encoding are in `configs/fixed_fiber_verifier_v4.json`.

After downloading all matrix artifacts to a new directory, collect all outcomes:

```sh
python scripts/collect_fixed_fiber_results.py --repo . --artifacts work/downloaded-fiber-artifacts --output results/fixed_fiber_verification_v4
```

The collector requires all sixteen attempts, source and checkpoint bindings,
complete circle covers, valid exact positive margins, and successful source-bound
replay receipts for every positive candidate. It cannot silently exclude failed
cases. Existing result directories are immutable; use new paths for separate
campaigns and explicitly update a protocol before changing the scientific design.

## Full study and additional evidence freeze

The original physical inputs must first pass the existing full gate:

```sh
python scripts/freeze_results.py --repo . --output results/completion/frozen_results_20261020.json
```

This command fails while CIFAR coverage, source ledgers, held-out replays, or
other required artifacts are incomplete. A partially populated checkout is not
enough. Required physical files include the source-bound checkpoints and official
image datasets. The additional gate also requires the unchanged embedding asset
from `primary-inputs-v3-2026-09-20` at
`results/continuous_fiber_inputs/analysis_data.npz`; its identity is the SHA-256
in `configs/completion/primary_inputs_v3.json`.

The validation workspace uses internal aliases under
`results/additional_source_map` to locate the original primary and OOD run JSONs.
The portable comparison directory contains the unchanged 140-cell comparison
recomputed from those internal paths. It must have been produced by the current
`scripts/diagnostic_comparison.py`, including analysis-source hashes.

After recovering the original run JSONs in their canonical study directories,
create that map without copying data or changing existing aliases:

```sh
python - <<'PY'
import json, os
from pathlib import Path
repo = Path('.').resolve()
root = repo/'results/additional_source_map'
root.mkdir(exist_ok=True)
registry = json.loads((repo/'configs/completion/production_cohorts_v3.json').read_text())
for cohort in registry['cohorts']:
    if cohort['name'] not in {'primary','ood-spurious','ood-concentrated','ood-unseen'}:
        continue
    target = (repo/'results/main' if cohort['name']=='primary'
              else repo/'results/ood'/Path(cohort['index']).stem)
    assert target.is_dir() and target.resolve().is_relative_to(repo)
    alias = root/cohort['name']
    relative = os.path.relpath(target, root)
    if alias.is_symlink():
        assert os.readlink(alias)==relative, 'Different existing source alias'
    elif alias.exists():
        raise ValueError('Source alias path already contains another object')
    else:
        alias.symlink_to(relative, target_is_directory=True)
PY
python scripts/diagnostic_comparison.py --repo . --evidence results/additional_source_map --ood results/ood_evaluation --variability results/additional_variability --aliases results/checkpoint_aliases --output results/additional_diagnostic_comparison_portable_v1
```

```sh
python scripts/freeze_additional_research.py --repo . --original-manifest results/completion/frozen_results_20261020.json --evidence results/additional_source_map --comparison results/additional_diagnostic_comparison_portable_v1 --output results/completion/combined_research_freeze_20261020.json
python scripts/final_analysis.py --repo . --manifest results/completion/combined_research_freeze_20261020.json --output results/final_analysis_20261020
python scripts/analyze_additional_research.py --repo . --manifest results/completion/combined_research_freeze_20261020.json --output results/additional_final_analysis_20261020
```

Use a new empty analysis output directory. The combined gate cannot bypass a
failed original freeze. It replays the predictive comparison, checks circle
collection bindings and receipts, and binds the original failed/inconclusive
attempts as well as the physical embedding input. The final analyses reject
consumed files absent from the freeze or changed after it.

The additional figures explicitly separate sampled grid metrics, whole-circle
bounds on four seed-zero models, and the five-seed predictive comparison.
Circle bounds are descriptive exploratory evidence; they do not increase the
number of independent training seeds. The recorded negative predictive result
and the limits in `CONTINUOUS_FIBER_CRITERION_AUDIT.md` remain part of the report.
