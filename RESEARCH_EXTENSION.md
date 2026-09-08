# Feature-topology research extension

**A tested additive implementation, not a claim that the full research study is finished.**

Audited base: `uskutsav-cpu/feature-topology` at
`d6c3f34b41feab45d7d963228c5c3e327098203b`.

## Install without overwriting your work

Unzip the bundle. From your existing repository root, run the installer from the
unzipped bundle (adjust its path to your actual download location):

```sh
python3 ~/Downloads/feature-topology-update/APPLY_UPDATE.py "$PWD" --dry-run
python3 ~/Downloads/feature-topology-update/APPLY_UPDATE.py "$PWD"
```

The installer verifies the payload's SHA256 hashes, checks the repository HEAD,
and refuses different existing files, symlinked destinations and unsafe paths.
It never changes your original source or results and does not commit or push.
It uses exclusive hard-link creation; a filesystem without hard-link support
fails safely. The manifest is an integrity record, not a publisher signature.

A different HEAD is rejected by default. Review compatibility first, then use
`--allow-different-base` only deliberately. That option still never allows
existing different files to be overwritten.

Use the same Python environment as your existing training runs. The commands
below assume `python` points to it. `requirements-research-ext.txt` lists extension
dependencies, but do not silently upgrade a frozen production environment.
The delivery validation used Python 3.13.5 and PyTorch 2.10.0+cpu; exact versions
are in `validation/smoke/environment.json` in the downloadable bundle.

## Check the installation

```sh
python -m pytest tests/research_ext -q
python scripts/research.py controls --output results/research_ext/exact_controls
python scripts/research.py smoke --output results/research_ext/smoke
```

`smoke` trains four tiny real configurations using your existing trainer: two
gammas × two seeds, width 8, depth 2, and 32 optimizer steps. It computes inexpensive
real metrics, runs the new analysis and replays an exact certificate from a trained
checkpoint. It deliberately omits PH and nonlinear-probe measurements. A smoke
success does not validate the 256-wide production study or establish H1–H4.

## Inspect and finish the primary computation

Do not start a second writer on directories being used by old running scripts.

```sh
python scripts/research.py audit --roots results/main \
  --output results/research_ext/current_audit.json

python scripts/research.py plan --suite primary \
  --output results/research_ext/primary_plan.json

# Read primary_plan.json, then execute its sequential stages:
python scripts/research.py execute-plan results/research_ext/primary_plan.json
```

The primary plan calls your existing calibration, training and full-trajectory
metric scripts, then produces the new audited report. It retains the pilot gate,
frozen learning-rate map and compatible pilot reuse. It hashes inputs, source,
environment and outputs for resume decisions and stops when a stage fails.
A stage exit alone does not imply convergence or a successful hypothesis.
Logs and state are under `results/research_ext/workflow_state/`.

For an already computed profile, use its exact profile-directory name to avoid
mixing settings:

```sh
python scripts/research.py analyze --roots results/main \
  --profile YOUR_METRIC_PROFILE_DIRECTORY \
  --output results/research_ext/analysis --plots
```

Configuration is `configs/research_ext/analysis.json`. The default common loss
threshold is 0.1; actual measured losses remain in the tables. No threshold-event
rules are invented. Add rules deliberately and record whether they are
preregistered or exploratory. The default profile selection is explicit in the
primary workflow plan.

## Ablations and image studies

```sh
python scripts/research.py plan --suite ablations --output results/research_ext/ablation_plan.json
python scripts/research.py execute-plan results/research_ext/ablation_plan.json

python scripts/research.py plan --suite images --image-device cpu \
  --output results/research_ext/image_plan.json
python scripts/research.py execute-plan results/research_ext/image_plan.json

# Requires CIFAR datasets already available to the original runner.
python scripts/research.py plan --suite cifar --image-device cpu \
  --output results/research_ext/cifar_plan.json
python scripts/research.py execute-plan results/research_ext/cifar_plan.json
```

The image plans wrap the existing runners; they do not assert those runners have
been fully validated here. Apple MPS is also an upstream option. CUDA is not added
by this wrapper. CIFAR files are not downloaded automatically. Image artifact
completion is audited with the native image schema rather than pretending its
measurements are torus-tangent measurements. dSprites remains unfinished.

## Decoder fragility diagnostic

```sh
python scripts/research.py noise-checkpoint \
  --checkpoint results/main/runs/RUN_ID/final.pt \
  --output results/research_ext/noise_RUN_ID.json
```

This measures how a clean-trained circular ridge decoder tolerates controlled
noise in held-out representations. It is not a proof of information destruction.
Run across independent training seeds before making population-level claims.

## Exact small-network checks

```sh
python scripts/research.py verify-certificate \
  results/research_ext/exact_controls/identity.json

python scripts/research.py export-checkpoint \
  --checkpoint PATH_TO_SMALL_MLP_CHECKPOINT --hidden-layers 1 \
  --output results/research_ext/network.json
```

Then supply a rational polygon with the **same input dimension** as the network:

```sh
python scripts/research.py certify-polygon \
  --network results/research_ext/network.json \
  --domain PATH_TO_MATCHING_DIMENSION_POLYGON_JSON \
  --output results/research_ext/polygon_certificate.json
```

`configs/research_ext/polygon_square.json` is a **two-dimensional control**, not
an input polygon for the main 16-dimensional network. Supply 16-dimensional
vertices for the latter. The smoke validation includes a real 16D-domain example.
JSON rational coordinates/weights must be integers or strings such as `"1/3"`;
implicit decimal-to-exact conversions are rejected. Small-network budgets are
intentional; do not treat a refused/over-budget certificate as a negative result.

Box checks use `certify-box` with a domain object containing `lower` and `upper`.
A box crossing activation boundaries or lacking a full-column-rank affine map
is inconclusive, not a proof of failure.

## Lean source and its actual status

```sh
python scripts/research.py export-lean \
  results/research_ext/exact_controls/identity.json
python scripts/research.py check-formal --output results/research_ext/formal
```

The project pins Lean `v4.19.0` and uses Std only. Install that toolchain through
Lean's official installation instructions before running the check. There was
no Lean executable in the delivery environment. **The supplied source was not
compiled here; `validation/formal/formal_status.json` explicitly records this.**
The command returns nonzero without a successful build and named-theorem axiom
audit. Even a successful build checks the supplied abstract graph and logical
lemmas, not the full Python network-to-graph construction or smooth torus.

## What is included

`research_ext/` contains artifact auditing, exact-step/common-threshold analysis,
paired seed bootstrap, descriptive model comparison, observed-event brackets,
noise diagnostics, exact polygon and fixed-sign box checks, checkpoint export,
Lean export/check commands and sequential workflow orchestration.

`tests/research_ext/` contains unit, property/control, rejection and workflow tests.
`formal/` contains the separate Lean sources. `validation/` in the downloadable
bundle contains actual local test logs, tiny-run checkpoints/metrics, repeated-run
comparisons, exact certificates and the explicit unverified Lean status. Validation
artifacts are **not installed into or substituted for your production results**.

Read `docs/RESEARCH_EXTENSION_VERIFICATION.md` for precise trust boundaries and
`docs/RESEARCH_EXTENSION_REMAINING.md` for the outstanding empirical/formal work.
The package intentionally does not claim perfection, full experiment completion,
publication readiness or a proven phase transition.
