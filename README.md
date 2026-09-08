# Feature topology

A computational study of whether changing representation geometry precedes
compression or loss of a task-irrelevant factor. **Research results must come from
completed runs; this repository is not evidence that a hypothesized transition exists.**

Read [the protocol](docs/PROTOCOL.md) for assumptions and limitations, and
[the original brief](docs/original_brief.txt) for the requested scope.

## Environment

Python 3.11+ and the dependencies in `requirements.lock.txt`. On this workspace,
the installed interpreter is `../../work/venv/bin/python` relative to this repository.
For a portable setup:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest -q
```

The lock captures the actual macOS ARM environment; other platforms may need
platform-appropriate PyTorch wheels. All computation uses actual PyTorch models.

## Staged execution

Commands below run from the repository directory with the environment activated.

```sh
python scripts/calibrate.py --config configs/synthetic/calibration_pilot.json --output results/calibration_pilot
python scripts/run_pilot.py --calibration results/calibration_pilot/gamma_to_lr.json --output results/pilot
python scripts/compute_metrics.py --runs results/pilot/runs --ntk-points 32 --jacobian-points 1000 --ph-size 150 --ph-repeats 3 --ph-maxdim 1
python scripts/summarize.py --root results/pilot
python scripts/run_controls.py
```

Those reduced PH/NTK/Jacobian settings are **pilot diagnostics**, not the full
production protocol. Production defaults in `compute_metrics.py` use 10,000
Jacobian points, 128 NTK points, 500 PH points, 20 repeats, and H2. Use
`--all-checkpoints` to analyze the full trajectory.

After inspecting matched-risk pilot results and recording an evidence-backed gate:

```sh
python scripts/calibrate.py --config configs/synthetic/calibration_main.json --output results/calibration_main --trials results/calibration_pilot/trials
python scripts/run_main_sweep.py --calibration results/calibration_main/gamma_to_lr.json --output results/main --pilot-gate results/pilot/gate.json
python scripts/compute_metrics.py --runs results/main/runs --all-checkpoints
python scripts/run_image_validation.py --output results/rotated_digits
```

Completed calibration trials and runs are cached by config fingerprint. The
shared calibration trial path avoids repeating the five pilot gamma calibrations.
Training checkpoints include optimizer and minibatch RNG state to resume interrupted
work. Completed runs are not rerun. Changed configurations intentionally identify
different experiments; do not change settings merely to restart a failed run.

## Representations and storage

All hidden activations are reproducible from saved weights and fixed grid. To
materialize them without reducing precision:

```sh
python scripts/extract_representations.py --run results/pilot/runs/RUN_ID --output /path/to/storage/RUN_ID
```

This writes float32 compressed archives and retains a 2 GiB free-space reserve.
The full primary trajectory collection is approximately 100 GB uncompressed;
the width/depth extensions increase that substantially. Metrics can be computed
checkpoint by checkpoint without retaining this entire activation collection in RAM.

## Contents

- `src/data`: exact torus/cylinder, periodic rotations of real digits, dSprites loader.
- `src/models`: plain MLP, CNN, and CIFAR-compatible ResNet18.
- `src/training`: centered output scaling, vanilla SGD, frozen LR calibration support.
- `src/metrics`: geometry, tangent Jacobians, sampled global margins, collisions,
  held-out probes, PH, empirical NTK, nulls, polygon image-graph quotient.
- `src/analysis`: seed bootstrap confidence intervals and segmented/smooth comparisons.
- `configs/sweeps`: requested main, width, depth, relevance, and control designs.
- `tests`: analytic invariants, known quotient controls, leakage checks, and resume behavior.

The real-image validation uses handwritten digits available offline through
scikit-learn. It is a small realistic controlled-rotation experiment, not a claim
to have run dSprites or CIFAR. The dSprites loader and CIFAR model/configurations
alone do not constitute completed image experiments.
