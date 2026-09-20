# Full ablation metrics

After a synthetic sweep finishes, prepare its verified training inputs:

```sh
python scripts/prepare_ablation_inputs.py --repo PATH_TO_RESULTS_REPO --study width --output PATH_TO_INPUT_DIRECTORY
```

Supported studies are width, depth, relevance, swapped, cylinder, small_network,
and OOD nuisance shift. Preparation refuses incomplete training coverage. Diverged
runs are accounted for separately and do not receive nonexistent final-checkpoint
metrics. Shared baseline checkpoints are packaged once within a study. OOD
conditions are packaged separately because they require different fixed analysis
arrays.

The output contains a study_index.json and one directory per data-condition fingerprint. Each condition directory contains input_index.json, analysis_data.npz, and checkpoint archives. Conditions include dimension, manifold, factor swapping, relevance, and relevance mode. Different widths and depths can share the same analysis data; different data conditions cannot.

For each condition, place its index at a distinct committed path and select that path using the Production metrics batch workflow's index_path input. Use a separate training-input release for that condition, containing its archives and analysis_data.npz. The default index and tags still select the primary study. Public artifact publication requires the pending explicit approval; generating local inputs does not publish them.

Select at most 24 run IDs per dispatch. The worker verifies the archive and array hashes, computes the full production profile at every checkpoint, and preserves compatible partial work. Returned worker archives use main/runs/RUN_ID internally for every study; match RUN_ID and checkpoint hashes to the original sweep locations when collecting results. Do not interpret that internal staging path as primary-study membership.

Input preparation and successful job dispatch do not establish metric completion. Verify all returned checkpoint and layer coverage using the production catalog and final freeze checks. Cross-study baseline reuse is valid only when checkpoint bytes, analysis data, metric options, and execution provenance match.
