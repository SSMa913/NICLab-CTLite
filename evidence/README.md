# Experiment evidence

- `original_runs/<dataset>/seed_<seed>.json`: 20 original run files, copied byte-for-byte. These contain the selected configuration, validation history, selected epoch, and recorded training duration. HumanActivity originated in the capacity study, PhysioNet in the raw-Adam study, and USHCN/MIMIC in the constant-learning-rate study. Historical candidate IDs are retained as provenance, not exposed as alternative release configurations.
- `reference_results.json`: the selected runs' full-precision test MSE/MAE and five-seed means/sample standard deviations, extracted from the final locked experiment summary. It also includes test-cache hashes and validation rechecks.
- `original_file_manifest.json`: original relative paths and SHA-256 values linking the run files and selected checkpoints. Hashes establish file identity, not an independent timestamped attestation.
- `checkpoints/`: the 15 original HumanActivity, USHCN, and PhysioNet checkpoints. Each has a tensor `state_dict`. MIMIC weights are omitted from this public release.
- `checkpoint_recheck.json`: release-time checks of model initialization, forward results, gradients, and all 20 checkpoint test evaluations against the original implementation. This is a new audit, not an original training log.
- `release_validation.json`: results of fresh five-seed training with the standalone release, compared with the original selected runs.
- `table1_baselines.csv`: the 48 imported baseline/dataset rows, with their individual source labels. These are reported aggregate statistics, not new baseline training logs.

Verify included file hashes and recompute reference means/standard deviations:

```bash
python -m ctlite.verify
```

Re-evaluate all 15 included checkpoints:

```bash
python -m ctlite.verify --evaluate
```

After completing the four training scripts, compare the new test results with the reference:

```bash
python -m ctlite.verify --results outputs
```

The comparison tolerance is an absolute 2e-6 per metric. This is a replication check under the documented environment, not a significance test. CUDA/library changes may affect rounding and early stopping. Fresh output folders contain their own per-epoch validation history, environment, code/data hashes, selected checkpoint, and test metrics.

Aggregate MIMIC logs contain no patient-level data. Original research files are retained in the private workspace, outside the release. Intermediate searches, superseded checkpoints, model variants, per-record predictions, and unrelated experiments are intentionally excluded.
