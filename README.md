# CTLite: Continuous-Time Lightweight Forecasting

A compact implementation of CTLite and the four-dataset accuracy comparison in Table I. The release contains the full CTLite model, fixed experiment configurations, five-seed training scripts, public benchmark splits, and archived experiment evidence. It does not include hyperparameter searches, ablations, visualization code, or efficiency experiments.

## Installation

The reference experiments used Python 3.10.15, NumPy 2.1.3, PyTorch 2.5.1 with CUDA 12.4, and an NVIDIA RTX 3090. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Other devices can run the implementation, but floating-point differences can change early stopping. Use `--device cpu` for a CPU run. Exact replication should use the reference CUDA environment. No installation of the original research framework is needed.

Release validation reran all 20 experiments from scratch on the reference environment. Test MSE/MAE, selected checkpoint tensors, and every epoch's validation metrics matched the archived experiments exactly. Details are in `evidence/release_validation.json`.

## Reproduce Table I

Each script runs seeds **2024, 2025, 2026, 2027, 2028**, trains from scratch, selects the lowest validation MSE, and evaluates that single checkpoint on the test split.

```bash
bash scripts/run_humanactivity.sh
bash scripts/run_ushcn.sh
bash scripts/run_physionet.sh
```

These three datasets are included as compact, lossless, preprocessed benchmark splits. PhysioNet refers to the **repaired P12 version** used in the manuscript, not the old P12 cache. See [data/README.md](data/README.md) for provenance and licenses.

MIMIC-III requires individual authorized access. Its data and trained checkpoints are **not included**. See [the private-data preparation instructions](data/README.md#mimic-iii-private-preparation), then run:

```bash
export MIMIC_DATA_DIR=/absolute/path/to/your/private/MIMIC
bash scripts/run_mimic.sh
```

Alternatively, after preparing MIMIC, run all four datasets:

```bash
export MIMIC_DATA_DIR=/absolute/path/to/your/private/MIMIC
bash scripts/run_all.sh
```

`PYTHON=/path/to/python` selects another interpreter. Individual scripts accept `--output /path/to/runs`, `--seeds 2024`, and `--resume`. Defaults reproduce all five seeds. Existing runs are not silently overwritten. A resumed run must match the configuration, data, code, and environment recorded in its result file.

Generate the comparison table from newly completed runs:

```bash
python -m ctlite.table1 --results outputs --output outputs/table1.csv
```

To reconstruct the manuscript table directly from archived CTLite results:

```bash
python -m ctlite.table1 --output outputs/table1_archived.csv
```

Both commands also write a Markdown table. The 12 baseline rows are imported reference results, **not models trained by these scripts**. HumanActivity, USHCN, and MIMIC baseline values come from Table 2 of *Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline* ([paper](https://arxiv.org/abs/2505.11250), [code](https://github.com/decisionintelligence/APN)). PhysioNet baseline values come from the supplied repaired-P12 replication summary. Its per-seed runs and full preprocessing hashes were not supplied, so the release does not assign our five-seed protocol to those rows or claim an independently verified rerun. Every baseline row records its source in `evidence/table1_baselines.csv`.

## Fixed configurations and reference results

All datasets use raw Adam weights, a constant learning rate, observed-entry MSE, gradient clipping at 1, a 200-epoch limit, and patience 30. There is no EMA, SWA, auxiliary loss, scheduler, or prediction ensemble. Rates use the normalized time coordinates in the data.

| Dataset | Parameters | Rank | MLP width | Decay rates | Learning rate | Batch size | Weight decay |
|---|---:|---:|---:|---|---:|---:|---:|
| HumanActivity | 365 | 1 | 8 | 0, 1, 4, 16, 64 | 0.01 | 16 | 0.0001 |
| USHCN | 88 | 1 | 2 | 0, 8 | 0.0003 | 4 | 0.01 |
| PhysioNet | 889 | 4 | 4 | 0, 2, 8, 32 | 0.001 | 64 | 0.0003 |
| MIMIC | 2,537 | 4 | 8 | 0, 2, 8, 32, 128 | 0.001 | 32 | 0.0001 |

| Dataset | MSE (mean ± sample SD) | MAE (mean ± sample SD) |
|---|---:|---:|
| HumanActivity | 0.04256 ± 0.00098 | 0.11672 ± 0.00051 |
| USHCN | 0.16482 ± 0.00084 | 0.27068 ± 0.00241 |
| PhysioNet | 0.28762 ± 0.00017 | 0.35265 ± 0.00096 |
| MIMIC | 0.40476 ± 0.00083 | 0.38438 ± 0.00087 |

Metrics pool all observed target scalars in each split, rather than averaging per-record or per-batch errors. The standard deviation uses `ddof=1` over five independent training seeds on the same fixed split. Training and validation reuse the parameter-free historical summaries. Test evaluation executes the full model. Split-wide padding, float32 arithmetic, encoding batches of 256, and evaluation batches of 128 match the archived procedure. TF32 matrix multiplication is disabled.

These configurations were selected during the preceding research process. This release fixes them for replication and does not imply that no earlier tuning occurred.

## Model interface

```python
import torch
from ctlite import CTLite

model = CTLite(channels=5, rank=1, width=2, rates=[0., 8.])
x = torch.randn(2, 10, 5)
mask = torch.ones_like(x)
history_times = torch.linspace(0., 0.7, 10)[None, :, None].expand(2, -1, -1)
query_times = torch.tensor([0.8, 0.9])[None, :, None].expand(2, -1, -1)
prediction = model(x, history_times, mask, query_times)  # [2, 2, 5]
```

`encode` constructs causal temporal moments and reliability features. `decode` applies the bounded local readout and low-rank cross-variable correction. The model never receives future target values as predictive inputs.

## Repository contents

```text
ctlite/       Model, data loader, training, Table I assembly, private MIMIC preparation
configs/      Four fixed paper configurations
scripts/      Four dataset scripts and a combined runner
data/         Three public datasets, format/provenance documentation, hashes
evidence/     Original epoch histories, per-seed results, public-data checkpoints, audits
tests/        Masking, summary reuse, target isolation, and metric checks
```

See [evidence/README.md](evidence/README.md) for verification commands and the distinction between original experiment records and release validation. Generated outputs are ignored by Git. Keep MIMIC inputs and any MIMIC model artifacts in your authorized private environment.

Run the small behavior checks with `python -m unittest discover -s tests -v`. The included GitHub Actions workflow runs these checks and verifies the archived file hashes on CPU.

## License and acknowledgments

The standalone CTLite software is released under the [MIT License](LICENSE). Dataset licenses are separate and are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The original experiments used the APN research framework and benchmark inputs, with public data derived from UCI, NOAA/USHCN, and PhysioNet. This release does not vendor the APN, GraFITi, or tsdm implementations.

Please cite the CTLite manuscript associated with this repository and the original datasets when using the code or data. A publication identifier and author citation will be added when the manuscript's bibliographic record is finalized.
