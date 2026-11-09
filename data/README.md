# Data and benchmark provenance

The included files are **processed forecasting splits**, not raw source archives. They preserve the exact six numerical fields consumed by the final CTLite experiments. There is no resplitting, re-normalization, resampling, or rounding at release time. Record identifiers and unused metadata were removed. `manifest.json` records the source-cache SHA-256, released NPZ SHA-256, record count, observed-target count, and padded tensor shapes. Public source licenses and attribution are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Dataset | Variables | Train / validation / test records | Train / validation / test observed scalar targets |
|---|---:|---|---|
| HumanActivity | 12 | 949 / 193 / 218 | 25,872 / 5,205 / 5,910 |
| USHCN | 5 | 901 / 101 / 112 | 2,964 / 334 / 368 |
| PhysioNet | 36 | 9,703 / 1,079 / 1,199 | 217,124 / 24,061 / 27,008 |
| MIMIC (not distributed) | 96 | 17,212 / 1,913 / 2,125 | 187,419 / 20,874 / 23,015 |

The public NPZ files contain concatenated float32 arrays `x`, `x_mask`, `x_mark`, `y`, `y_mask`, `y_mark`, together with an int64 `<field>_offsets` array for each field. Record `i` is `values[offsets[i]:offsets[i+1]]`. Time arrays have one column. All files load with `numpy.load(..., allow_pickle=False)`. The loader pads each complete split to match the archived arithmetic. Masks identify observed values. The five seeds change model initialization and training order, not these fixed partitions.

## HumanActivity

Derived from **Localization Data for Person Activity**, by Vedrana Vidulin, Mitja Lustrek, Bostjan Kaluza, Rok Piltaver, and Jana Krivec (2010), UCI Machine Learning Repository, [DOI: 10.24432/C57G8X](https://doi.org/10.24432/C57G8X). This is the four-tag localization dataset, not the smartphone HAR dataset. The source is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The release distributes the transformed 12-channel forecasting inputs, with the existing benchmark's normalization, windows, and split preserved. Retain attribution when redistributing.

## USHCN

Derived from NOAA's [U.S. Historical Climatology Network](https://www.ncei.noaa.gov/products/land-based-station/us-historical-climatology-network), using the five-variable sporadic subset distributed with [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes/tree/master/gru_ode_bayes/datasets/Climate). The source subset is `small_chunked_sporadic.csv`. Credit NOAA/NCEI and Edward De Brouwer, Jaak Simm, Adam Arany, and Yves Moreau, *GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series*, NeurIPS 2019. NOAA-produced environmental observations are public-domain data in the United States under the [NCEI Open Data Policy](https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf). The upstream repository's MIT notice is retained in `data/USHCN/NOTICE.md`.

## PhysioNet 2012 (repaired P12)

Contains information from **Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012**, version 1.0.0, by Ikaro Silva, George Moody, Roger Mark, and Leo Anthony Celi, available under the [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/). [Source and access policy](https://physionet.org/content/challenge-2012/1.0.0/). This particular PhysioNet dataset is open access and does not require MIMIC credentialing.

The forecasting benchmark is a derived database under ODC-By, using 36 longitudinal variables. It uses the corrected historical/forecast windows and observation masks stored in the supplied repaired caches. The old `exp33_cross_backbone/data/P12` cache has different cutoffs and target membership and is intentionally excluded. The repaired train/validation and test source paths are recorded separately in `manifest.json`. Do not combine the old and repaired versions. This packaging step preserves the repaired inputs exactly and does not claim to independently reconstruct the earlier repair from raw clinical text.

## MIMIC-III private preparation

[MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) requires credentialed access, approved training, and a signed data-use agreement. See the [PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). No MIMIC records, patient identifiers, per-record predictions, or trained MIMIC weights are included here. Only aggregate training/validation/test statistics and file hashes are published.

After obtaining authorization, prepare the De Brouwer-style `complete_tensor.csv` with the [upstream MIMIC preprocessing workflow](https://github.com/edebrouwer/gru_ode_bayes/tree/master/data_preproc). The source used in these experiments contains the `UNIQUE_ID`, `TIME_STAMP`, `LABEL_CODE`, and `VALUENORM` fields and has SHA-256:

```text
42085d921d91ae380499e0fee04970dd4c29606ca8046dfd3a87ca457674d644
```

Upstream preprocessing versions can differ. This release starts from that authorized intermediate CSV, rather than providing an end-to-end extractor for the original hospital database tables. To generate the exact benchmark splits:

```bash
python -m pip install pandas==2.2.3
python -m ctlite.prepare_mimic \
  --csv /private/path/complete_tensor.csv \
  --destination /private/path/CTLite_MIMIC
export MIMIC_DATA_DIR=/private/path/CTLite_MIMIC
bash scripts/run_mimic.sh
```

The preparation script pivots the 96 normalized variables, orders records by patient key, uses the fixed 81%/9%/10% partition, divides timestamps by 96, uses a history cutoff of 71.5/96, and predicts the next three recorded timestamps. It preserves the normalization already in `VALUENORM`. The resulting three NPZ hashes were checked against the exact original benchmark caches. Compare your hashes to the MIMIC entries in `manifest.json` before claiming exact replication. This comparison requires no disclosure of patient identifiers.

If you already have an authorized copy of the exact cached splits, an alternative converter is available:

```bash
python -m ctlite.import_mimic \
  --source /private/path/cached_MIMIC_III \
  --destination /private/path/CTLite_MIMIC \
  --trust-source
```

The source directory must contain `train.pt`, `val.pt`, and `test.pt`. The trust flag is required because original research caches may contain pickle objects. Both conversion tools reject destinations inside the public repository. Keep the resulting data private. Data access authorization alone does not establish that an arbitrary MIMIC preprocessing variant matches the paper.
