#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Check before starting any training, so a four-dataset run cannot silently omit MIMIC.
: "${MIMIC_DATA_DIR:?Set MIMIC_DATA_DIR to the authorized external MIMIC split directory}"
for dataset in humanactivity ushcn physionet mimic; do
    bash "scripts/run_${dataset}.sh" "$@"
done
echo 'Training complete. Assemble Table I with: python -m ctlite.table1 --results <output-directory>'
