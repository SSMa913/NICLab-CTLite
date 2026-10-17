#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MIMIC_DATA_DIR:?Set MIMIC_DATA_DIR to your authorized external directory containing train.npz, val.npz, test.npz}"
"${PYTHON:-python}" -m ctlite.train --config configs/MIMIC.json --data-dir "$MIMIC_DATA_DIR" "$@"
