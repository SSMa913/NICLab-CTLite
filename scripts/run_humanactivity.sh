#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python}" -m ctlite.train --config configs/HumanActivity.json "$@"
