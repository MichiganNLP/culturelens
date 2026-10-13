#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

"$PYTHON" -m preprocessing.build_wvs_backbone "${OVERWRITE_ARGS[@]}"
"$PYTHON" -m preprocessing.build_countries_config "${OVERWRITE_ARGS[@]}"
