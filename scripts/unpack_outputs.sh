#!/usr/bin/env bash
# The six files of raw model answers are stored gzipped to stay under GitHub's
# file-size limit. This restores them in place; the analysis code reads .jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
find output -name '*.jsonl.gz' -print0 | while IFS= read -r -d '' file; do
  [[ -f "${file%.gz}" ]] || { gunzip -k "$file"; echo "unpacked ${file%.gz}"; }
done
