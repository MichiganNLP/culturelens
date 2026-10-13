#!/usr/bin/env bash
# Runs on your machine: publish the code tree to the sync volume.
#
#   bash scripts/vessl/upload_code.sh [sync-volume-slug]
#
# The volume cannot delete individual files, so a file removed here would
# otherwise linger there and be copied into every later job. MANIFEST lists
# exactly what this upload contains, and jobs install only those files.
# Pipeline state (data/interim, data/processed/culturelens) is never uploaded.
set -euo pipefail
VOL="${1:?Pass the VESSL object volume id}"
VESSL="${VESSL:-$HOME/.local/bin/vesslctl}"
cd "$(dirname "$0")/../.."

find src prompts scripts configs tests data/raw data/processed/references \
     pyproject.toml README.md -type f \
     ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '.DS_Store' | sort > MANIFEST
echo "MANIFEST: $(wc -l < MANIFEST | tr -d ' ') files"
put() {  # local-file remote-prefix -- retried: the API returns transient 5xx
  local attempt
  for attempt in 1 2 3 4 5; do
    "$VESSL" volume upload "$VOL" "$1" --remote-prefix "$2" --overwrite >/dev/null 2>&1 && return 0
    sleep $((attempt * 3))
  done
  echo "upload failed: $1" >&2; return 1
}
# dirname of a root-level file is ".", and "code/./" makes the volume store it
# under a literal "./" directory where install_code cannot find it by its
# MANIFEST path -- the job then dies on the first such file.
while IFS= read -r f; do
  d=$(dirname "$f")
  if [[ "$d" == "." ]]; then put "$f" "code/"; else put "$f" "code/$d/"; fi
done < MANIFEST
# Last, so a job never sees a MANIFEST naming files that have not arrived.
put MANIFEST code/
echo "uploaded $(wc -l < MANIFEST | tr -d ' ') files + MANIFEST to $VOL:code/"
