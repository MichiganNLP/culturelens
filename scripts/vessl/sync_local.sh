#!/usr/bin/env bash
# Runs on your machine: pulls pipeline results down as each stage finishes.
#
#   bash scripts/vessl/sync_local.sh <job-slug> [sync-volume-slug]
#
# Every stage the GPU job completes leaves a marker under done/ on the sync
# volume. This polls for new markers and, when one appears, mirrors the whole
# benchmark tree into data/interim/benchmark -- the layout every stage reads
# from, so a run can be picked up elsewhere (for example another cluster) from exactly
# where it stopped. It also pulls a partial snapshot every PARTIAL_EVERY polls,
# so batch checkpoints from an unfinished stage are not stranded in the cloud.
set -euo pipefail

JOB="${1:?usage: sync_local.sh <job-slug> [sync-volume-slug]}"
VOL="${2:?Pass the VESSL object volume id}"
POLL="${POLL:-300}"
PARTIAL_EVERY="${PARTIAL_EVERY:-6}"
VESSL="${VESSL:-$HOME/.local/bin/vesslctl}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/interim/benchmark"
META="$ROOT/data/interim/.vessl"
mkdir -p "$DEST" "$META/done" "$META/logs"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
# A remote object can be truncated if the job that wrote it was torn down
# mid-copy. Everything is downloaded to a staging area first, and a JSON or
# JSONL file replaces the local copy only if every line of it parses; a bad one
# is reported and the previous local copy is kept.
valid() {
  python3 - "$1" 2>/dev/null <<'CHECK'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    if path.endswith(".jsonl"):
        for line in handle:
            if line.strip():
                json.loads(line)
    else:
        json.load(handle)
CHECK
}

pull() {
  local spec prefix dest stage f rel bad=0 moved=0
  for spec in "benchmark/:$DEST" \
              "processed/culturelens/:$ROOT/data/processed/culturelens" \
              "gold_responses_v2/:$ROOT/data/interim/gold_responses_v2" \
              "logs/:$META/logs"; do
    prefix="${spec%%:*}"; dest="${spec#*:}"; stage="$META/staging/${prefix%/}"
    rm -rf "$stage"; mkdir -p "$stage"
    "$VESSL" volume download "$VOL" "$stage" --remote-prefix "$prefix" --overwrite \
      --exclude '*.tmp' --exclude '*.part' >/dev/null 2>&1 || continue
    while IFS= read -r -d '' f; do
      rel="${f#"$stage"/}"
      if [[ "$f" == *.json || "$f" == *.jsonl ]] && ! valid "$f"; then
        log "SKIP truncated remote file $prefix$rel (kept the local copy)"
        bad=$((bad + 1)); continue
      fi
      mkdir -p "$dest/$(dirname "$rel")"; mv -f "$f" "$dest/$rel"; moved=$((moved + 1))
    done < <(find "$stage" -type f -print0)
  done
  rm -rf "$META/staging"
  log "synced $moved file(s)$( (( bad )) && echo ", skipped $bad truncated")"
}

polls=0
while true; do
  before=$(ls "$META/done" 2>/dev/null | sort)
  "$VESSL" volume download "$VOL" "$META/done" --remote-prefix done/ --overwrite >/dev/null 2>&1 || true
  after=$(ls "$META/done" 2>/dev/null | sort)
  new=$(comm -13 <(echo "$before") <(echo "$after") | grep -v '^$' || true)
  polls=$((polls + 1))

  if [[ -n "$new" ]]; then
    log "stage(s) done: $(echo "$new" | tr '\n' ' ')"
    pull
  elif (( polls % PARTIAL_EVERY == 0 )); then
    log "partial snapshot"; pull
  fi

  state=$("$VESSL" job show "$JOB" -o json 2>/dev/null |
          python3 -c "import sys,json; print(str(json.load(sys.stdin).get('workloadState','?')).lower())" 2>/dev/null || echo "?")
  if [[ -f "$META/done/${UNTIL:-ALL}" ]]; then log "${UNTIL:-ALL} complete"; pull; exit 0; fi
  case "$state" in
    succeeded|failed|terminated) log "job $state -- final pull"; pull; exit 0 ;;
  esac
  sleep "$POLL"
done
