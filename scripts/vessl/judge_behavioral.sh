#!/usr/bin/env bash
# GPU job (2 GPUs): one judge scores the three models' open-ended responses.
# JUDGE=gemma (default) is Gemma-4-31B; JUDGE=qwen is Qwen-3.6-27B.
#
# The judge call is scripts/behavioral_evaluation/judge.sbatch's, with the same
# pinned snapshot, prompt and decoding settings, so these judgments are the
# ones that job would have written. To continue judgments started elsewhere,
# upload them to the sync volume under behavioral_evaluation/unconstrained/
# judgments/<model>/<judge>.jsonl first. Judging appends batch by batch and
# resumes from what is already there; progress is published to the sync volume
# every ten minutes, so a job that stops early loses at most that much.
#
# RETRIES: the judge decodes greedily with a fixed seed, so retrying an answer
# it could not parse repeats the same output. Use RETRIES=0 for the main pass
# (unparsed items stay pending) and a later pass with a larger MAX_TOKENS for
# what is left.
source "$(dirname "$0")/common.sh"

case "${JUDGE:-gemma}" in
  gemma)
    JUDGE_REPO="google/gemma-4-31B-it"
    JUDGE_REVISION="842da3794eaa0b77d5f08bae87a17459d91ff475"
    JUDGE_SLUG=gemma-4-31b
    EXTRA=(--language-model-only --disable-custom-all-reduce --enforce-eager)
    ;;
  qwen)
    JUDGE_REPO="Qwen/Qwen3.6-27B"
    JUDGE_REVISION="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    JUDGE_SLUG=qwen-3.6-27b
    EXTRA=(--trust-remote-code --no-think --language-model-only
           --disable-custom-all-reduce --enforce-eager)
    ;;
  *) echo "JUDGE must be gemma or qwen" >&2; exit 2 ;;
esac
JUDGE_DIR="$HF_HOME/hub/models--${JUDGE_REPO/\//--}/snapshots/$JUDGE_REVISION"
ROOT="$SYNC/behavioral_evaluation/unconstrained"
OUT="$WORK/behavioral_evaluation/unconstrained/judgments"
DEST="$ROOT/judgments"
RECORDS="$SYNC/activations/data/records.json"
TP="${TP:-2}"
BATCH="${BATCH:-256}"
MODELS=(llama-3.1-8b gemma-3-12b qwen-3.5-27b)

install_code
responses=()
for slug in "${MODELS[@]}"; do
  [[ -s "$ROOT/responses/$slug.jsonl" ]] || { log "missing responses for $slug"; exit 1; }
  responses+=(--response "$slug=$ROOT/responses/$slug.jsonl")
done
[[ -s "$RECORDS" ]] || { log "no records at $RECORDS"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

log "caching $JUDGE_REPO@$JUDGE_REVISION"
python3 - "$JUDGE_REPO" "$JUDGE_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
[[ -f "$JUDGE_DIR/config.json" ]] || { log "judge snapshot is missing at $JUDGE_DIR"; exit 1; }
du -sh "$HF_HOME" | tail -1

# A fresh work volume resumes from the judgments an earlier job published.
mkdir -p "$OUT" "$DEST" "$SYNC/done"
while IFS= read -r -d '' file; do
  [[ -e "$OUT/$file" ]] && continue
  mkdir -p "$OUT/$(dirname "$file")"
  cp "$DEST/$file" "$OUT/$file"
  log "resuming from published $file ($(wc -l < "$OUT/$file") rows)"
done < <(cd "$DEST" && find . -type f -name '*.jsonl' -print0)

( while sleep 600; do publish "$OUT" "$DEST" >/dev/null 2>&1 || true; done ) &
PUBLISHER=$!
trap 'kill "$PUBLISHER" 2>/dev/null || true; publish "$OUT" "$DEST" || true' EXIT

python3 -u -m behavioral_evaluation.judge \
  --model "$JUDGE_DIR" --judge-slug "$JUDGE_SLUG" \
  --dataset "$RECORDS" "${responses[@]}" \
  --output-root "$OUT" \
  --tensor-parallel-size "$TP" --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --batch-size "$BATCH" \
  --max-model-len 8192 --max-tokens "${MAX_TOKENS:-512}" --retries "${RETRIES:-2}" --seed 42 \
  --progress-every "${PROGRESS_EVERY:-1}" \
  "${EXTRA[@]}"

publish "$OUT" "$DEST"
wc -l "$OUT"/*/"$JUDGE_SLUG".jsonl
date -u +%FT%TZ > "$SYNC/done/JUDGE_$JUDGE_SLUG"
log "$JUDGE_SLUG JUDGING DONE"
