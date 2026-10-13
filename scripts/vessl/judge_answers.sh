#!/usr/bin/env bash
# GPU job (2 GPUs): one judge scores the answers written under attention
# knockout (mechanistic.sh, stage "answers"). JUDGE=gemma or qwen.
#
# The judge call is the behavioural evaluation's, with the same pinned
# snapshot, prompt and decoding settings; only the answers differ. Each arm is
# judged as its own response set, named <model>.<arm>. Mount the volume that
# already holds the judge's weights as /work.
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
ANSWERS="$SYNC/mechanistic-study/answers"
OUT="$WORK/mechanistic-study/judgments"
DEST="$SYNC/mechanistic-study/judgments"
RECORDS="$SYNC/activations/data/records.json"

install_code
responses=()
while IFS= read -r file; do
  [[ -s "$file" ]] || continue
  model=$(basename "$(dirname "$file")"); arm=$(basename "$file" .jsonl)
  responses+=(--response "$model.$arm=$file")
done < <(find "$ANSWERS" -name '*.jsonl' ! -name '*.part' | sort)
(( ${#responses[@]} )) || { log "no answers under $ANSWERS"; exit 1; }
log "judging $(( ${#responses[@]} / 2 )) answer sets with $JUDGE_SLUG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 - "$JUDGE_REPO" "$JUDGE_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
[[ -f "$JUDGE_DIR/config.json" ]] || { log "judge snapshot is missing at $JUDGE_DIR"; exit 1; }

mkdir -p "$OUT" "$DEST"
# Continue from judgments an earlier job published for this judge.
while IFS= read -r -d '' file; do
  [[ -e "$OUT/$file" ]] && continue
  mkdir -p "$OUT/$(dirname "$file")"
  cp "$DEST/$file" "$OUT/$file"
done < <(cd "$DEST" && find . -type f -name "$JUDGE_SLUG.jsonl" -print0)
trap 'publish "$OUT" "$DEST" || true' EXIT

# Greedy decoding with a fixed seed repeats itself, so a failed item is not
# retried; MAX_TOKENS is generous instead (truncated verdicts were the failures).
python3 -u -m behavioral_evaluation.judge \
  --model "$JUDGE_DIR" --judge-slug "$JUDGE_SLUG" \
  --dataset "$RECORDS" "${responses[@]}" --output-root "$OUT" \
  --tensor-parallel-size "${TP:-2}" --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --batch-size "${BATCH:-512}" \
  --max-model-len 8192 --max-tokens "${MAX_TOKENS:-1536}" --retries 0 --seed 42 \
  --progress-every 1 "${EXTRA[@]}" || log "some answers could not be judged"
publish "$OUT" "$DEST"
date -u +%FT%TZ > "$SYNC/done/JUDGE_ANSWERS_$JUDGE_SLUG"
log "$JUDGE_SLUG ANSWER JUDGING DONE"
