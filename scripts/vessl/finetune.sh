#!/usr/bin/env bash
# GPU job: one GRPO / Critique-GRPO run on Llama-3.1-8B-Instruct.
#
#   REWARD=critique CF=0|1 STEPS=... [EXTRA="--flag value ..."]
#
# The run directory lives on the cluster volume and is mirrored to the sync
# volume every PUBLISH_EVERY seconds and on exit, so checkpoints survive a job
# that dies.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/finetune_common.sh"

REWARD="${REWARD:?set REWARD}"
CF="${CF:-0}"
STEPS="${STEPS:-300}"
EXTRA="${EXTRA:-}"
PUBLISH_EVERY="${PUBLISH_EVERY:-600}"

install_code
install_peft
verify_llama || { log "llama is not assembled/verified; run finetune_prefetch.sh"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cf_flag=(); [[ "$CF" == 1 ]] && cf_flag=(--counterfactual)
BASE_METHOD=$(python3 -c "from finetune import method_name; print(method_name('$REWARD', $([[ $CF == 1 ]] && echo True || echo False)))")
# A smoke or ablation run gets its own name so it never claims the real run's directory.
METHOD="$BASE_METHOD${METHOD_SUFFIX:+-$METHOD_SUFFIX}"
RUN="$WORK/finetune/runs/$METHOD"
DEST="$SYNC/finetune/runs/$METHOD"
log "method=$METHOD steps=$STEPS run=$RUN extra=[$EXTRA]"
if [[ -d "$RUN" ]] && [[ -n "$(ls -A "$RUN")" ]]; then
  log "$RUN already exists; refusing to overwrite a run"; exit 1
fi

mkdir -p "$WORK/finetune"   # tee opens the log before python creates the run directory
( while sleep "$PUBLISH_EVERY"; do mirror "$RUN" "$DEST" || true; done ) &
PUBLISHER=$!
trap 'kill $PUBLISHER 2>/dev/null; mirror "$RUN" "$DEST" || true' EXIT

# shellcheck disable=SC2086
python3 -m finetune.grpo --model "$LLAMA_DIR" --reward "$REWARD" "${cf_flag[@]}" \
  --train "$SPLITS/train.json" --validation "$SPLITS/validation.json" \
  --steps "$STEPS" --output-dir "$RUN" --method "$METHOD" $EXTRA 2>&1 | tee "$WORK/finetune/$METHOD.log"
status=${PIPESTATUS[0]}
mirror "$RUN" "$DEST"
mkdir -p "$SYNC/logs"; cp -f "$WORK/finetune/$METHOD.log" "$SYNC/logs/" 2>/dev/null || true
[[ $status == 0 ]] || { log "TRAINING FAILED ($status)"; exit "$status"; }
mkdir -p "$SYNC/done"; date -u +%FT%TZ > "$SYNC/done/finetune.$METHOD"
log "DONE $METHOD"
