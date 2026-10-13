#!/usr/bin/env bash
# GPU job (2 GPUs): evaluate models on both test splits and judge with Gemma.
#
#   MODELS="slug=adapter-dir-or-base ..."   e.g. "base=base critique-grpo-critique-country=/work/.../best"
#
# For each model: open-ended and constrained A/B answers under every monitor
# condition, then one Gemma-4-31B judge pass over all open answers. The judge
# is loaded once for every model in MODELS, so judging the base model beside a
# fine-tuned one costs one model load, not two.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/finetune_common.sh"

MODELS="${MODELS:?set MODELS}"
CONDITIONS="${CONDITIONS:-country_only country_value country_norm country_opposite_norm}"
EVAL="$WORK/finetune/eval"
DEST="$SYNC/finetune/eval"
install_code
verify_llama || { log "llama is not assembled/verified"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
trap 'mirror "$EVAL" "$DEST" || true' EXIT

responses=()
for spec in $MODELS; do
  slug="${spec%%=*}"; adapter="${spec#*=}"
  lora=(); [[ "$adapter" != base ]] && lora=(--lora "$adapter")
  for split in test_seen test_unseen; do
    out="$EVAL/$slug/$split"
    log "generate $slug $split (open)"
    python3 -m behavioral_evaluation.generate --model "$LLAMA_DIR" "${lora[@]}" \
      --input "$SPLITS/$split.json" --output "$out/unconstrained.jsonl" \
      --conditions $CONDITIONS --max-tokens 1024 --batch-size 64
    log "generate $slug $split (constrained)"
    python3 -m behavioral_evaluation.generate --model "$LLAMA_DIR" "${lora[@]}" --options \
      --answer-only-retry --input "$SPLITS/$split.json" --output "$out/constrained.jsonl" \
      --conditions $CONDITIONS --max-tokens 768 --batch-size 64
    responses+=(--response "$slug-$split=$out/unconstrained.jsonl")
  done
  mirror "$EVAL" "$DEST"
done

log "judge with $GEMMA_JUDGE"
python3 -m behavioral_evaluation.judge --model "$GEMMA_JUDGE" --judge-slug gemma-4-31b \
  --dataset "$SPLITS/records.json" "${responses[@]}" --output-root "$EVAL/judgments" \
  --tensor-parallel-size 2 --batch-size 64 --max-tokens 512

python3 -m finetune.summarize --eval-root "$EVAL" --judge-slug gemma-4-31b \
  --output "$EVAL/summary.json"
mirror "$EVAL" "$DEST"
mkdir -p "$SYNC/done"; date -u +%FT%TZ > "$SYNC/done/finetune_eval.$(echo $MODELS | md5sum | cut -c1-8)"
log "EVAL DONE"
