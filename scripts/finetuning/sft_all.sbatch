#!/usr/bin/env bash
# SFT on both tasks: open-ended and A/B examples, trained on country_only prompts.
#
# One job: fine-tune Llama-3.1-8B-Instruct, generate answers on both test splits
# with the best checkpoint, judge the open answers with Gemma-4-31B, and write a
# results table.
#
# Submit from the repository root, adding your own allocation:
#   sbatch --account=<account> --partition=<gpu-partition> scripts/finetuning/sft_all.sbatch
#
# Needs: torch 2.11, transformers 5.12, peft 0.21, accelerate 1.14, vllm 0.24,
# pydantic, pyyaml -- and `huggingface-cli login`, because Llama-3.1 is gated.
# Data must be at data/processed/culturelens/ (records.json,
# splits/{train,validation,test_seen,test_unseen}.json) and the gold answers at
# data/processed/culturelens/gold/ (train.country_only.jsonl;
# the matching validation.* files are used for selection when present).
#
# If the job is killed by its time limit, submit it again. Training restores its
# saved state, and generation and the judge skip what they already wrote.
#
# Every setting below can be overridden from the environment, e.g.
#   OUTPUT=/path/to/output EVAL_BASE=1 sbatch ... scripts/finetuning/sft_all.sbatch
#
#SBATCH --job-name=cl-sft-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail

CF=0                   # 1 adds country_opposite_norm, country_norm and country_value
ALL=1                  # 1 also trains on the A/B task for the same records and views

# ---------------------------------------------------------------- settings
# sbatch runs a spooled copy of this file, so the submit directory is the only
# reliable anchor for the repository: submit from its root, or set REPO.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${PYTHON:-python}"
POLICY_MODEL="${POLICY_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"   # HF id or local directory
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-31B-it}"
JUDGE_SLUG="${JUDGE_SLUG:-gemma-4-31b}"
DATA="${DATA:-$REPO/data/processed/culturelens}"
OUTPUT="${OUTPUT:-$REPO/output/finetune}"
GOLD="${GOLD:-$DATA/gold}"

# Training runs on one GPU; the second is for the judge, which needs two.
# An epoch is every example once: 5,984 for the country recipe, about 20,000
# with CF=1 (three views per record, plus the value view where there is one),
# and twice that with ALL=1 (one open-ended and one A/B example each).
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"               # examples per optimiser step
MICRO_BATCH="${MICRO_BATCH:-8}"              # lower this first on a smaller GPU
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_PER_VIEW="${EVAL_PER_VIEW:-200}"
STATE_EVERY="${STATE_EVERY:-50}"             # steps between resumable saves
TRAIN_EXTRA="${TRAIN_EXTRA:-}"               # any further `python -m finetune.sft` flags

# Evaluation.
CONDITIONS="${CONDITIONS:-country_only country_value country_norm country_opposite_norm}"
EVAL_BATCH="${EVAL_BATCH:-64}"
JUDGE_TP="${JUDGE_TP:-2}"                    # Gemma-4-31B needs two 48 GB GPUs
JUDGE_BATCH="${JUDGE_BATCH:-64}"
GENERATE_EXTRA="${GENERATE_EXTRA:-}"         # any further `behavioral_evaluation.generate` flags
JUDGE_EXTRA="${JUDGE_EXTRA:-}"               # any further `behavioral_evaluation.judge` flags
EVAL_BASE="${EVAL_BASE:-0}"                  # 1 also evaluates the untuned model, for comparison;
                                             # needed once, in any one of the jobs

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# ------------------------------------------------------------------ checks
[[ -d "$REPO/src/finetune" ]] || { echo "REPO=$REPO is not the repository root; submit from it or set REPO" >&2; exit 1; }
for f in train validation test_seen test_unseen; do
  [[ -s "$DATA/splits/$f.json" ]] || { echo "missing $DATA/splits/$f.json" >&2; exit 1; }
done
[[ -s "$DATA/records.json" ]] || { echo "missing $DATA/records.json" >&2; exit 1; }

[[ -s "$GOLD/train.country_only.jsonl" ]] || { echo "missing $GOLD/train.country_only.jsonl" >&2; exit 1; }
if [[ "$CF" == 1 ]]; then
  [[ -s "$GOLD/train.country_opposite_norm.jsonl" ]] || { echo "missing $GOLD/train.country_opposite_norm.jsonl" >&2; exit 1; }
fi

METHOD=$("$PYTHON" -c "from finetune.sft import method_name; print(method_name(bool($CF), bool($ALL)))")
RUN="$OUTPUT/runs/$METHOD"
EVAL="$OUTPUT/eval"
mkdir -p "$OUTPUT/runs" "$EVAL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
log "method=$METHOD epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE run=$RUN"

# ------------------------------------------------------------------- train
# ${cf_flag[@]+...}: an empty array is "unbound" under `set -u` in bash < 4.4.
cf_flag=(); [[ "$CF" == 1 ]] && cf_flag=(--counterfactual)
[[ "$ALL" == 1 ]] && cf_flag+=(--constrained)
if [[ -f "$RUN/manifest.json" ]]; then
  log "training already finished; skipping to evaluation"
else
  # shellcheck disable=SC2086
  "$PYTHON" -m finetune.sft --model "$POLICY_MODEL" ${cf_flag[@]+"${cf_flag[@]}"} \
    --train "$DATA/splits/train.json" --validation "$DATA/splits/validation.json" \
    --gold-dir "$GOLD" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --micro-batch-size "$MICRO_BATCH" --learning-rate "$LEARNING_RATE" \
    --eval-every "$EVAL_EVERY" --eval-per-view "$EVAL_PER_VIEW" --state-every "$STATE_EVERY" \
    --output-dir "$RUN" --method "$METHOD" --resume $TRAIN_EXTRA
fi
[[ -f "$RUN/best/adapter_config.json" ]] || { echo "no best/ adapter in $RUN" >&2; exit 1; }

# ---------------------------------------------------------- generate answers
models=("$METHOD"); [[ "$EVAL_BASE" == 1 ]] && models+=(base)
responses=()
for model in "${models[@]}"; do
  lora=(); [[ "$model" != base ]] && lora=(--lora "$RUN/best")
  for split in test_seen test_unseen; do
    out="$EVAL/$model/$split"
    log "generate $model $split (open-ended)"
    # shellcheck disable=SC2086
    "$PYTHON" -m behavioral_evaluation.generate --model "$POLICY_MODEL" ${lora[@]+"${lora[@]}"} \
      --input "$DATA/splits/$split.json" --output "$out/unconstrained.jsonl" \
      --conditions $CONDITIONS --max-tokens 1024 --batch-size "$EVAL_BATCH" $GENERATE_EXTRA
    log "generate $model $split (A/B)"
    # shellcheck disable=SC2086
    "$PYTHON" -m behavioral_evaluation.generate --model "$POLICY_MODEL" ${lora[@]+"${lora[@]}"} --options \
      --answer-only-retry --input "$DATA/splits/$split.json" --output "$out/constrained.jsonl" \
      --conditions $CONDITIONS --max-tokens 768 --batch-size "$EVAL_BATCH" $GENERATE_EXTRA
    responses+=(--response "$model-$split=$out/unconstrained.jsonl")
  done
done

# ------------------------------------------------------------------- judge
log "judge with $JUDGE_MODEL"
# shellcheck disable=SC2086
"$PYTHON" -m behavioral_evaluation.judge --model "$JUDGE_MODEL" --judge-slug "$JUDGE_SLUG" \
  --dataset "$DATA/records.json" "${responses[@]}" --output-root "$EVAL/judgments" \
  --tensor-parallel-size "$JUDGE_TP" --batch-size "$JUDGE_BATCH" --max-tokens 512 $JUDGE_EXTRA

# One table for every model evaluated so far: A/B accuracy and judged alignment
# per model, split and condition.
"$PYTHON" -m finetune.summarize --eval-root "$EVAL" --judge-slug "$JUDGE_SLUG" --output "$EVAL/summary.json"
log "DONE $METHOD -- table in $EVAL/summary.json"
