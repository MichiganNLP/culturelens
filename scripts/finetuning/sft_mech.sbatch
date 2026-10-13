#!/usr/bin/env bash
# Mechanistically placed variants of sft-cf (open-ended answers + counterfactual), for
# Llama-3.1-8B. Same data, epochs and learning rate; only where the adapter goes,
# or one added loss, differs. Choose the variant with ARM:
#
#   ARM=band      adapter on the blocks where the stated norm is read (12-15)
#   ARM=control   adapter on a later band of the same width (24-27)
#   ARM=heads     adapter on the eight attention heads that carry the norm
#   ARM=conflict  adapter everywhere, plus the conflict loss after block 16
#   ARM=distill   adapter everywhere, plus distillation from the untuned model with
#                 the contradictory norm knocked out of its attention
#   ARM=gate      adapter everywhere, plus a loss that closes the eight heads'
#                 attention to a contradictory norm
#   ARM=framings  adapter everywhere; the stated norm is presented in five ways
#   ARM=mirror    mirror verification (finetune.mirror): the adapter learns to write
#                 the opposite of a stated norm, to say which of the two is accurate
#                 for the country, and to answer; at evaluation a stated norm is
#                 compared with its opposite before the answer is written. It is not
#                 trained on the contradictory-norm view, and has no A/B evaluation.
#   ARM=mirror-v2 the same, and the adapter also learns to write the norm that applies
#                 when none is stated, so those prompts are verified too. Their direct
#                 answers are generated and judged as well, as <method>-direct.
#
# The last three are also evaluated with the contradictory norm reframed
# (attributed to hearsay, to a guide, followed by doubt, placed before the country).
#
# After evaluation the job also measures, on held-out test records, how the
# fine-tuned model reads a stated norm (attention knockout by layer window,
# norm erasure by depth) and whether its state separates a norm that fits the
# country from one that does not (mechanistic_analysis.conflict).
#   ARM=value     sft-cf, and the adapter also learns to write the value at stake for a
#                 country-only prompt (finetune.value_first). At evaluation a country-only
#                 prompt is answered through the value the model writes; its direct
#                 answers are generated and judged as well, as <method>-direct.
#   ARM=memory    sft-cf, with trainable vectors per country read where a stated norm
#                 goes (finetune.country_memory). Answers are generated with transformers,
#                 because the vectors are swapped in at the embedding layer. Country-only
#                 answers without the memory are judged as well, as <method>-direct.
#   ARM=plain     sft-cf itself, with no change. With TRAIN=<file named train.json> it is
#                 trained on that file instead (a subsample, for a data-scaling curve),
#                 and METHOD_NAME names the run. A run folder that already holds
#                 manifest.json and best/ (an adapter trained elsewhere) is only evaluated.
#   ARM=reanalysis runs only that last part, for the runs named in RUNS and for
#   the untuned model (default RUNS="sft-all-cf").
#
# One job: fine-tune Llama-3.1-8B-Instruct, generate answers on both test splits
# with the best checkpoint, judge the open answers with Gemma-4-31B, and write a
# results table.
#
# Submit from the repository root, adding your own allocation:
#   sbatch --account=<account> --partition=<gpu-partition> scripts/finetuning/sft_all_cf.sbatch
#
# Needs: torch 2.11, transformers 5.12, peft 0.21, accelerate 1.14, vllm 0.24,
# pydantic, pyyaml -- and `huggingface-cli login`, because Llama-3.1 is gated.
# Data must be at data/processed/culturelens/ (records.json,
# splits/{train,validation,test_seen,test_unseen}.json) and the gold answers at
# data/processed/culturelens/gold/ (train.country_only.jsonl, train.country_opposite_norm.jsonl;
# the matching validation.* files are used for selection when present).
#
# If the job is killed by its time limit, submit it again. Training restores its
# saved state, and generation and the judge skip what they already wrote.
#
# Every setting below can be overridden from the environment, e.g.
#   OUTPUT=/path/to/output EVAL_BASE=1 sbatch ... scripts/finetuning/sft_all_cf.sbatch
#
#SBATCH --job-name=cl-sft-mech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail

CF=1                   # 1 adds country_opposite_norm, country_norm and country_value
ALL="${ALL:-0}"         # 1 also trains on the A/B task; 0 trains on open-ended answers only
ARM="${ARM:?set ARM to band, control, heads, conflict, distill, gate, framings, mirror, mirror-v2, value, memory, plain or reanalysis}"
case "$ARM" in
  band)       ARM_FLAGS="--lora-layers 12 13 14 15" ;;
  control)    ARM_FLAGS="--lora-layers 24 25 26 27" ;;
  heads)      ARM_FLAGS="--lora-heads 12.4 12.22 13.4 13.21 13.22 13.23 14.5 14.24" ;;
  conflict)   ARM_FLAGS="--conflict-weight ${CONFLICT_WEIGHT:-0.1} --conflict-layer 16" ;;
  distill)    ARM_FLAGS="--distill-weight ${DISTILL_WEIGHT:-0.5} --distill-activation-weight ${DISTILL_STATE_WEIGHT:-0.5} --distill-layer 16" ;;
  gate)       ARM_FLAGS="--gate-weight ${GATE_WEIGHT:-1.0} --gate-heads 12.4 12.22 13.4 13.21 13.22 13.23 14.5 14.24" ;;
  framings)   ARM_FLAGS="--framings" ;;
  mirror)     ARM_FLAGS="--mirror --task-tokens ${TASK_TOKENS:-300}" ;;
  mirror-v2)  ARM_FLAGS="--mirror --propose --task-tokens ${TASK_TOKENS:-300}" ;;
  memory)     ARM_FLAGS="--country-memory ${MEMORY_SLOTS:-8}" ;;
  value)      ARM_FLAGS="--value-first --task-tokens ${TASK_TOKENS:-300}" ;;
  plain|reanalysis) ARM_FLAGS="" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac

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
GEOMETRY="${GEOMETRY:-$REPO/output/mechanistic_analysis/country_space/llama-3.1-8b/geometry.npz}"
TRAIN="${TRAIN:-$DATA/splits/train.json}"     # the gold answers are found by this file's name

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
FRAMED="country_opposite_norm_hearsay country_opposite_norm_guide country_opposite_norm_doubt opposite_norm_country"
case "$ARM" in distill|gate|framings) CONDITIONS="$CONDITIONS $FRAMED" ;; esac
EVAL_BATCH="${EVAL_BATCH:-64}"
CONSTRAINED="${CONSTRAINED:-1}"; [[ "$ARM" == mirror* ]] && CONSTRAINED=0   # 0 skips the A/B evaluation
MECHANISM="${MECHANISM:-1}"                   # 0 skips the knockout, erasure and conflict checks
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

METHOD="sft-cf-$ARM"; [[ "$ALL" == 1 ]] && METHOD="sft-all-cf-$ARM"; [[ "$ARM" == mirror* ]] && METHOD="sft-$ARM"
[[ "$ARM" == plain ]] && METHOD="${METHOD_NAME:-sft-cf}"
RUN="$OUTPUT/runs/$METHOD"
EVAL="$OUTPUT/eval"
MECH="$OUTPUT/mechanism"
MODEL_SLUG="${MODEL_SLUG:-llama-3.1-8b}"
MECH_SAMPLE="${MECH_SAMPLE:-400}"            # test records for knockout and erasure
CONFLICT_SAMPLE="${CONFLICT_SAMPLE:-1000}"   # test records for the conflict test
mkdir -p "$OUTPUT/runs" "$EVAL" "$MECH"

# How a model reads a stated norm, on held-out records only. $1 is a name,
# $2 an adapter directory (empty for the untuned model). Steps already done are skipped.
mechanism() {
  local name=$1 adapter=$2 lora=() step
  [[ -n "$adapter" ]] && lora=(--lora "$adapter")
  [[ -s "$MECH/test_all.json" ]] || "$PYTHON" -c "
import json, sys
rows = [r for f in sys.argv[2:] for r in json.load(open(f))]
json.dump(rows, open(sys.argv[1], 'w'))" "$MECH/test_all.json" "$DATA/splits/test_seen.json" "$DATA/splits/test_unseen.json"
  for step in conflict localize erasure; do
    local out="$MECH/$name/$step" module="mechanistic_analysis.interventions.$step" sample="$MECH_SAMPLE"
    [[ "$step" == conflict ]] && { module="mechanistic_analysis.conflict"; sample="$CONFLICT_SAMPLE"; }
    if compgen -G "$out/$MODEL_SLUG/*.json" > /dev/null; then log "mechanism $name $step: done"; continue; fi
    log "mechanism $name $step"
    "$PYTHON" -m "$module" --model "$POLICY_MODEL" --model-slug "$MODEL_SLUG" ${lora[@]+"${lora[@]}"} \
      --records "$MECH/test_all.json" --sample "$sample" --output-root "$out" --device-map none --device cuda:0
  done
}

if [[ "$ARM" == reanalysis ]]; then
  mechanism base ""
  for name in ${RUNS:-sft-cf}; do
    if [[ -f "$OUTPUT/runs/$name/best/adapter_config.json" ]]; then mechanism "$name" "$OUTPUT/runs/$name/best"
    else log "no adapter for $name yet; skipped"; fi
  done
  log "DONE reanalysis"
  exit 0
fi
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
    --train "$TRAIN" --validation "$DATA/splits/validation.json" \
    --gold-dir "$GOLD" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --micro-batch-size "$MICRO_BATCH" --learning-rate "$LEARNING_RATE" \
    --eval-every "$EVAL_EVERY" --eval-per-view "$EVAL_PER_VIEW" --state-every "$STATE_EVERY" \
    --output-dir "$RUN" --method "$METHOD" --resume $ARM_FLAGS $TRAIN_EXTRA
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
    generator=behavioral_evaluation.generate; propose=""; second=""
    [[ "$ARM" == mirror* && "$model" != base ]] && generator=finetune.mirror
    [[ "$ARM" == mirror-v2 && "$model" != base ]] && propose="--propose"
    [[ "$ARM" == value && "$model" != base ]] && { generator=finetune.value_first; propose="--predict"; }
    [[ "$ARM" == memory && "$model" != base ]] && { generator=finetune.country_memory
      propose="--geometry $GEOMETRY --local-files-only"; second="--no-memory --local-files-only"; }
    # shellcheck disable=SC2086
    "$PYTHON" -m "$generator" --model "$POLICY_MODEL" ${lora[@]+"${lora[@]}"} \
      --input "$DATA/splits/$split.json" --output "$out/unconstrained.jsonl" \
      --conditions $CONDITIONS --max-tokens 1024 --batch-size "$EVAL_BATCH" $propose $GENERATE_EXTRA
    if [[ -n "$propose" ]]; then
      log "generate $model $split (open-ended, no norm proposed)"
      # shellcheck disable=SC2086
      direct="country_only country_value"; [[ "$ARM" == value || "$ARM" == memory ]] && direct="country_only"
      "$PYTHON" -m "$generator" --model "$POLICY_MODEL" --lora "$RUN/best" \
        --input "$DATA/splits/$split.json" --output "$EVAL/$model-direct/$split/unconstrained.jsonl" \
        --conditions $direct --max-tokens 1024 --batch-size "$EVAL_BATCH" $second $GENERATE_EXTRA
      responses+=(--response "$model-direct-$split=$EVAL/$model-direct/$split/unconstrained.jsonl")
    fi
    if [[ "$CONSTRAINED" == 1 ]]; then
      log "generate $model $split (A/B)"
      # shellcheck disable=SC2086
      "$PYTHON" -m behavioral_evaluation.generate --model "$POLICY_MODEL" ${lora[@]+"${lora[@]}"} --options \
        --answer-only-retry --input "$DATA/splits/$split.json" --output "$out/constrained.jsonl" \
        --conditions $CONDITIONS --max-tokens 768 --batch-size "$EVAL_BATCH" $GENERATE_EXTRA
    fi
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
log "evaluated $METHOD -- table in $EVAL/summary.json"

if [[ "$MECHANISM" == 1 ]]; then
  mechanism "$METHOD" "$RUN/best"
  log "DONE $METHOD -- mechanism in $MECH/$METHOD"
else
  log "DONE $METHOD -- mechanism checks skipped (ARM=reanalysis RUNS=$METHOD runs them later)"
fi
