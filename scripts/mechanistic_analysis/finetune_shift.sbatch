#!/usr/bin/env bash
#SBATCH --job-name=clv2-ft-shift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# What fine-tuning changes, for one model and one LoRA adapter, on the held-out test
# records. STEPS chooses what runs (default: shift read answers full judge):
#   shift    margins and last-prompt-token states of the untuned and the tuned model on
#            the no-country and country-only prompts, then the mean shift added to the
#            untuned model and removed from the tuned one (finetune_shift);
#   read     knockout and erasure of the country line, untuned and tuned (country_reading);
#   conflict whether the state separates a norm that fits the country from one that
#            does not, untuned and tuned (mechanistic_analysis.conflict);
#   route    how the answer reads a stated norm, untuned and tuned: attention knockout by
#            layer window (interventions.localize; not for Qwen, whose layers cannot be
#            masked) and norm erasure by depth (interventions.erasure);
#   answers  answers written with the mean shift added or removed, on a sample;
#   full     the tuned model's answers to the two no-norm prompts on every test record;
#   norms    the tuned model's answers on every test record with a true or a
#            contradictory norm stated, with and without the country;
#   value    the tuned model's answers on every test record that has a value, with the
#            country and the value stated;
#   swap     the country-swap control (country_swap.py): the tuned model's answers on
#            every test record with a distant country's line in place of the right one;
#            the untuned model's on a sample, with the wrong and with the right country;
#            and the conflict test with the wrong country, untuned and tuned
#            (SWAP_UNTUNED=0 leaves out the untuned model's answers);
#   request  personal requests (behavioral_evaluation.requests): a writer model drafts,
#            per test record, a sentence in which a person in the situation asks for the
#            behaviour the custom does not prescribe; the untuned and tuned models then
#            answer a sample of those prompts (VARIANT=country: the tuned model only);
#   stance   whether each of those answers does what was asked (behavioral_evaluation.stance
#            with request_stance.txt), coded by JUDGE_KEY;
#   judge    judgments of the answer sets named in JUDGE_SETS (shift full norms value;
#            swapfull swap500 norm500 for the swap control; request for the requests), by
#            JUDGE_KEY=gemma (Gemma-4-31B) or qwen (Qwen-3.6-27B). JUDGE_TP=2 spreads the
#            judge over two GPUs (needed when one has under about 64 GB).
# Steps already done are skipped, so a job cut off by its time limit can be submitted again.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo>,MODEL_KEY=llama scripts/mechanistic_analysis/finetune_shift.sbatch
# MODEL_KEY is llama, gemma or qwen. VARIANT is cf (the sft-cf adapter) or country (plain
# sft); RUN_NAME names the adapter folder under output/finetune/runs if it is another one.
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
HUB=/models/.cache/huggingface/hub
MODEL_KEY=${MODEL_KEY:-llama}
VARIANT=${VARIANT:-cf}
case "$MODEL_KEY" in
  llama) SLUG=llama-3.1-8b; MODEL=$HUB/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
         PREFIX=collab; BATCH=32; EXTRA=(); READ=(--window 4 --cuts 16); STEER_LAYERS=${STEER_LAYERS-12 20} ;;
  gemma) SLUG=gemma-3-12b; MODEL=$HUB/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
         PREFIX=collab-gemma; BATCH=24; EXTRA=(); READ=(--window 6 --cuts 12); STEER_LAYERS=${STEER_LAYERS-18 30} ;;
  qwen)  SLUG=qwen-3.5-27b; MODEL=$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654
         PREFIX=collab-qwen; BATCH=16; EXTRA=(--trust-remote-code --no-thinking); READ=(--cuts 16); STEER_LAYERS=${STEER_LAYERS-24 40} ;;
  *) echo "MODEL_KEY must be llama, gemma or qwen" >&2; exit 1 ;;
esac
RUN_NAME=${RUN_NAME:-$PREFIX-sft-$VARIANT}
STEPS=${STEPS:-shift read answers full judge}
JUDGE_SETS=${JUDGE_SETS:-shift full norms}
JUDGE_KEY=${JUDGE_KEY:-gemma}
SAMPLE=${SAMPLE:-3000}                 # more than the test records: all of them
READ_SAMPLE=${READ_SAMPLE:-1000}
ANSWER_SAMPLE=${ANSWER_SAMPLE:-500}
MECH=output/finetune/mechanism
ADAPTER=output/finetune/runs/$RUN_NAME/best
RECORDS=$MECH/test_all.json
BASE_NAME=base-$SLUG; [[ "$MODEL_KEY" == llama ]] && BASE_NAME=base
has() { [[ " $STEPS " == *" $1 "* ]]; }

module load singularity
cd "$WORKSPACE"
[[ -s "$RECORDS" ]] || { echo "missing $RECORDS (scripts/finetuning/sft_mech.sbatch writes it)" >&2; exit 1; }
[[ -s "$ADAPTER/adapter_config.json" ]] || { echo "no adapter at $ADAPTER" >&2; exit 1; }
py() {
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src:$STORAGE/py-peft --bind "$STORAGE/py-peft" \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface --env HF_HUB_OFFLINE=1 \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m "$@"
}
common=(--model "$MODEL" --model-slug "$SLUG" --records "$RECORDS" --local-files-only
        --device-map none --device cuda:0 ${EXTRA[@]+"${EXTRA[@]}"})
SHIFT=$MECH/$RUN_NAME/finetune_shift

if has shift; then
  if [[ -s "$SHIFT/$SLUG/metrics.json" ]]; then echo "finetune_shift: done"; else
    echo "===== finetune_shift $RUN_NAME ====="
    py mechanistic_analysis.interventions.finetune_shift "${common[@]}" --adapter "$ADAPTER" \
      --sample "$SAMPLE" --batch-size "$BATCH" --output-root "$SHIFT"
  fi
fi
for step in read conflict; do
  has "$step" || continue
  module=mechanistic_analysis.interventions.country_reading; folder=country_reading
  flags=(--sample "$READ_SAMPLE" "${READ[@]}")
  [[ "$step" == conflict ]] && { module=mechanistic_analysis.conflict; folder=conflict; flags=(--sample 1000); }
  for name in "$BASE_NAME" "$RUN_NAME"; do
    out="$MECH/$name/$folder"
    if [[ -s "$out/$SLUG/metrics.json" ]]; then echo "$folder $name: done"; continue; fi
    echo "===== $folder $name ====="
    lora=(); [[ "$name" != "$BASE_NAME" ]] && lora=(--lora "$ADAPTER")
    py "$module" "${common[@]}" ${lora[@]+"${lora[@]}"} "${flags[@]}" --batch-size "$BATCH" --output-root "$out"
  done
done

if has route; then
  window=4; [[ "$MODEL_KEY" == gemma ]] && window=6
  for step in localize erasure; do
    [[ "$step" == localize && "$MODEL_KEY" == qwen ]] && continue
    flags=(); [[ "$step" == localize ]] && flags=(--window "$window")
    for name in "$BASE_NAME" "$RUN_NAME"; do
      out="$MECH/$name/$step"
      if [[ -s "$out/$SLUG/metrics.json" ]]; then echo "$step $name: done"; continue; fi
      echo "===== $step $name ====="
      lora=(); [[ "$name" != "$BASE_NAME" ]] && lora=(--lora "$ADAPTER")
      py "mechanistic_analysis.interventions.$step" "${common[@]}" ${lora[@]+"${lora[@]}"} \
        ${flags[@]+"${flags[@]}"} --sample "${ROUTE_SAMPLE:-400}" --batch-size "$BATCH" --output-root "$out"
    done
  done
fi

generate() {   # $1 output root, $2 sample, then further flags
  local root=$1 sample=$2; shift 2
  # shellcheck disable=SC2086
  py mechanistic_analysis.interventions.finetune_shift_generate "${common[@]}" --adapter "$ADAPTER" \
    --shift "$SHIFT/$SLUG/finetune_shift.npz" --sample "$sample" --steer-layers ${STEER_LAYERS:-0} \
    --batch-size "$BATCH" --max-new-tokens 1024 --output-root "$root" "$@"
}
if has answers; then
  echo "===== answers with the shift ====="
  generate "$MECH/$RUN_NAME/shift_answers" "$ANSWER_SAMPLE"
fi
if has full; then
  echo "===== the tuned model's answers on every test record ====="
  generate "$MECH/$RUN_NAME/tuned_answers" "$SAMPLE" --arms tuned
fi
if has norms; then
  echo "===== the tuned model's answers with a norm stated ====="
  generate "$MECH/$RUN_NAME/norm_answers" "$SAMPLE" --arms tuned \
    --prompts country_norm country_opposite_norm norm_only opposite_norm_only
fi
if has value; then
  echo "===== the tuned model's answers with the value stated ====="
  generate "$MECH/$RUN_NAME/value_answers" "$SAMPLE" --arms tuned --prompts country_value
fi
if has swap; then
  SWAPPED=$MECH/test_all_swapped.json
  if [[ ! -s "$SWAPPED" ]]; then
    py mechanistic_analysis.interventions.country_swap --records "$RECORDS" \
      --all-records data/processed/culturelens/records.json \
      --geometry output/mechanistic_analysis/country_space/llama-3.1-8b/geometry.npz --output "$SWAPPED"
  fi
  WITH_NORM="country_norm country_opposite_norm"
  echo "===== swap: the tuned model, every test record, wrong country ====="
  # shellcheck disable=SC2086
  generate "$MECH/$RUN_NAME/swap_answers_full" "$SAMPLE" --records "$SWAPPED" --arms tuned \
    --prompts country_only $WITH_NORM
  if [[ "${SWAP_UNTUNED:-1}" == 1 ]]; then
    echo "===== swap: the untuned model, a sample, wrong country ====="
    # shellcheck disable=SC2086
    generate "$MECH/$RUN_NAME/swap_answers_500" "$ANSWER_SAMPLE" --records "$SWAPPED" --arms untuned \
      --prompts country_only $WITH_NORM
    echo "===== swap: the untuned model, the same sample, right country, norm stated ====="
    # shellcheck disable=SC2086
    generate "$MECH/$RUN_NAME/norm_answers_500" "$ANSWER_SAMPLE" --arms untuned --prompts $WITH_NORM
  fi
  for name in "$BASE_NAME" "$RUN_NAME"; do
    out="$MECH/$name/conflict_swapped"
    if [[ -s "$out/$SLUG/metrics.json" ]]; then echo "conflict_swapped $name: done"; continue; fi
    echo "===== conflict with the wrong country: $name ====="
    lora=(); [[ "$name" != "$BASE_NAME" ]] && lora=(--lora "$ADAPTER")
    py mechanistic_analysis.conflict "${common[@]}" --records "$SWAPPED" ${lora[@]+"${lora[@]}"} \
      --sample 1000 --batch-size "$BATCH" --output-root "$out"
  done
fi
GEMMA4=$HUB/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
QWEN36=$HUB/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
REQUESTS=$MECH/test_requests.json
if { has request || has write; } && [[ ! -s "$REQUESTS" ]]; then      # STEPS=write: the requests only
  echo "===== writing the requests ====="
  py behavioral_evaluation.requests --model "$GEMMA4" --records "$RECORDS" --output "$REQUESTS" \
    --dtype bfloat16 --language-model-only --disable-custom-all-reduce --enforce-eager
fi
if has request; then
  arms="untuned tuned"; [[ "$VARIANT" == country ]] && arms="tuned"
  echo "===== answers to a personal request ====="
  # shellcheck disable=SC2086
  generate "$MECH/$RUN_NAME/request_answers" "$ANSWER_SAMPLE" --records "$REQUESTS" --arms $arms \
    --prompts country_opposite_norm
fi
if has stance; then
  coder=(--model "$GEMMA4" --coder-slug gemma-4-31b --language-model-only)
  [[ "$JUDGE_KEY" == qwen ]] && coder=(--model "$QWEN36" --coder-slug qwen-3.6-27b --trust-remote-code --no-think)
  asked=()
  for file in "$MECH/$RUN_NAME/request_answers/$SLUG"/*.jsonl; do
    [[ -s "$file" ]] && asked+=(--response "request-$RUN_NAME.$(basename "$file" .jsonl)=$file")
  done
  echo "===== does the answer do what was asked: ${coder[3]} ====="
  py behavioral_evaluation.stance "${coder[@]}" --dataset "$REQUESTS" "${asked[@]}" \
    --conditions country_opposite_norm --prompt prompts/behavioral_evaluation/request_stance.txt \
    --output-root "$MECH/$RUN_NAME/request_stance" --dtype bfloat16 --batch-size 64 \
    --disable-custom-all-reduce --enforce-eager \
    || echo "some answers could not be coded; submit again to retry them"
fi
if has judge; then
  responses=()
  for set in $JUDGE_SETS; do
    case "$set" in shift) folder=shift_answers ;; full) folder=tuned_answers ;; norms) folder=norm_answers ;;
      swapfull) folder=swap_answers_full ;; swap500) folder=swap_answers_500 ;; norm500) folder=norm_answers_500 ;;
      request) folder=request_answers ;; value) folder=value_answers ;;
      *) echo "unknown answer set $set" >&2; exit 1 ;; esac
    for file in "$MECH/$RUN_NAME/$folder/$SLUG"/*.jsonl; do
      [[ -s "$file" ]] && responses+=(--response "$set-$RUN_NAME.$(basename "$file" .jsonl)=$file")
    done
  done
  if [[ ${#responses[@]} -eq 0 ]]; then echo "nothing to judge"; else
    judge=(--model "$GEMMA4" --judge-slug gemma-4-31b)
    [[ "$JUDGE_KEY" == qwen ]] && judge=(--model "$QWEN36" --judge-slug qwen-3.6-27b --trust-remote-code --no-think)
    echo "===== judge: ${judge[3]} ====="
    py behavioral_evaluation.judge "${judge[@]}" \
      --dataset data/processed/culturelens/records.json "${responses[@]}" \
      --output-root "$MECH/$RUN_NAME/judgments" --tensor-parallel-size "${JUDGE_TP:-1}" --batch-size 64 --max-tokens 1536 \
      --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 --seed 42 \
      --language-model-only --disable-custom-all-reduce --enforce-eager --retries "${JUDGE_RETRIES:-0}"
  fi
fi
echo "finished: $RUN_NAME ($STEPS)"
