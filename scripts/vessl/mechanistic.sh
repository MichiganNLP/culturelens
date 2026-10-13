#!/usr/bin/env bash
# GPU job: the mechanistic study of resisting a false norm, three models.
#
#   cached     decision readout and country x claim interaction, from the
#              activation caches (all three models)
#   localize   attention knockout by layer window, scored by the margin
#              (models whose every layer can be masked)
#   answers    open-ended answers under knockout and under the extra prompt
#              conditions, written for the judges (judge_answers.sh)
#   erasure    the claim's positions erased after a given depth, scored by the
#              margin (all three models; needs no attention mask)
#   heads      which heads of the claim window carry the claim (development sample)
#   answers2   answers under erasure (all three models) and with only the
#              selected heads blocked
#
# Inputs already on the volumes: activation caches and models under /work,
# records under /sync/activations/data, outcome tables under
# /sync/mechanistic-study/outcomes (upload output/mechanistic_analysis/outcomes).
# Every step skips what is done and publishes as it goes. Set CODE to a folder
# of its own when another job may be running.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/activations_common.sh"

RECORDS="$SYNC/activations/data/records.json"
OUTCOMES="$SYNC/mechanistic-study/outcomes"
RESULTS="$WORK/mechanistic-study"
DEST="$SYNC/mechanistic-study"
read -r -a STAGES <<< "${STAGES:-cached localize answers}"
SAMPLE="${SAMPLE:-400}"

install_code
python3 -c "import accelerate" 2>/dev/null || {
  pip list --format=freeze > /tmp/image-constraints.txt
  pip install --quiet --root-user-action=ignore -c /tmp/image-constraints.txt accelerate
}
[[ -s "$RECORDS" ]] || { log "no records at $RECORDS"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$RESULTS" "$DEST" "$SYNC/done"
has_stage() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }

if has_stage cached; then
  for slug in llama-3.1-8b gemma-3-12b qwen-3.5-27b; do
    [[ -s "$OUTCOMES/$slug.npz" ]] || { log "no outcome table at $OUTCOMES/$slug.npz"; exit 1; }
    if [[ ! -e "$RESULTS/readout/$slug/metrics.json" ]]; then
      log "===== readout / $slug ====="
      python3 -u -m mechanistic_analysis.readout \
        --activations-root "$ACTIVATIONS" --model-slug "$slug" --records "$RECORDS" \
        --outcomes "$OUTCOMES/$slug.npz" --output-root "$RESULTS/readout" --layer-step 2 --text
    fi
    if [[ ! -e "$RESULTS/interaction/$slug/metrics.json" ]]; then
      log "===== interaction / $slug ====="
      python3 -u -m mechanistic_analysis.interaction \
        --activations-root "$ACTIVATIONS" --model-slug "$slug" --records "$RECORDS" \
        --output-root "$RESULTS/interaction"
    fi
    publish "$RESULTS" "$DEST"
  done
fi

run() {  # module folder slug model batch extra...
  local module=$1 folder=$2 slug=$3 model=$4 batch=$5; shift 5
  [[ -e "$RESULTS/$folder/$slug/metrics.json" && "$folder" != answers ]] && { log "$folder / $slug done"; return 0; }
  log "===== $folder / $slug ====="
  python3 -u -m "mechanistic_analysis.interventions.$module" \
    --model "$model" --model-slug "$slug" --records "$RECORDS" \
    --output-root "$RESULTS/$folder" --batch-size "$batch" --local-files-only "$@" \
    || log "$folder / $slug FAILED; continuing"
  publish "$RESULTS" "$DEST"
}
if has_stage localize; then
  run localize localize llama-3.1-8b "$LLAMA_DIR" 32 --sample "$SAMPLE" --window 4
  run localize localize gemma-3-12b "$GEMMA3_DIR" 24 --sample "$SAMPLE" --window 6
fi
if has_stage answers; then
  # Generation resumes arm by arm from the answers already written.
  run generate answers llama-3.1-8b "$LLAMA_DIR" 48 --sample "$SAMPLE"
  run generate answers gemma-3-12b "$GEMMA3_DIR" 32 --sample "$SAMPLE"
  run generate answers qwen-3.5-27b "$QWEN_DIR" 16 --sample "${QWEN_SAMPLE:-200}" \
    --arms false claim-only --trust-remote-code --no-thinking
fi
if has_stage erasure; then
  run erasure erasure llama-3.1-8b "$LLAMA_DIR" 32 --sample "$SAMPLE" --cuts 16
  run erasure erasure gemma-3-12b "$GEMMA3_DIR" 24 --sample "$SAMPLE" --cuts 12
  run erasure erasure qwen-3.5-27b "$QWEN_DIR" 12 --sample "${QWEN_SAMPLE:-200}" --cuts 8 \
    --trust-remote-code --no-thinking
fi
if has_stage heads; then
  run heads heads llama-3.1-8b "$LLAMA_DIR" 32
  run heads heads gemma-3-12b "$GEMMA3_DIR" 24
fi
if has_stage answers2; then
  ERASE_ARMS="false-erase-first false-erase-before false-erase-after"
  HEAD_ARMS="false-noclaim-heads false-noclaim-randheads"
  # shellcheck disable=SC2086
  run generate answers llama-3.1-8b "$LLAMA_DIR" 48 --sample "$SAMPLE" \
    --heads "$RESULTS/heads/llama-3.1-8b/selection.json" --arms $ERASE_ARMS $HEAD_ARMS
  # shellcheck disable=SC2086
  run generate answers qwen-3.5-27b "$QWEN_DIR" 16 --sample "${QWEN_SAMPLE:-200}" \
    --arms $ERASE_ARMS --trust-remote-code --no-thinking
  # shellcheck disable=SC2086
  run generate answers gemma-3-12b "$GEMMA3_DIR" 32 --sample "$SAMPLE" \
    --heads "$RESULTS/heads/gemma-3-12b/selection.json" --arms $ERASE_ARMS $HEAD_ARMS
fi
publish "$RESULTS" "$DEST"
date -u +%FT%TZ > "$SYNC/done/MECHANISTIC"
log "MECHANISTIC STUDY STAGES DONE"
