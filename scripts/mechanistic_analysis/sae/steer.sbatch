#!/usr/bin/env bash
#SBATCH --job-name=clv2-sae-steer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Causal follow-up to the SAE analysis, for Qwen-3.5-27B (2 GPUs):
#   1. read the chosen features under the framings and under norm erasure
#      (mechanistic_analysis/sae/read.py; forward passes only);
#   2. generate answers while one feature is ablated or clamped
#      (interventions/generate.py with --feature-plan);
#   3. judge those answers with both judges.
# Every step keeps what is already written, so the job can be submitted again.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/mechanistic_analysis/sae/steer.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
SLUG=qwen-3.5-27b
SAMPLE=${SAMPLE:-200}
STEPS=${STEPS:-read generate judge}
HUB=/models/.cache/huggingface/hub
MODEL=$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654
SAE=/models/SAE-Res-Qwen3.5-27B-W80K-L0_50
PLAN=${PLAN:-output/mechanistic_analysis/sae/steering/plan.$SLUG.json}
RECORDS=data/processed/culturelens/records.json
ANSWERS=output/mechanistic_analysis/answers

module load singularity
cd "$WORKSPACE"
[[ -s "$PLAN" ]] || { echo "ERROR: missing $PLAN" >&2; exit 1; }
run() {
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface --env HF_HUB_OFFLINE=1 \
    --pwd /workspace "$SIF" /usr/bin/python3 -u "$@"
}
arms=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['arms']))" "$PLAN")
features=$(python3 -c "
import json,sys
s=json.load(open(sys.argv[1]))['selection']
print(' '.join(sorted({f\"{v['layer']}:{f}\" for v in s.values() for f in (v['feature'], v['random']['feature'])})))" "$PLAN")
echo "arms: $arms"; echo "features: $features"

if [[ " $STEPS " == *" read "* ]]; then
  echo "===== feature readings ====="
  # shellcheck disable=SC2086
  run -m mechanistic_analysis.sae.read --model "$MODEL" --model-slug "$SLUG" --records "$RECORDS" \
    --sample "$SAMPLE" --sae-root "$SAE" --features $features --erase-layers 24 36 \
    --output-root output/mechanistic_analysis/sae/readings --batch-size 16 \
    --trust-remote-code --no-thinking --local-files-only
fi
if [[ " $STEPS " == *" generate "* ]]; then
  echo "===== answers under feature steering ====="
  # shellcheck disable=SC2086
  run -m mechanistic_analysis.interventions.generate --model "$MODEL" --model-slug "$SLUG" \
    --records "$RECORDS" --sample "$SAMPLE" --output-root "$ANSWERS" --batch-size 16 \
    --arms true $arms --feature-plan "$PLAN" --sae-root "$SAE" \
    --trust-remote-code --no-thinking --local-files-only
fi
if [[ " $STEPS " == *" judge "* ]]; then
  responses=()
  for arm in true $arms; do
    [[ -s "$ANSWERS/$SLUG/$arm.jsonl" ]] && responses+=(--response "$SLUG.$arm=$ANSWERS/$SLUG/$arm.jsonl")
  done
  common=(--dataset "$RECORDS" "${responses[@]}" --output-root output/mechanistic_analysis/judgments
          --tensor-parallel-size "${JUDGE_TP:-2}" --dtype bfloat16 --gpu-memory-utilization 0.90 --batch-size 64
          --max-model-len 8192 --max-tokens 1536 --retries 0 --seed 42 --progress-every 10
          --language-model-only --disable-custom-all-reduce --enforce-eager)
  echo "===== judge: gemma-4-31b ====="
  run -m behavioral_evaluation.judge --judge-slug gemma-4-31b "${common[@]}" \
    --model "$HUB/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475" \
    || echo "some answers could not be judged; rerun to retry them"
  echo "===== judge: qwen-3.6-27b ====="
  run -m behavioral_evaluation.judge --judge-slug qwen-3.6-27b "${common[@]}" --trust-remote-code --no-think \
    --model "$HUB/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9" \
    || echo "some answers could not be judged; rerun to retry them"
fi
echo "SAE steering is complete."
