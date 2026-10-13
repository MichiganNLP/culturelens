#!/usr/bin/env bash
#SBATCH --job-name=clv2-beh-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

JUDGE=${JUDGE:?Set JUDGE to gemma or qwen}
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
ROOT=output/behavioral_evaluation

case "$JUDGE" in
  gemma)
    JUDGE_MODEL=/models/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
    JUDGE_SLUG=gemma-4-31b
    EXTRA=(--language-model-only --disable-custom-all-reduce --enforce-eager)
    ;;
  qwen)
    JUDGE_MODEL=/models/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
    JUDGE_SLUG=qwen-3.6-27b
    EXTRA=(--trust-remote-code --no-think --language-model-only \
      --disable-custom-all-reduce --enforce-eager)
    ;;
  *) echo "ERROR: JUDGE must be gemma or qwen" >&2; exit 2 ;;
esac

module load singularity
cd "$WORKSPACE"
for slug in llama-3.1-8b gemma-3-12b qwen-3.5-27b; do
  [[ -s "$ROOT/unconstrained/responses/$slug.jsonl" ]] || {
    echo "ERROR: missing unconstrained responses for $slug" >&2
    exit 1
  }
done

singularity exec --cleanenv --nv \
  --bind "$WORKSPACE":/workspace \
  --bind "$MODELS_DIR":/models \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/models/.cache/huggingface \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --pwd /workspace "$SIF" /usr/bin/python3 -u -m behavioral_evaluation.judge \
    --model "$JUDGE_MODEL" --judge-slug "$JUDGE_SLUG" \
    --dataset "$DATASET" \
    --response "llama-3.1-8b=$ROOT/unconstrained/responses/llama-3.1-8b.jsonl" \
    --response "gemma-3-12b=$ROOT/unconstrained/responses/gemma-3-12b.jsonl" \
    --response "qwen-3.5-27b=$ROOT/unconstrained/responses/qwen-3.5-27b.jsonl" \
    --output-root "$ROOT/unconstrained/judgments" \
    --tensor-parallel-size 2 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --batch-size 16 \
    --max-model-len 8192 --max-tokens 512 --retries 2 --seed 42 \
    "${EXTRA[@]}"

echo "$JUDGE_SLUG judging is complete."
