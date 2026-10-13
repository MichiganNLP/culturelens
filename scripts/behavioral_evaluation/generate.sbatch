#!/usr/bin/env bash
#SBATCH --job-name=clv2-beh-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=2
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
ROOT=output/behavioral_evaluation

module load singularity
cd "$WORKSPACE"
[[ -s "$DATASET" ]] || { echo "ERROR: missing $WORKSPACE/$DATASET" >&2; exit 1; }
[[ -s "$SIF" ]] || { echo "ERROR: missing $SIF" >&2; exit 1; }

infer() {
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace \
    --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --pwd /workspace "$SIF" /usr/bin/python3 -u "$@"
}

run_model() {
  local model=$1 slug=$2 batch=$3
  shift 3
  local extra=("$@")

  echo "===== constrained: $model ====="
  infer -m behavioral_evaluation.generate \
    --model "$model" --input "$DATASET" \
    --output "$ROOT/constrained/responses/$slug.jsonl" \
    --options --answer-only-retry \
    --tensor-parallel-size 2 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --batch-size "$batch" \
    --max-model-len 8192 --max-tokens 512 --seed 42 \
    --disable-custom-all-reduce "${extra[@]}"

  echo "===== unconstrained: $model ====="
  infer -m behavioral_evaluation.generate \
    --model "$model" --input "$DATASET" \
    --output "$ROOT/unconstrained/responses/$slug.jsonl" \
    --tensor-parallel-size 2 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --batch-size "$batch" \
    --max-model-len 8192 --max-tokens 512 --seed 42 \
    --disable-custom-all-reduce "${extra[@]}"
}

run_model \
  /models/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  llama-3.1-8b 32
run_model \
  /models/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  gemma-3-12b 32
run_model \
  /models/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654 \
  qwen-3.5-27b 16 \
  --trust-remote-code --no-think --language-model-only --enforce-eager

echo "All constrained and unconstrained response generation is complete."
