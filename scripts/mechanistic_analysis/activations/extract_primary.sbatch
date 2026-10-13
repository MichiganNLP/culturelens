#!/usr/bin/env bash
#SBATCH --job-name=clv2-act-primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=2
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

WORKSPACE=${WORKSPACE:?Set WORKSPACE to the repository root}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
ACTIVATIONS_ROOT=${ACTIVATIONS_ROOT:-$STORAGE/culturelens-v2-activations/assistant-start-v1}
DATASET=${DATASET:-data/processed/culturelens/records.json}

CONDITIONS=(
  base
  country_only
  country_norm
  country_opposite_norm
)

LLAMA_MODEL=/models/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
GEMMA_MODEL=/models/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
QWEN_MODEL=/models/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654

module load singularity

for required in "$WORKSPACE" "$WORKSPACE/$DATASET" "$SIF" "$MODELS_DIR"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
mkdir -p "$ACTIVATIONS_ROOT"

echo "CultureLens v2 assistant-start activation extraction"
echo "Workspace:       $WORKSPACE"
echo "Dataset:         $WORKSPACE/$DATASET"
echo "Activation root: $ACTIVATIONS_ROOT"
echo "Conditions:      ${CONDITIONS[*]}"
echo "Models:          llama-3.1-8b gemma-3-12b qwen-3.5-27b"
nvidia-smi -L

extract_model() {
  local model=$1
  local slug=$2
  local batch_size=$3
  shift 3
  local extra=("$@")

  echo "===== extracting $slug ====="
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace \
    --bind "$MODELS_DIR":/models \
    --bind "$ACTIVATIONS_ROOT":/activations \
    --env PYTHONPATH=/workspace/src \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface \
    --pwd /workspace "$SIF" /usr/bin/python3 -u \
      -m mechanistic_analysis.activations.extract \
      --model "$model" \
      --model-slug "$slug" \
      --input "$DATASET" \
      --output-root /activations \
      --variant unconstrained \
      --conditions "${CONDITIONS[@]}" \
      --dtype bfloat16 \
      --device-map balanced \
      --batch-size "$batch_size" \
      --checkpoint-size 128 \
      --max-length 8192 \
      --placebo-seed 42 \
      --local-files-only \
      "${extra[@]}"
}

extract_model "$LLAMA_MODEL" llama-3.1-8b 16
extract_model "$GEMMA_MODEL" gemma-3-12b 8
extract_model "$QWEN_MODEL" qwen-3.5-27b 4 --trust-remote-code --no-thinking

echo "All requested assistant-start activation caches are complete."
find "$ACTIVATIONS_ROOT/unconstrained" -type f -name '*.pt' -print -exec du -h {} \;
