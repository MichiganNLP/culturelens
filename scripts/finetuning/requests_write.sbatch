#!/usr/bin/env bash
#SBATCH --job-name=cl-requests-write
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Personal requests for the training records (behavioral_evaluation.requests): for each
# record a writer model drafts a sentence in which a person in the situation asks for the
# behaviour the custom does not prescribe. The output is the input of
# `benchmark.generate gold_request`, whose answers `finetune.sft --requests` trains on.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/finetuning/requests_write.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
HUB=/models/.cache/huggingface/hub
WRITER=${WRITER:-$HUB/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}
SPLITS=${SPLITS:-train}
OUT=data/processed/culturelens/requests

module load singularity
cd "$WORKSPACE"
mkdir -p "$OUT"
for split in $SPLITS; do
  if [[ -s "$OUT/$split.json" ]]; then echo "$split: done"; continue; fi
  echo "===== requests for $split ====="
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface --env HF_HUB_OFFLINE=1 \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m behavioral_evaluation.requests \
    --model "$WRITER" --records "data/processed/culturelens/splits/$split.json" --output "$OUT/$split.json" \
    --dtype bfloat16 --language-model-only --disable-custom-all-reduce --enforce-eager
done
echo "finished: $SPLITS"
