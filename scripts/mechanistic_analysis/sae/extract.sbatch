#!/usr/bin/env bash
#SBATCH --job-name=clv2-sae-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Sparse-autoencoder features of the cached assistant-start states, for every
# model, every layer that has a pretrained SAE and every cached prompt condition
# (mechanistic_analysis/sae/extract.py). No language model is loaded. Runs on one
# GPU, or on CPUs alone when none is allocated (slower: allow about ten hours
# and at least 16 cores). Existing feature files are kept, so a job that ran
# out of time can simply be submitted again.
#
#   sbatch --account=<account> --partition=<gpu-partition> --gpus=1 \
#     --export=ALL,WORKSPACE=<repo> scripts/mechanistic_analysis/sae/extract.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
ACTIVATIONS_ROOT=${ACTIVATIONS_ROOT:-$STORAGE/culturelens-v2-activations/assistant-start-v1-vessl}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/mechanistic_analysis/sae/features}
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
LAYERS=${LAYERS:-all}

# Local checkpoint folder of each model's SAE suite, under MODELS_DIR.
declare -A SAE_DIR=(
  [llama-3.1-8b]=saes-llama-3.1-8b-instruct
  [gemma-3-12b]=gemma-scope-2-12b-it
  [qwen-3.5-27b]=SAE-Res-Qwen3.5-27B-W80K-L0_50
)

module load singularity
cd "$WORKSPACE"
for required in "$SIF" "$MODELS_DIR" "$ACTIVATIONS_ROOT"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
gpu=(); nvidia-smi -L >/dev/null 2>&1 && gpu=(--nv)
echo "device: ${gpu[*]:-cpu only}  cpus: ${SLURM_CPUS_PER_TASK:-?}"

for slug in $MODELS; do
  [[ -d "$MODELS_DIR/${SAE_DIR[$slug]}" ]] || { echo "ERROR: no SAE checkpoints for $slug" >&2; exit 1; }
  echo "===== SAE features: $slug ====="
  singularity exec --cleanenv ${gpu[@]+"${gpu[@]}"} \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models --bind "$ACTIVATIONS_ROOT":/activations \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m mechanistic_analysis.sae.extract \
    --model "$slug" --activation-root /activations --output-root "$OUTPUT_ROOT" \
    --sae-root "/models/${SAE_DIR[$slug]}" --layers "$LAYERS" --sae-dtype float32 --batch-size 1024
done
echo "SAE feature extraction is complete."
