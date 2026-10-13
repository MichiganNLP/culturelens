#!/usr/bin/env bash
#SBATCH --job-name=clv2-sae-describe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Follow-ups to the SAE analysis, CPU only: a description of each layer's most
# predictive feature (mechanistic_analysis/sae/describe.py), then the analysis
# repeated with the second judge's labels.
#
#   sbatch --account=<account> --partition=<partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/mechanistic_analysis/sae/describe.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
ACTIVATIONS_ROOT=${ACTIVATIONS_ROOT:-$STORAGE/culturelens-v2-activations/assistant-start-v1-vessl}
MODELS=${MODELS:-qwen-3.5-27b gemma-3-12b llama-3.1-8b}
HUB=/models/.cache/huggingface/hub
declare -A SAE_DIR=(
  [llama-3.1-8b]=saes-llama-3.1-8b-instruct
  [gemma-3-12b]=gemma-scope-2-12b-it
  [qwen-3.5-27b]=SAE-Res-Qwen3.5-27B-W80K-L0_50
)
declare -A MODEL_DIR=(
  [llama-3.1-8b]=$HUB/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
  [gemma-3-12b]=$HUB/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
  [qwen-3.5-27b]=$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654
)

module load singularity
cd "$WORKSPACE"
run() {
  singularity exec --cleanenv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models --bind "$ACTIVATIONS_ROOT":/activations \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HUB_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /workspace "$SIF" /usr/bin/python3 -u "$@"
}
for slug in $MODELS; do
  echo "===== feature description: $slug ====="
  run -m mechanistic_analysis.sae.describe --model "$slug" \
    --features-root output/mechanistic_analysis/sae/features \
    --analysis-root output/mechanistic_analysis/sae/analysis \
    --sae-root "/models/${SAE_DIR[$slug]}" --activation-root /activations \
    --records data/processed/culturelens/records.json \
    --outcomes "output/mechanistic_analysis/outcomes/$slug.npz" --model-dir "${MODEL_DIR[$slug]}"
done
for slug in $MODELS; do
  echo "===== SAE analysis, second judge: $slug ====="
  run -m mechanistic_analysis.sae.analyze --model "$slug" --judge qwen-3.6-27b --skip-profile \
    --features-root output/mechanistic_analysis/sae/features --activation-root /activations \
    --records data/processed/culturelens/records.json \
    --outcomes "output/mechanistic_analysis/outcomes/$slug.npz" \
    --output-root output/mechanistic_analysis/sae/analysis --draws 300
done
echo "SAE follow-ups are complete."
