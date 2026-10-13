#!/usr/bin/env bash
#SBATCH --job-name=clv2-sae-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Analyses of the extracted SAE features (mechanistic_analysis/sae/analyze.py):
# sparse readouts of resisting a contradictory norm, features that register the
# conflict between country and norm, and an audit of the earlier country-by-value
# profile. CPU only. Each layer is saved as it finishes, so a job that ran out
# of time can be submitted again and continues.
#
#   sbatch --account=<account> --partition=<partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/mechanistic_analysis/sae/analyze.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
ACTIVATIONS_ROOT=${ACTIVATIONS_ROOT:-$STORAGE/culturelens-v2-activations/assistant-start-v1-vessl}
FEATURES_ROOT=${FEATURES_ROOT:-output/mechanistic_analysis/sae/features}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/mechanistic_analysis/sae/analysis}
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
DRAWS=${DRAWS:-300}

module load singularity
cd "$WORKSPACE"
for slug in $MODELS; do
  echo "===== SAE analysis: $slug ====="
  singularity exec --cleanenv \
    --bind "$WORKSPACE":/workspace --bind "$ACTIVATIONS_ROOT":/activations \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" --env MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m mechanistic_analysis.sae.analyze \
    --model "$slug" --features-root "$FEATURES_ROOT" --activation-root /activations \
    --records data/processed/culturelens/records.json \
    --outcomes "output/mechanistic_analysis/outcomes/$slug.npz" \
    --output-root "$OUTPUT_ROOT" --draws "$DRAWS"
done
echo "SAE analysis is complete."
