#!/usr/bin/env bash
#SBATCH --job-name=clv2-country-space
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Country geometry and the follow-the-stated-norm probe, from cached states
# (mechanistic_analysis/country_space.py). CPU only. Models already done are skipped.
#
#   sbatch --account=<account> --partition=<cpu-partition> \
#     --export=ALL,WORKSPACE=<repo>,ACTIVATIONS_ROOT=<caches> scripts/mechanistic_analysis/country_space.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
ACTIVATIONS_ROOT=${ACTIVATIONS_ROOT:?Set ACTIVATIONS_ROOT to the activation caches}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
OUT=output/mechanistic_analysis/country_space

module load singularity
cd "$WORKSPACE"
for slug in ${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}; do
  if [[ -s "$OUT/$slug/metrics.json" ]]; then echo "$slug: done"; continue; fi
  echo "===== $slug ====="
  singularity exec --cleanenv --bind "$WORKSPACE":/workspace --bind "$ACTIVATIONS_ROOT":/activations \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m mechanistic_analysis.country_space \
    --activations-root /activations --model-slug "$slug" \
    --records data/processed/culturelens/records.json \
    --judgments-root output/behavioral_evaluation/unconstrained/judgments --output-root "$OUT"
done
