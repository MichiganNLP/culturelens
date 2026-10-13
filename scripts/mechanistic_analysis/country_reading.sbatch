#!/usr/bin/env bash
#SBATCH --job-name=clv2-country-read
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Where the country is read when no norm is stated: knockout by layer window and
# erasure by depth of the country line on the country-only prompt, scored by the
# decision margin (mechanistic_analysis/interventions/country_reading.py).
# Models already done are skipped, so the job can be submitted again.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/mechanistic_analysis/country_reading.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
SAMPLE=${SAMPLE:-400}
HUB=/models/.cache/huggingface/hub
OUT=output/mechanistic_analysis/country_reading
RECORDS=data/processed/culturelens/records.json

module load singularity
cd "$WORKSPACE"
run() {
  local slug=$1 model=$2 batch=$3; shift 3
  if [[ -s "$OUT/$slug/metrics.json" ]]; then echo "$slug: done"; return; fi
  echo "===== $slug ====="
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface --env HF_HUB_OFFLINE=1 \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m mechanistic_analysis.interventions.country_reading \
    --model "$model" --model-slug "$slug" --records "$RECORDS" --sample "$SAMPLE" \
    --output-root "$OUT" --batch-size "$batch" --local-files-only "$@"
}
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}   # which models to run
wanted() { [[ " $MODELS " == *" $1 "* ]]; }
wanted llama-3.1-8b && run llama-3.1-8b "$HUB/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" 32 --window 4 --cuts 16
wanted gemma-3-12b && run gemma-3-12b "$HUB/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80" 24 --window 6 --cuts 12
wanted qwen-3.5-27b && run qwen-3.5-27b "$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654" 16 --cuts 16 --trust-remote-code --no-thinking --sample 200
echo "finished: $MODELS"
