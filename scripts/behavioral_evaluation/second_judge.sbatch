#!/usr/bin/env bash
#SBATCH --job-name=clv2-second-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# The second judge (Qwen-3.6-27B) on the diagnostic answers the primary judge
# has scored: the framing experiment, then the false norm without the country
# on the full benchmark. Resumes from what is already written. PARTS limits
# the run to "framing" or "claim-only".
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/behavioral_evaluation/second_judge.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
ROOT=output/behavioral_evaluation/unconstrained
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
PARTS=${PARTS:-framing claim-only}
JUDGE=/models/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9

module load singularity
cd "$WORKSPACE"
[[ -s "$DATASET" && -s "$SIF" ]] || { echo "ERROR: missing dataset or image" >&2; exit 1; }

judge() {  # output-root responses...
  local root=$1; shift
  [[ $# -gt 0 ]] || { echo "no answers to judge for $root"; return 0; }
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m behavioral_evaluation.judge \
    --model "$JUDGE" --judge-slug qwen-3.6-27b --dataset "$DATASET" "$@" --output-root "$root" \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --batch-size 64 --max-model-len 8192 --max-tokens 1536 --retries 0 --seed 42 \
    --progress-every 10 --trust-remote-code --no-think --language-model-only \
    --disable-custom-all-reduce --enforce-eager \
    || echo "some answers could not be judged; rerun to retry them"
}

framing=() claim=()
for slug in $MODELS; do
  file=$ROOT/framing/responses/$slug.jsonl
  [[ -s "$file" ]] && framing+=(--response "$slug=$file")
  file=$ROOT/responses/$slug.opposite_norm_only.jsonl
  [[ -s "$file" ]] && claim+=(--response "$slug.opposite_norm_only=$file")
done
if [[ " $PARTS " == *" framing "* ]]; then
  echo "===== second judge: framing ====="
  judge "$ROOT/framing/judgments" ${framing[@]+"${framing[@]}"}
fi
if [[ " $PARTS " == *" claim-only "* ]]; then
  echo "===== second judge: false norm without the country ====="
  judge "$ROOT/judgments" ${claim[@]+"${claim[@]}"}
fi
echo "second judge is complete."
