#!/usr/bin/env bash
#SBATCH --job-name=clv2-stance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# Code how answers treat the stated false norm (behavioral_evaluation/stance.py):
# first the answers written under the mechanistic interventions, then every
# model's answers to country + false norm on the full benchmark. Resumes from
# what is already written. PARTS limits the run to "arms" or "full".
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/behavioral_evaluation/stance.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
PARTS=${PARTS:-arms full}
ARMS=${ARMS:-false false-noclaim false-noclaim-win false-noclaim-heads false-erase-first false-erase-before claim-only country}
CODER=/models/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475

module load singularity
cd "$WORKSPACE"
[[ -s "$DATASET" && -s "$SIF" ]] || { echo "ERROR: missing dataset or image" >&2; exit 1; }

arms=() full=()
for slug in $MODELS; do
  for arm in $ARMS; do
    file=output/mechanistic_analysis/answers/$slug/$arm.jsonl
    [[ -s "$file" ]] && arms+=(--response "$slug.$arm=$file")
  done
  file=output/behavioral_evaluation/unconstrained/responses/$slug.jsonl
  [[ -s "$file" ]] && full+=(--response "$slug=$file")
done

code() {  # output-root conditions... -- responses...
  local root=$1; shift
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface \
    --pwd /workspace "$SIF" /usr/bin/python3 -u -m behavioral_evaluation.stance \
    --model "$CODER" --coder-slug gemma-4-31b --dataset "$DATASET" --output-root "$root" \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --batch-size 64 --max-model-len 8192 --max-tokens 512 --seed 42 --progress-every 10 \
    --language-model-only --disable-custom-all-reduce --enforce-eager "$@" \
    || echo "some answers could not be coded; rerun to retry them"
}

if [[ " $PARTS " == *" arms "* ]]; then
  echo "===== stance: answers under intervention ====="
  code output/mechanistic_analysis/stance \
    --conditions country_opposite_norm opposite_norm_only country_only "${arms[@]}"
fi
if [[ " $PARTS " == *" full "* ]]; then
  echo "===== stance: full benchmark, country + false norm ====="
  code output/behavioral_evaluation/unconstrained/stance \
    --conditions country_opposite_norm "${full[@]}"
fi
echo "stance coding is complete."
