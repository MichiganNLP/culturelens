#!/usr/bin/env bash
#SBATCH --job-name=clv2-framing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# The same false norm under four framings (attributed to hearsay, attributed to
# a guide, followed by the user's doubt, placed before the country line), on the
# confirmation sample of the mechanistic study: open-ended answers, the primary
# judge, and the stance coding. The bare framing is the benchmark's own
# country_opposite_norm. Every step resumes from what is already written.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/behavioral_evaluation/framing.sbatch
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
SAMPLES=data/processed/culturelens/samples
ROOT=output/behavioral_evaluation/unconstrained/framing
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
CONDITIONS="country_opposite_norm_hearsay country_opposite_norm_guide country_opposite_norm_doubt opposite_norm_country"
HUB=/models/.cache/huggingface/hub
JUDGE=$HUB/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475

module load singularity
cd "$WORKSPACE"
[[ -s "$DATASET" && -s "$SIF" ]] || { echo "ERROR: missing dataset or image" >&2; exit 1; }

infer() {
  singularity exec --cleanenv --nv \
    --bind "$WORKSPACE":/workspace --bind "$MODELS_DIR":/models \
    --env PYTHONPATH=/workspace/src --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=/models/.cache/huggingface \
    --pwd /workspace "$SIF" /usr/bin/python3 -u "$@"
}

generate() {  # model slug sample batch extra...
  local model=$1 slug=$2 sample=$3 batch=$4; shift 4
  [[ " $MODELS " == *" $slug "* ]] || return 0
  echo "===== framing answers: $slug ====="
  # shellcheck disable=SC2086
  infer -m behavioral_evaluation.generate --model "$model" --input "$SAMPLES/$sample.json" \
    --output "$ROOT/responses/$slug.jsonl" --conditions $CONDITIONS \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --batch-size "$batch" --max-model-len 8192 --max-tokens 512 --seed 42 \
    --disable-custom-all-reduce "$@"
}
generate "$HUB/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" llama-3.1-8b confirmation_400 32
generate "$HUB/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80" gemma-3-12b confirmation_400 32
generate "$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654" qwen-3.5-27b confirmation_200 16 \
  --trust-remote-code --no-think --language-model-only --enforce-eager

responses=()
for slug in $MODELS; do
  [[ -s "$ROOT/responses/$slug.jsonl" ]] && responses+=(--response "$slug=$ROOT/responses/$slug.jsonl")
done
common=(--dataset "$DATASET" --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90
        --batch-size 64 --max-model-len 8192 --seed 42 --progress-every 10
        --language-model-only --disable-custom-all-reduce --enforce-eager)

echo "===== judge: gemma-4-31b ====="
infer -m behavioral_evaluation.judge --model "$JUDGE" --judge-slug gemma-4-31b "${responses[@]}" \
  --output-root "$ROOT/judgments" --max-tokens 1536 --retries 0 "${common[@]}" \
  || echo "some answers could not be judged; rerun to retry them"

echo "===== stance: gemma-4-31b ====="
# shellcheck disable=SC2086
infer -m behavioral_evaluation.stance --model "$JUDGE" --coder-slug gemma-4-31b "${responses[@]}" \
  --conditions $CONDITIONS --output-root "$ROOT/stance" --max-tokens 512 "${common[@]}" \
  || echo "some answers could not be coded; rerun to retry them"
echo "framing experiment is complete."
