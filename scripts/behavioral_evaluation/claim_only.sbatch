#!/usr/bin/env bash
#SBATCH --job-name=clv2-claim-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/%x-%j.log
set -euo pipefail

# The false norm with no country line (diagnostic condition opposite_norm_only),
# for every record and all three models: open-ended answers, then the primary
# judge. Compared with country_opposite_norm, it shows what naming the country
# does to a false claim. Answers and judgments are kept apart from the
# benchmark's own files; both steps resume from what is already written.
#
#   sbatch --account=<account> --partition=<gpu-partition> \
#     --export=ALL,WORKSPACE=<repo> scripts/behavioral_evaluation/claim_only.sbatch
#
# MODELS (space-separated slugs) limits the run to some models; JUDGE_ONLY=1
# skips generation and judges the answers already written for them.
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the synced CultureLens v2 directory}
STORAGE=${CULTURELENS_STORAGE_ROOT:?Set CULTURELENS_STORAGE_ROOT to the directory that holds models and caches}
SIF=${VLLM_SIF:-$STORAGE/vllm_official.sif}
MODELS_DIR=${MODELS_DIR:-$STORAGE/models}
DATASET=data/processed/culturelens/records.json
ROOT=output/behavioral_evaluation/unconstrained
CONDITION=opposite_norm_only
MODELS=${MODELS:-llama-3.1-8b gemma-3-12b qwen-3.5-27b}
JUDGE_ONLY=${JUDGE_ONLY:-0}
HUB=/models/.cache/huggingface/hub

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

generate() {  # model slug batch extra...
  local model=$1 slug=$2 batch=$3; shift 3
  [[ "$JUDGE_ONLY" != 1 && " $MODELS " == *" $slug "* ]] || return 0
  echo "===== $CONDITION: $slug ====="
  infer -m behavioral_evaluation.generate --model "$model" --input "$DATASET" \
    --output "$ROOT/responses/$slug.$CONDITION.jsonl" --conditions "$CONDITION" \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --batch-size "$batch" --max-model-len 8192 --max-tokens 512 --seed 42 \
    --disable-custom-all-reduce "$@"
}
generate "$HUB/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" llama-3.1-8b 32
generate "$HUB/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80" gemma-3-12b 32
generate "$HUB/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654" qwen-3.5-27b 16 \
  --trust-remote-code --no-think --language-model-only --enforce-eager

echo "===== judge: gemma-4-31b ====="
responses=()
for slug in $MODELS; do
  responses+=(--response "$slug.$CONDITION=$ROOT/responses/$slug.$CONDITION.jsonl")
done
infer -m behavioral_evaluation.judge \
  --model "$HUB/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475" \
  --judge-slug gemma-4-31b --dataset "$DATASET" "${responses[@]}" \
  --output-root "$ROOT/judgments" --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --batch-size 64 --max-model-len 8192 --max-tokens 1536 \
  --retries 0 --seed 42 --progress-every 10 \
  --language-model-only --disable-custom-all-reduce --enforce-eager \
  || echo "some answers could not be judged; rerun to retry them"
echo "claim-only condition is complete."
