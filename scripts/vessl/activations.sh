#!/usr/bin/env bash
# GPU job: assistant-start residual activations, one set of conditions, three models.
#
# The extractor is called with exactly the arguments of
# scripts/mechanistic_analysis/activations/extract_primary.sbatch, so a cache
# written here is the cache that job would have written. Extraction resumes
# from its own checkpoint shards, so a restarted job loses at most one shard.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/activations_common.sh"

# Override to extract other conditions, e.g. the claim controls:
#   CONDITIONS="norm_only opposite_norm_only shuffled_country_norm shuffled_country_opposite_norm"
read -r -a CONDITIONS <<< "${CONDITIONS:-base country_only country_norm country_opposite_norm}"
DEST="$SYNC/activations/assistant-start-v1"

install_code
python3 -c "import accelerate" 2>/dev/null || {
  pip list --format=freeze > /tmp/image-constraints.txt
  pip install --quiet --root-user-action=ignore -c /tmp/image-constraints.txt accelerate
}
verify_model "$LLAMA_DIR" "$LLAMA_SHA256" || { log "llama not verified; run activations_prefetch.sh"; exit 1; }
verify_model "$GEMMA3_DIR" "$GEMMA3_SHA256" || { log "gemma not verified; run activations_prefetch.sh"; exit 1; }
[[ -f "$QWEN_DIR/config.json" ]] || { log "qwen not cached; run activations_prefetch.sh"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
log "conditions: ${CONDITIONS[*]}"
df -h "$WORK" | tail -1
mkdir -p "$ACTIVATIONS" "$SYNC/done"

# Only finished caches are published: the .parts shard directories are working
# state, removed by the extractor once a condition's file is complete.
publish_finished() {
  local f rel
  while IFS= read -r -d '' f; do
    rel="${f#"$ACTIVATIONS"/}"
    mkdir -p "$DEST/$(dirname "$rel")"
    if [[ ! -f "$DEST/$rel" || "$(stat -c %s "$f")" != "$(stat -c %s "$DEST/$rel" 2>/dev/null)" ]]; then
      cp -f "$f" "$DEST/$rel.part" && mv -f "$DEST/$rel.part" "$DEST/$rel"
      log "published $rel ($(du -h "$f" | cut -f1))"
    fi
  done < <(find "$ACTIVATIONS" -type f -not -path '*.parts/*' -print0)
}

extract_model() {
  local model=$1 slug=$2 batch_size=$3; shift 3
  log "===== extracting $slug ====="
  python3 -u -m mechanistic_analysis.activations.extract \
    --model "$model" --model-slug "$slug" \
    --input "$SYNC/activations/data/records.json" --output-root "$ACTIVATIONS" \
    --variant unconstrained --conditions "${CONDITIONS[@]}" \
    --dtype bfloat16 --device-map balanced --batch-size "$batch_size" \
    --checkpoint-size 128 --max-length 8192 --placebo-seed 42 --local-files-only "$@"
  publish_finished
  date -u +%FT%TZ > "$SYNC/done/activations.$slug"
}

extract_model "$LLAMA_DIR" llama-3.1-8b 16
extract_model "$GEMMA3_DIR" gemma-3-12b 8
extract_model "$QWEN_DIR" qwen-3.5-27b 4 --trust-remote-code --no-thinking

find "$ACTIVATIONS" -type f -name '*.pt' -not -path '*.parts/*' -exec du -h {} \;
du -sh "$ACTIVATIONS"
date -u +%FT%TZ > "$SYNC/done/ACTIVATIONS"
log "ALL ACTIVATIONS DONE"
