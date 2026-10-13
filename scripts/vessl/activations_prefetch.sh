#!/usr/bin/env bash
# CPU job: assemble and verify the three models and prove the extractor
# imports, so no GPU hour is spent on a download or an ImportError.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/activations_common.sh"

install_code
[[ -s "$SYNC/activations/data/records.json" ]] || { log "missing records.json on the sync volume"; exit 1; }

# device_map="balanced" needs accelerate. Installed against the image's own
# versions if absent: nothing already present may change.
python3 -c "import accelerate" 2>/dev/null || {
  pip list --format=freeze > /tmp/image-constraints.txt
  pip install --quiet --root-user-action=ignore -c /tmp/image-constraints.txt accelerate
}
python3 -c "import accelerate, torch, transformers; print('accelerate', accelerate.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
python3 -m mechanistic_analysis.activations.extract --help > /dev/null && log "extractor imports ok"

assemble_model "NousResearch/Meta-Llama-3.1-8B-Instruct" "model-0000*-of-00004.safetensors" \
  "$LLAMA_DIR" "$SYNC/models/llama-3.1-8b-instruct-meta" "$LLAMA_SHA256"
assemble_model "unsloth/gemma-3-12b-it" "model-0000*-of-00005.safetensors" \
  "$GEMMA3_DIR" "$SYNC/models/gemma-3-12b-it-google" "$GEMMA3_SHA256"

log "caching $QWEN_REPO @ $QWEN_REVISION"
python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$QWEN_REPO', revision='$QWEN_REVISION'))"
[[ -f "$QWEN_DIR/config.json" ]] || { log "Qwen snapshot not at $QWEN_DIR"; exit 1; }

du -sh "$WORK/models" "$HF_HOME" 2>/dev/null; df -h "$WORK" | tail -1
mkdir -p "$SYNC/done"; date -u +%FT%TZ > "$SYNC/done/activations_prefetch"
log "ACTIVATIONS PREFETCH OK"
