#!/usr/bin/env bash
# CPU job: assemble a verified Llama-3.1-8B-Instruct, cache the Gemma judge,
# and prove the fine-tuning stack imports -- all before any GPU hour is spent.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/finetune_common.sh"

install_code
install_peft
python3 -c "import finetune.grpo, finetune.losses, behavioral_evaluation.judge; print('finetune imports ok')"
for f in train validation test_seen test_unseen; do
  [[ -s "$SPLITS/$f.json" ]] || { log "missing $SPLITS/$f.json"; exit 1; }
done
[[ -s "$SPLITS/records.json" ]] || { log "missing $SPLITS/records.json"; exit 1; }

if verify_llama; then
  log "llama already assembled and verified"
else
  log "assembling llama from $LLAMA_MIRROR"
  mkdir -p "$LLAMA_DIR"
  python3 - "$LLAMA_MIRROR" "$LLAMA_DIR" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, target = sys.argv[1], Path(sys.argv[2])
path = Path(snapshot_download(repo, allow_patterns=["model-0000*-of-00004.safetensors"]))
for shard in sorted(path.glob("model-0000*-of-00004.safetensors")):
    shutil.copyfile(shard.resolve(), target / shard.name)
    print("copied", shard.name, flush=True)
PY
  cp -f "$SYNC/models/llama-3.1-8b-instruct-meta/"* "$LLAMA_DIR/"
  verify_llama || { log "LLAMA VERIFICATION FAILED -- refusing to continue"; exit 1; }
  log "llama assembled: all 9 files match Meta's snapshot"
fi

log "caching $GEMMA_JUDGE"
python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$GEMMA_JUDGE', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','tokenizer*']))"
du -sh "$WORK/models" "$HF_HOME" 2>/dev/null
mkdir -p "$SYNC/done"; date -u +%FT%TZ > "$SYNC/done/finetune_prefetch"
log "FINETUNE PREFETCH OK"
