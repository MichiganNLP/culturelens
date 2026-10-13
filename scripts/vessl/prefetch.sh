#!/usr/bin/env bash
# CPU job: prove the environment, then fill the model cache.
#
# Everything that can fail without a GPU fails here, at CPU prices: the image
# accepting --cmd, both volume mounts, rename semantics, the package imports.
# The two checkpoints (~117 GB) are then downloaded once into the cluster
# volume, so no GPU hour is ever spent waiting on a download.
source "$(dirname "$0")/common.sh"

log "python/vllm"
# Package metadata, not `import vllm`: a CUDA build can refuse to import on a
# CPU node, and the GPU job is where vLLM itself gets exercised.
python3 -c "import sys, importlib.metadata as m; print(sys.version.split()[0], 'vllm', m.version('vllm'))"
nproc; free -g | head -2

log "volumes"
for d in "$SYNC" "$WORK"; do
  echo probe > "$d/.probe" && mv "$d/.probe" "$d/.probe2" && rm "$d/.probe2"
  echo "  rename ok: $d"
done
df -h "$WORK" | tail -1

log "code + imports"
install_code
python3 - <<'PY'
import importlib, numpy, pandas
print("numpy", numpy.__version__, "pandas", pandas.__version__)
for m in ("benchmark.generate", "benchmark.validate_norms",
          "benchmark.merge_value_tags", "benchmark.validate_scenarios",
          "benchmark.validate_options"):
    importlib.import_module(m)
from benchmark.common import load_countries
print("countries", len(load_countries("configs/countries.yaml")))
PY

log "model cache -> $HF_HOME"
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("google/gemma-4-31B-it", "Qwen/Qwen3.6-27B"):
    path = snapshot_download(repo, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    print("cached", repo, "->", path, flush=True)
PY
du -sh "$HF_HOME"

mkdir -p "$SYNC/done"
date -u +%FT%TZ > "$SYNC/done/prefetch"
log "PREFETCH OK"
