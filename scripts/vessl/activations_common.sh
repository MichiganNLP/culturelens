# Shared by the activation-extraction jobs. Source it after common.sh.
#
# Llama-3.1-8B-Instruct and Gemma-3-12B-it are gated on the Hub. Their weight
# shards come from ungated mirrors and are accepted only if byte-identical to
# the copies downloaded from meta-llama and google themselves on the Slurm cluster
# (the hashes below). The small files -- config, tokenizer, chat template -- are
# those authentic copies, uploaded to the sync volume. Qwen is public and is
# pinned to the same revision the Slurm cluster uses.
LLAMA_DIR="$WORK/models/llama-3.1-8b-instruct"
GEMMA3_DIR="$WORK/models/gemma-3-12b-it"
QWEN_REPO="Qwen/Qwen3.5-27B"
QWEN_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
QWEN_DIR="$HF_HOME/hub/models--Qwen--Qwen3.5-27B/snapshots/$QWEN_REVISION"
ACTIVATIONS="$WORK/activations/assistant-start-v1"

LLAMA_SHA256="
2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668  model-00001-of-00004.safetensors
09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15  model-00002-of-00004.safetensors
fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa  model-00003-of-00004.safetensors
92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b  model-00004-of-00004.safetensors
29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e  config.json
189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd  generation_config.json
146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b  model.safetensors.index.json
79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4  tokenizer.json
177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424  tokenizer_config.json
"
GEMMA3_SHA256="
4847447e92599833e8dbaa3067cd201c3bb5c052efa91f11ba891e43234f7832  model-00001-of-00005.safetensors
891bd54eed03cba9ee1e705533a02a8217fcc29f356e4a1f53e5fd0d178883ad  model-00002-of-00005.safetensors
7cee411d9d57324e50ce064a192cc5a858276d508611b12fc599e0c9767112e0  model-00003-of-00005.safetensors
8bc75a29a730c9e743cad013feda3b0991a913fafe787c58a1c6e20afad97723  model-00004-of-00005.safetensors
ed14bd4908c98fed9f61e8cd410167e0846de9abd78e0452ab092072e5d9252d  model-00005-of-00005.safetensors
50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946  added_tokens.json
fe16baf728db49457cde32802cd7efc0ac8a7a9877dbe22fe3322b2d9dc6ccd9  chat_template.json
39172c4124d3470341bbbb25f2926fd97edf68f0fe3a9fa4cde6acb9b7ed2cc6  config.json
fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3  generation_config.json
788cc42a1a92835df62d9a3791f47105f63504c7c404637a73288e9b11bc7b82  model.safetensors.index.json
f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d  preprocessor_config.json
3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53  processor_config.json
2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397  special_tokens_map.json
bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1  tokenizer_config.json
4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795  tokenizer.json
"

verify_model() {  # dir sha-list
  [[ -d "$1" ]] || return 1
  (cd "$1" && printf '%s\n' "$2" | sed '/^$/d' | sha256sum --check --quiet)
}

# mirror-repo shard-glob target-dir small-files-dir sha-list
assemble_model() {
  local mirror="$1" pattern="$2" target="$3" small="$4" hashes="$5"
  if verify_model "$target" "$hashes"; then log "verified already: $target"; return 0; fi
  log "assembling $target from $mirror"
  mkdir -p "$target"
  python3 - "$mirror" "$pattern" "$target" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, pattern, target = sys.argv[1:4]
# local_dir writes real files into the target, so the shards are stored once.
snapshot_download(repo, allow_patterns=[pattern], local_dir=target)
PY
  cp -f "$small"/* "$target/"
  verify_model "$target" "$hashes" || { log "VERIFICATION FAILED for $target -- refusing to continue"; exit 1; }
  log "assembled and verified: $target"
}
