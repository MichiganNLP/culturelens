# Shared by the fine-tuning jobs. Source it after common.sh.
#
# Llama-3.1-8B-Instruct is gated on the Hub. Its weight shards are taken from
# an ungated mirror and accepted only if byte-identical to Meta's -- the hashes
# below are of the copy downloaded from meta-llama itself on the Slurm cluster. The
# small files (config, tokenizer and its chat template) are that authentic
# copy, uploaded to the sync volume, because the mirrors' differ from Meta's
# current revision. The assembled directory is Meta's snapshot 0e9e39f2.
LLAMA_DIR="$WORK/models/llama-3.1-8b-instruct"
LLAMA_MIRROR="NousResearch/Meta-Llama-3.1-8B-Instruct"
GEMMA_JUDGE="google/gemma-4-31B-it"
SPLITS="$SYNC/finetune/data"

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

verify_llama() {
  [[ -d "$LLAMA_DIR" ]] || return 1
  (cd "$LLAMA_DIR" && printf '%s\n' "$LLAMA_SHA256" | sed '/^$/d' | sha256sum --check --quiet)
}

# peft is not in the vLLM image. Installed against the image's own versions:
# new packages may be added, nothing already installed may change.
install_peft() {
  pip list --format=freeze > /tmp/image-constraints.txt
  pip install --quiet --root-user-action=ignore -c /tmp/image-constraints.txt peft
  python3 -c "import peft, torch, transformers; print('peft', peft.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
}

# Mirror a directory to the sync volume without touching files already there.
mirror() {
  local src="$1" dst="$2"
  [[ -d "$src" ]] || return 0
  publish "$src" "$dst"
}
