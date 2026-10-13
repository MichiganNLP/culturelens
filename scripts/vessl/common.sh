# Shared by the VESSL job scripts. Source it; do not execute.
#
# Two volumes, because no single one does both jobs:
#   SYNC  object volume  -- the only kind vesslctl can upload to and download
#                           from, so code arrives here and results leave here.
#   WORK  cluster volume -- CephFS on the same cluster as the GPUs. Real POSIX,
#                           so the pipeline's write-temp-then-rename is atomic,
#                           and fast enough to reload model weights every stage.
set -euo pipefail

SYNC="${SYNC:-/sync}"
WORK="${WORK:-/work}"
export HF_HOME="$WORK/hf"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1

GEMMA='google/gemma-4-31B-it'
QWEN='Qwen/Qwen3.6-27B'
# Per-job override: two jobs sharing one tree delete each other's working
# directory when they reinstall it, and the earlier job's next Python start fails.
CODE="${CODE:-$WORK/code}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# Fresh code on every start: it is small, and a restart then runs whatever was
# last uploaded. Live state is outside this tree, so replacing it loses nothing.
install_code() {
  rm -rf "$CODE"
  mkdir -p "$CODE"
  if [[ -f "$SYNC/code/MANIFEST" ]]; then
    # Exactly the files of the last upload: the volume cannot delete, so
    # anything removed locally would otherwise survive here.
    while IFS= read -r f; do
      mkdir -p "$CODE/$(dirname "$f")"
      cp "$SYNC/code/$f" "$CODE/$f"
    done < "$SYNC/code/MANIFEST"
  else
    cp -r "$SYNC/code/." "$CODE/"
  fi
  cd "$CODE"
  # benchmark.common imports pandas, which the vLLM image does not ship.
  # --no-deps so the image's own numpy stays in front of vLLM.
  pip install --quiet --no-deps --root-user-action=ignore \
    "pandas==2.2.3" python-dateutil pytz tzdata pycountry
  export PYTHONPATH="$CODE/src"
}

# Mirror a live tree to the object volume.
#
# The object volume is S3 behind FUSE: a copy interrupted when a container is
# torn down leaves a truncated object under the real name, and a local sync
# then pulls it. So each file is written to <name>.part and renamed into place
# -- an interrupted copy leaves only a .part, which nothing reads -- and only
# files that changed since this container last published are copied, which
# keeps that window small. The record is container-local on purpose: a new
# job republishes everything once, repairing anything the last one left short.
PUBLISHED="/tmp/.published"
publish() {
  local src="$1" dst="$2" f key
  mkdir -p "$dst"; touch "$PUBLISHED"
  while IFS= read -r -d '' f; do
    f="${f#./}"
    key="$dst/$f $(stat -c '%s:%Y' "$src/$f")"
    grep -qxF "$key" "$PUBLISHED" && continue
    mkdir -p "$dst/$(dirname "$f")"
    cp -f "$src/$f" "$dst/$f.part" && mv -f "$dst/$f.part" "$dst/$f" && echo "$key" >> "$PUBLISHED"
  done < <(cd "$src" && find . -type f ! -name '*.tmp' ! -name '*.part' -print0)
  sync
}
