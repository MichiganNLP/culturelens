#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:?Set WORKSPACE to the repository root}
SCRIPT_DIR=$WORKSPACE/scripts/behavioral_evaluation

generation=$(sbatch --parsable \
  --export=ALL,WORKSPACE="$WORKSPACE" \
  "$SCRIPT_DIR/generate.sbatch")

gemma=$(sbatch --parsable --dependency="afterok:$generation" \
  --export=ALL,WORKSPACE="$WORKSPACE",JUDGE=gemma \
  "$SCRIPT_DIR/judge.sbatch")

qwen=$(sbatch --parsable --dependency="afterok:$generation" \
  --export=ALL,WORKSPACE="$WORKSPACE",JUDGE=qwen \
  "$SCRIPT_DIR/judge.sbatch")

printf 'generation\t%s\n' "$generation"
printf 'gemma_judge\t%s\n' "$gemma"
printf 'qwen_judge\t%s\n' "$qwen"
