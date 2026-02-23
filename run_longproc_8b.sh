#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_8B="${MODEL_8B:-autodl-tmp/Qwen/Qwen3-8B}"
OUT_ROOT_8B="${OUT_ROOT_8B:-suite_longproc_8b}"

# Forward all options to the main runner; user args can still override defaults
# because they are appended at the end.
bash run_longproc_32b.sh \
  --model-path "${MODEL_8B}" \
  --out-root "${OUT_ROOT_8B}" \
  "$@"
