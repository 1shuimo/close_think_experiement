#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_8B="${MODEL_8B:-/autodl-tmp/Qwen/Qwen3-8B}"
OUT_ROOT_8B="${OUT_ROOT_8B:-suite_longproc_8b}"

_resolve_model_path() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    echo "${p}"
    return 0
  fi
  local cands=("/${p}" "/root/${p}")
  local c
  for c in "${cands[@]}"; do
    if [[ -d "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

if RESOLVED_MODEL="$(_resolve_model_path "${MODEL_8B}")"; then
  MODEL_8B="${RESOLVED_MODEL}"
else
  echo "Model path not found: ${MODEL_8B}" >&2
  echo "Try one of: /autodl-tmp/Qwen/Qwen3-8B or /root/autodl-tmp/Qwen/Qwen3-8B" >&2
  exit 1
fi

# Forward all options to the main runner; user args can still override defaults
# because they are appended at the end.
bash run_longproc_32b.sh \
  --model-path "${MODEL_8B}" \
  --out-root "${OUT_ROOT_8B}" \
  "$@"
