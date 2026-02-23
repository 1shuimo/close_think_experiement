#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_8B="${MODEL_8B:-autodl-tmp/Qwen/Qwen3-8B}"
TASKS_FILE="${TASKS_FILE:-tasks_math_mix5_corrupt.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-suite_math_8b}"
MODE="${MODE:-corrupt}"   # corrupt | clean
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-prompts/system_enhanced_v1.txt}"
INJECT_TEXT_FILE="${INJECT_TEXT_FILE:-prompts/inject_think_v2.txt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_8B="$2"; shift 2 ;;
    --tasks-file) TASKS_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --inject-text-file) INJECT_TEXT_FILE="$2"; shift 2 ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash run_math_8b.sh [options]
  --mode corrupt|clean
  --model-path PATH
  --tasks-file FILE
  --output-dir DIR
  --system-prompt-file FILE
  --inject-text-file FILE
USAGE
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

INJECT_TEXT="$(cat "${INJECT_TEXT_FILE}")"

if [[ "${MODE}" == "corrupt" ]]; then
  WORKFLOW="math_corrupt"
elif [[ "${MODE}" == "clean" ]]; then
  WORKFLOW="math_clean"
else
  echo "Invalid --mode: ${MODE} (expected corrupt|clean)" >&2
  exit 1
fi

python test_close_suite.py \
  --workflow "${WORKFLOW}" \
  --model-paths "${MODEL_8B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode enhanced \
  --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
  --inject-text "${INJECT_TEXT}" \
  --save-task-texts \
  --print-full-output \
  --output-dir "${OUTPUT_DIR}"
