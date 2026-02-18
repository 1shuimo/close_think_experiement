#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_32B="${MODEL_32B:-/scratch-ssd/guoeng/huggingface/models/Qwen3-32B}"
TASK="${TASK:-tom_tracking_0.5k}"
LONGPROC_DATA_PATH="${LONGPROC_DATA_PATH:-../LongProc/data}"
LONGPROC_CODE_PATH="${LONGPROC_CODE_PATH:-../LongProc}"
N_SAMPLES="${N_SAMPLES:-6}"
CHECKPOINT_DELAY="${CHECKPOINT_DELAY:-120}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-1200}"
MAX_NEW_AFTER="${MAX_NEW_AFTER:-400}"
OUT_ROOT="${OUT_ROOT:-suite_longproc_32b}"
PROMPT_BASE_FILE="${PROMPT_BASE_FILE:-prompts/system_base_v1.txt}"
PROMPT_ENH_FILE="${PROMPT_ENH_FILE:-prompts/system_enhanced_v1.txt}"
INJECT_TEXT_FILE="${INJECT_TEXT_FILE:-prompts/inject_think_v1.txt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-prefix-tokens)
      MAX_PREFIX_TOKENS="$2"
      shift 2
      ;;
    --max-new-after)
      MAX_NEW_AFTER="$2"
      shift 2
      ;;
    --checkpoint-delay)
      CHECKPOINT_DELAY="$2"
      shift 2
      ;;
    --n-samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_32B="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash run_longproc_32b.sh [options]
  --max-prefix-tokens N
  --max-new-after N
  --checkpoint-delay N
  --n-samples N
  --task NAME
  --out-root DIR
  --model-path PATH
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

INJECT_TEXT="$(cat "${INJECT_TEXT_FILE}")"

mkdir -p "${OUT_ROOT}"

echo "[LongProc A] baseline"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --longproc-task "${TASK}" \
  --longproc-data-path "${LONGPROC_DATA_PATH}" \
  --longproc-code-path "${LONGPROC_CODE_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --prompt-mode baseline \
  --system-prompt-file "${PROMPT_BASE_FILE}" \
  --checkpoint-mode regex \
  --checkpoint-regex '__auto__' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '__auto__' \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --output-dir "${OUT_ROOT}/baseline" \
  --save-task-texts

echo "[LongProc B] enhanced"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --longproc-task "${TASK}" \
  --longproc-data-path "${LONGPROC_DATA_PATH}" \
  --longproc-code-path "${LONGPROC_CODE_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --prompt-mode enhanced \
  --system-prompt-file "${PROMPT_ENH_FILE}" \
  --inject-text "${INJECT_TEXT}" \
  --checkpoint-mode regex \
  --checkpoint-regex '__auto__' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '__auto__' \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --output-dir "${OUT_ROOT}/enhanced" \
  --save-task-texts

echo "[LongProc C] enhanced + match-cover"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --longproc-task "${TASK}" \
  --longproc-data-path "${LONGPROC_DATA_PATH}" \
  --longproc-code-path "${LONGPROC_CODE_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --prompt-mode enhanced \
  --system-prompt-file "${PROMPT_ENH_FILE}" \
  --inject-text "${INJECT_TEXT}" \
  --apply-match-cover \
  --checkpoint-mode regex \
  --checkpoint-regex '__auto__' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '__auto__' \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --output-dir "${OUT_ROOT}/enhanced_cover" \
  --save-task-texts

echo "Done. Outputs in ${OUT_ROOT}"
