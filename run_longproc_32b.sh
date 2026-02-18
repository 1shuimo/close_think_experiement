#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_32B="${MODEL_32B:-/scratch-ssd/guoeng/huggingface/models/Qwen3-32B}"
TASK="${TASK:-tom_tracking_0.5k}"
LONGPROC_DATA_PATH="${LONGPROC_DATA_PATH:-../LongProc/data}"
LONGPROC_CODE_PATH="${LONGPROC_CODE_PATH:-../LongProc}"
N_SAMPLES="${N_SAMPLES:-6}"
OUT_ROOT="${OUT_ROOT:-suite_longproc_32b}"
PROMPT_BASE_FILE="${PROMPT_BASE_FILE:-prompts/system_base_v1.txt}"
PROMPT_ENH_FILE="${PROMPT_ENH_FILE:-prompts/system_enhanced_v1.txt}"
INJECT_TEXT_FILE="${INJECT_TEXT_FILE:-prompts/inject_think_v1.txt}"
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
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
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
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
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
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir "${OUT_ROOT}/enhanced_cover" \
  --save-task-texts

echo "Done. Outputs in ${OUT_ROOT}"
