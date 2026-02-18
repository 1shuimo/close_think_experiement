#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_32B="${MODEL_32B:-/scratch-ssd/guoeng/huggingface/models/Qwen3-32B}"
TASK="${TASK:-tom_tracking_0.5k}"
LONGPROC_DATA_PATH="${LONGPROC_DATA_PATH:-../bench/LongProc/data}"
LONGPROC_CODE_PATH="${LONGPROC_CODE_PATH:-../bench/LongProc}"
N_SAMPLES="${N_SAMPLES:-6}"
OUT_ROOT="${OUT_ROOT:-suite_longproc_32b}"

mkdir -p "${OUT_ROOT}"

echo "[LongProc A] baseline"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --longproc-task "${TASK}" \
  --longproc-data-path "${LONGPROC_DATA_PATH}" \
  --longproc-code-path "${LONGPROC_CODE_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --prompt-mode baseline \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
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
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
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
  --apply-match-cover \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir "${OUT_ROOT}/enhanced_cover" \
  --save-task-texts

echo "Done. Outputs in ${OUT_ROOT}"
