#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 用法:
#   MODEL_8B="autodl-tmp/Qwen/Qwen3-8B" \
#   MODEL_32B="/scratch-ssd/guoeng/huggingface/models/Qwen3-32B" \
#   bash close/run_test_matrix.sh

MODEL_8B="${MODEL_8B:-autodl-tmp/Qwen/Qwen3-8B}"
MODEL_32B="${MODEL_32B:-/scratch-ssd/guoeng/huggingface/models/Qwen3-32B}"

OUT_ROOT="${OUT_ROOT:-suite_runs}"
TASKS_FILE="${TASKS_FILE:-tasks_math_steps.jsonl}"
mkdir -p "${OUT_ROOT}"

echo "[Test 1] 单例人工改错 + 分叉"
python test_branch_manual.py \
  --model-path "${MODEL_8B}" \
  --prompt-mode enhanced \
  --output-dir "${OUT_ROOT}/manual_8b"

echo "[Test 2] 8B baseline prompt（无 match-cover）"
python test_close_suite.py \
  --model-paths "${MODEL_8B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode baseline \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 200 \
  --output-dir "${OUT_ROOT}/suite_8b_baseline"

echo "[Test 3] 8B enhanced prompt（无 match-cover）"
python test_close_suite.py \
  --model-paths "${MODEL_8B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode enhanced \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 200 \
  --output-dir "${OUT_ROOT}/suite_8b_enhanced"

echo "[Test 4] 8B enhanced prompt + match-cover"
python test_close_suite.py \
  --model-paths "${MODEL_8B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode enhanced \
  --apply-match-cover \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 200 \
  --output-dir "${OUT_ROOT}/suite_8b_enhanced_cover"

echo "[Test 5] 32B enhanced prompt（无 match-cover）"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode enhanced \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 200 \
  --output-dir "${OUT_ROOT}/suite_32b_enhanced"

echo "[Test 6] 32B enhanced prompt + match-cover"
python test_close_suite.py \
  --model-paths "${MODEL_32B}" \
  --tasks-file "${TASKS_FILE}" \
  --prompt-mode enhanced \
  --apply-match-cover \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 200 \
  --output-dir "${OUT_ROOT}/suite_32b_enhanced_cover"

echo "Done. Outputs in ${OUT_ROOT}"
