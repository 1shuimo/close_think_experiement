#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_32B="${MODEL_32B:-/scratch-ssd/guoeng/huggingface/models/Qwen3-32B}"
N_SAMPLES="${N_SAMPLES:-20}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-suite_longproc_32b_3tasks_${RUN_TAG}}"

TASKS=(
  "tom_tracking_0.5k"
  "pseudo_to_code_0.5k"
  "path_traversal_0.5k"
)

for TASK in "${TASKS[@]}"; do
  echo "===== Running task: ${TASK} ====="
  bash run_longproc_32b.sh \
    --model-path "${MODEL_32B}" \
    --task "${TASK}" \
    --n-samples "${N_SAMPLES}" \
    --branch-mode ab \
    --checkpoint-mode think_end_then_regex \
    --checkpoint-regex "__auto__" \
    --corrupt-anchor-regex "__auto__" \
    --max-prefix-tokens 2000 \
    --max-new-after 1200 \
    --out-root "${OUT_ROOT_BASE}/${TASK}"
done

echo "Done. Outputs in ${OUT_ROOT_BASE}"
