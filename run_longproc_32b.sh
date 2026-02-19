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
CHECKPOINT_MODE="${CHECKPOINT_MODE:-regex}"
CHECKPOINT_REGEX="${CHECKPOINT_REGEX:-__auto__}"
CHECKPOINT_MID_MIN_TOKENS="${CHECKPOINT_MID_MIN_TOKENS:-80}"
CHECKPOINT_MID_MAX_TOKENS="${CHECKPOINT_MID_MAX_TOKENS:-220}"
CHECKPOINT_MID_AVOID_FINAL_REGEX="${CHECKPOINT_MID_AVOID_FINAL_REGEX:-(?i)\\bfinal\\s*:|\\bfinal answer\\b}"
CORRUPT_ANCHOR_REGEX="${CORRUPT_ANCHOR_REGEX:-__auto__}"
CORRUPT_MODE="${CORRUPT_MODE:-anchor_number_shift}"
CORRUPT_AFTER_FIRST_THINK="${CORRUPT_AFTER_FIRST_THINK:-0}"
CORRUPT_PREFER_SIGN_FLIP="${CORRUPT_PREFER_SIGN_FLIP:-0}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-1200}"
MAX_NEW_AFTER="${MAX_NEW_AFTER:-400}"
MIN_B_TOKENS_BEFORE_EOS="${MIN_B_TOKENS_BEFORE_EOS:-64}"
B_RETRY_TIMES="${B_RETRY_TIMES:-2}"
AUTO_CLOSE_UNCLOSED_THINK="${AUTO_CLOSE_UNCLOSED_THINK:-0}"
APPLY_CROSS_THINK_COVER="${APPLY_CROSS_THINK_COVER:-1}"
OUT_ROOT="${OUT_ROOT:-suite_longproc_32b}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
BRANCH_MODE="${BRANCH_MODE:-ab}"
PRINT_FULL_OUTPUT="${PRINT_FULL_OUTPUT:-0}"
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
    --checkpoint-mode)
      CHECKPOINT_MODE="$2"
      shift 2
      ;;
    --checkpoint-regex)
      CHECKPOINT_REGEX="$2"
      shift 2
      ;;
    --checkpoint-mid-min-tokens)
      CHECKPOINT_MID_MIN_TOKENS="$2"
      shift 2
      ;;
    --checkpoint-mid-max-tokens)
      CHECKPOINT_MID_MAX_TOKENS="$2"
      shift 2
      ;;
    --checkpoint-mid-avoid-final-regex)
      CHECKPOINT_MID_AVOID_FINAL_REGEX="$2"
      shift 2
      ;;
    --corrupt-anchor-regex)
      CORRUPT_ANCHOR_REGEX="$2"
      shift 2
      ;;
    --corrupt-mode)
      CORRUPT_MODE="$2"
      shift 2
      ;;
    --corrupt-after-first-think)
      CORRUPT_AFTER_FIRST_THINK=1
      shift 1
      ;;
    --corrupt-prefer-sign-flip)
      CORRUPT_PREFER_SIGN_FLIP=1
      shift 1
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
    --out-suffix)
      OUT_SUFFIX="$2"
      shift 2
      ;;
    --branch-mode)
      BRANCH_MODE="$2"
      shift 2
      ;;
    --min-b-tokens-before-eos)
      MIN_B_TOKENS_BEFORE_EOS="$2"
      shift 2
      ;;
    --b-retry-times)
      B_RETRY_TIMES="$2"
      shift 2
      ;;
    --auto-close-unclosed-think)
      AUTO_CLOSE_UNCLOSED_THINK=1
      shift 1
      ;;
    --apply-cross-think-cover)
      APPLY_CROSS_THINK_COVER=1
      shift 1
      ;;
    --no-apply-cross-think-cover)
      APPLY_CROSS_THINK_COVER=0
      shift 1
      ;;
    --print-full-output)
      PRINT_FULL_OUTPUT=1
      shift 1
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash run_longproc_32b.sh [options]
  --max-prefix-tokens N
  --max-new-after N
  --checkpoint-delay N
  --checkpoint-mode think_end|regex|think_end_then_regex|think_end_mid
  --checkpoint-regex REGEX
  --checkpoint-mid-min-tokens N
  --checkpoint-mid-max-tokens N
  --checkpoint-mid-avoid-final-regex REGEX
  --corrupt-anchor-regex REGEX
  --corrupt-mode number_shift|anchor_number_shift|none
  --corrupt-after-first-think
  --corrupt-prefer-sign-flip
  --n-samples N
  --task NAME
  --out-root DIR
  --out-suffix TAG
  --model-path PATH
  --branch-mode ab|b
  --min-b-tokens-before-eos N
  --b-retry-times N
  --auto-close-unclosed-think
  --apply-cross-think-cover
  --no-apply-cross-think-cover
  --print-full-output
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${OUT_SUFFIX}" ]]; then
  OUT_ROOT="${OUT_ROOT}_${OUT_SUFFIX}"
fi

EXTRA_ARGS=()
if [[ "${PRINT_FULL_OUTPUT}" == "1" ]]; then
  EXTRA_ARGS+=(--print-full-output)
fi
if [[ "${AUTO_CLOSE_UNCLOSED_THINK}" == "1" ]]; then
  EXTRA_ARGS+=(--auto-close-unclosed-think)
fi
if [[ "${APPLY_CROSS_THINK_COVER}" == "1" ]]; then
  EXTRA_ARGS+=(--apply-cross-think-cover)
fi
if [[ "${CORRUPT_AFTER_FIRST_THINK}" == "1" ]]; then
  EXTRA_ARGS+=(--corrupt-after-first-think)
fi
if [[ "${CORRUPT_PREFER_SIGN_FLIP}" == "1" ]]; then
  EXTRA_ARGS+=(--corrupt-prefer-sign-flip)
fi

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
  --checkpoint-mode "${CHECKPOINT_MODE}" \
  --checkpoint-regex "${CHECKPOINT_REGEX}" \
  --checkpoint-mid-min-tokens "${CHECKPOINT_MID_MIN_TOKENS}" \
  --checkpoint-mid-max-tokens "${CHECKPOINT_MID_MAX_TOKENS}" \
  --checkpoint-mid-avoid-final-regex "${CHECKPOINT_MID_AVOID_FINAL_REGEX}" \
  --corrupt-mode "${CORRUPT_MODE}" \
  --corrupt-anchor-regex "${CORRUPT_ANCHOR_REGEX}" \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --branch-mode "${BRANCH_MODE}" \
  --min-b-tokens-before-eos "${MIN_B_TOKENS_BEFORE_EOS}" \
  --b-retry-times "${B_RETRY_TIMES}" \
  --output-dir "${OUT_ROOT}/baseline" \
  --save-task-texts \
  "${EXTRA_ARGS[@]}"

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
  --checkpoint-mode "${CHECKPOINT_MODE}" \
  --checkpoint-regex "${CHECKPOINT_REGEX}" \
  --checkpoint-mid-min-tokens "${CHECKPOINT_MID_MIN_TOKENS}" \
  --checkpoint-mid-max-tokens "${CHECKPOINT_MID_MAX_TOKENS}" \
  --checkpoint-mid-avoid-final-regex "${CHECKPOINT_MID_AVOID_FINAL_REGEX}" \
  --corrupt-mode "${CORRUPT_MODE}" \
  --corrupt-anchor-regex "${CORRUPT_ANCHOR_REGEX}" \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --branch-mode "${BRANCH_MODE}" \
  --min-b-tokens-before-eos "${MIN_B_TOKENS_BEFORE_EOS}" \
  --b-retry-times "${B_RETRY_TIMES}" \
  --output-dir "${OUT_ROOT}/enhanced" \
  --save-task-texts \
  "${EXTRA_ARGS[@]}"

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
  --checkpoint-mode "${CHECKPOINT_MODE}" \
  --checkpoint-regex "${CHECKPOINT_REGEX}" \
  --checkpoint-mid-min-tokens "${CHECKPOINT_MID_MIN_TOKENS}" \
  --checkpoint-mid-max-tokens "${CHECKPOINT_MID_MAX_TOKENS}" \
  --checkpoint-mid-avoid-final-regex "${CHECKPOINT_MID_AVOID_FINAL_REGEX}" \
  --corrupt-mode "${CORRUPT_MODE}" \
  --corrupt-anchor-regex "${CORRUPT_ANCHOR_REGEX}" \
  --checkpoint-delay "${CHECKPOINT_DELAY}" \
  --max-prefix-tokens "${MAX_PREFIX_TOKENS}" \
  --max-new-after "${MAX_NEW_AFTER}" \
  --branch-mode "${BRANCH_MODE}" \
  --min-b-tokens-before-eos "${MIN_B_TOKENS_BEFORE_EOS}" \
  --b-retry-times "${B_RETRY_TIMES}" \
  --output-dir "${OUT_ROOT}/enhanced_cover" \
  --save-task-texts \
  "${EXTRA_ARGS[@]}"

echo "Done. Outputs in ${OUT_ROOT}"
