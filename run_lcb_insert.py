#!/usr/bin/env python3
"""
LiveCodeBench insertion runner (no corruption).

Notes:
- This script does NOT modify AIME entry scripts.
- It wraps test_close_suite_corrupt.py with a fixed insertion-only setup for LCB tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run LCB tasks with insertion-only branch B setup.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--tasks-file", required=True, help="LCB tasks jsonl prepared by prepare_lcb_codegen_tasks.py")
    p.add_argument("--output-dir", default=str(here / "suite_lcb_insert"))

    p.add_argument("--system-prompt-file", default=str(here / "prompts" / "system_enhanced_v1.txt"))
    p.add_argument("--inject-text-file", default=str(here / "prompts" / "inject_think_v2.txt"))
    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])

    p.add_argument(
        "--checkpoint-mode",
        default="think_end_then_regex",
        choices=["think_end", "regex", "think_end_then_regex", "think_end_mid"],
    )
    p.add_argument("--checkpoint-regex", default=r"(?i)step\s*3")
    p.add_argument("--checkpoint-delay", type=int, default=0)
    p.add_argument("--checkpoint-mid-min-tokens", type=int, default=20)
    p.add_argument("--checkpoint-mid-max-tokens", type=int, default=30)
    p.add_argument("--checkpoint-mid-avoid-final-regex", default=r"(?i)\bfinal\s*:|\bfinal answer\b")

    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument("--step-wait-extra-tokens", type=int, default=1200)
    p.add_argument("--no-step-fallback-offset-tokens", type=int, default=300)
    p.add_argument("--max-new-after", type=int, default=1200)

    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument("--enable-think-word-limit", action="store_true")
    p.add_argument("--enable-first-think-max-words", action="store_true")
    p.add_argument("--first-think-max-words", type=int, default=120)

    p.add_argument("--save-task-texts", action="store_true")
    p.add_argument("--print-full-output", action="store_true")
    p.add_argument("--apply-match-cover", action="store_true")
    p.add_argument("--apply-cross-think-cover", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    inject_text = Path(args.inject_text_file).read_text(encoding="utf-8")

    cmd = [
        sys.executable,
        str(here / "test_close_suite_corrupt.py"),
        "--model-paths",
        args.model_paths,
        "--dtype",
        args.dtype,
        "--tasks-file",
        args.tasks_file,
        "--output-dir",
        args.output_dir,
        "--system-prompt-file",
        args.system_prompt_file,
        "--inject-text",
        inject_text,
        "--prompt-mode",
        args.prompt_mode,
        "--no-math-step-format-guidance",
        "--think-word-limit",
        str(args.think_word_limit),
        "--first-think-max-words",
        str(args.first_think_max_words),
        "--checkpoint-mode",
        args.checkpoint_mode,
        "--checkpoint-regex",
        args.checkpoint_regex,
        "--checkpoint-delay",
        str(args.checkpoint_delay),
        "--checkpoint-mid-min-tokens",
        str(args.checkpoint_mid_min_tokens),
        "--checkpoint-mid-max-tokens",
        str(args.checkpoint_mid_max_tokens),
        "--checkpoint-mid-avoid-final-regex",
        args.checkpoint_mid_avoid_final_regex,
        "--max-prefix-tokens",
        str(args.max_prefix_tokens),
        "--step-wait-extra-tokens",
        str(args.step_wait_extra_tokens),
        "--no-step-fallback-offset-tokens",
        str(args.no_step_fallback_offset_tokens),
        "--max-new-after",
        str(args.max_new_after),
        "--branch-mode",
        "b",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--locator-only",
        "--corrupt-mode",
        "none",
        "--corrupt-min-step",
        "2",
        "--corrupt-after-first-think",
    ]

    if args.enable_think_word_limit:
        cmd.append("--enable-think-word-limit")
    if args.enable_first_think_max_words:
        cmd.append("--enable-first-think-max-words")
    if args.save_task_texts:
        cmd.append("--save-task-texts")
    if args.print_full_output:
        cmd.append("--print-full-output")
    if args.apply_match_cover:
        cmd.append("--apply-match-cover")
    if args.apply_cross_think_cover:
        cmd.append("--apply-cross-think-cover")

    print("[run_lcb_insert] command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(here))


if __name__ == "__main__":
    main()
