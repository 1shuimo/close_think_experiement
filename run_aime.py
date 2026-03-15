#!/usr/bin/env python3
"""
AIME2025 一键评测入口。

定位：
- 只做“中途插入 <think>”评测（Branch B only），不做改错/扰动。
- 内部调用精简后的 test_close_suite.py。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run AIME2025 with insertion-only Branch-B evaluation.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument(
        "--tasks-file",
        default=str(here / "data" / "tasks_aime2025.jsonl"),
        help="AIME task jsonl path.",
    )
    p.add_argument(
        "--output-dir",
        default=str(here / "outputs" / "aime" / "suite_aime2025_insert"),
        help="Output directory for results/summary.",
    )

    p.add_argument(
        "--system-prompt-file",
        default=str(here / "prompts" / "system_enhanced_v2.txt"),
        help="System prompt file.",
    )
    p.add_argument(
        "--inject-text-file",
        default=str(here / "prompts" / "inject_think_v3.txt"),
        help="Injected <think> text file.",
    )
    p.add_argument(
        "--first-think-early-stop-text-file",
        default=str(here / "prompts" / "first_think_early_stop_v1.txt"),
        help="Text inserted immediately before forced </think> when first-think budget is exhausted.",
    )

    p.add_argument("--enable-first-think-max-words", action="store_true")
    p.add_argument("--first-think-max-words", type=int, default=120)
    p.add_argument("--enable-first-think-smooth-close", action="store_true")
    p.add_argument(
        "--first-think-smooth-close-text",
        default="I think this local check is enough, so I will close this think block and continue from this exact point.",
    )
    p.add_argument("--checkpoint-mid-min-tokens", type=int, default=300)
    p.add_argument("--checkpoint-mid-max-tokens", type=int, default=400)

    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument(
        "--first-think-budget-tokens",
        type=int,
        default=None,
        help="Optional explicit budget for the first native <think>. Defaults to max-prefix-tokens.",
    )
    p.add_argument("--max-new-after", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--apply-match-cover", action="store_true", help="Enable overlap trimming.")
    p.add_argument("--apply-cross-think-cover", action="store_true", help="Enable cross-think overlap trimming.")
    p.add_argument("--save-task-texts", action="store_true", help="Save branch full texts per task.")
    p.add_argument("--print-full-output", action="store_true", help="Print full A/B outputs to stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    inject_text = Path(args.inject_text_file).read_text(encoding="utf-8").strip()
    first_think_early_stop_text = Path(args.first_think_early_stop_text_file).read_text(encoding="utf-8").strip()
    first_think_budget_tokens = (
        int(args.first_think_budget_tokens)
        if args.first_think_budget_tokens is not None
        else int(args.max_prefix_tokens)
    )
    effective_max_prefix_tokens = max(
        int(args.max_prefix_tokens),
        int(first_think_budget_tokens) + int(args.checkpoint_mid_max_tokens),
    )

    cmd = [
        sys.executable,
        str(here / "test_close_suite.py"),
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
        "--no-math-step-format-guidance",
        "--inject-text",
        inject_text,
        "--prompt-mode",
        "baseline",
        "--first-think-max-words",
        str(args.first_think_max_words),
        "--first-think-smooth-close-text",
        args.first_think_smooth_close_text,
        "--checkpoint-mode",
        "think_end_punct",
        "--checkpoint-regex",
        "__auto__",
        "--checkpoint-delay",
        "0",
        "--checkpoint-mid-min-tokens",
        str(args.checkpoint_mid_min_tokens),
        "--checkpoint-mid-max-tokens",
        str(args.checkpoint_mid_max_tokens),
        "--checkpoint-mid-avoid-final-regex",
        r"(?i)\bfinal\s*:|\bfinal answer\b",
        "--first-think-budget-tokens",
        str(first_think_budget_tokens),
        "--first-think-early-stop-text",
        first_think_early_stop_text,
        "--max-prefix-tokens",
        str(effective_max_prefix_tokens),
        "--max-new-after",
        str(args.max_new_after),
        "--corrupt-after-first-think",
        "--branch-mode",
        "b",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]

    if args.apply_match_cover:
        cmd.append("--apply-match-cover")
    if args.enable_first_think_max_words:
        cmd.append("--enable-first-think-max-words")
    if args.enable_first_think_smooth_close:
        cmd.append("--enable-first-think-smooth-close")
    if args.apply_cross_think_cover:
        cmd.append("--apply-cross-think-cover")
    if args.save_task_texts:
        cmd.append("--save-task-texts")
    if args.print_full_output:
        cmd.append("--print-full-output")

    print("[run_aime] command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(here))


if __name__ == "__main__":
    main()
