#!/usr/bin/env python3
"""
AIME2025 一键评测入口。

定位：
- 只做“中途插入 <think>”对照评测（A/B），不做改错/扰动。
- 内部调用精简后的 test_close_suite.py。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run AIME2025 with insertion-only A/B evaluation.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument(
        "--tasks-file",
        default=str(here / "tasks_aime2025.jsonl"),
        help="AIME task jsonl path.",
    )
    p.add_argument(
        "--output-dir",
        default=str(here / "suite_aime2025_insert"),
        help="Output directory for results/summary.",
    )

    p.add_argument(
        "--system-prompt-file",
        default=str(here / "prompts" / "system_enhanced_v1.txt"),
        help="System prompt file.",
    )
    p.add_argument(
        "--inject-text-file",
        default=str(here / "prompts" / "inject_think_v2.txt"),
        help="Injected <think> text file.",
    )

    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])
    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument("--enable-think-word-limit", action="store_true")
    p.add_argument("--checkpoint-mode", default="think_end_mid", choices=["think_end", "regex", "think_end_then_regex", "think_end_mid"])
    p.add_argument("--checkpoint-regex", default="__auto__")
    p.add_argument("--checkpoint-delay", type=int, default=0)
    p.add_argument("--checkpoint-mid-min-tokens", type=int, default=20)
    p.add_argument("--checkpoint-mid-max-tokens", type=int, default=30)
    p.add_argument(
        "--checkpoint-mid-avoid-final-regex",
        default=r"(?i)\bfinal\s*:|\bfinal answer\b",
    )

    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument("--max-new-after", type=int, default=1200)
    p.add_argument("--branch-mode", default="ab", choices=["ab", "b"])
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

    inject_text = Path(args.inject_text_file).read_text(encoding="utf-8")

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
        "--inject-text",
        inject_text,
        "--prompt-mode",
        args.prompt_mode,
        "--think-word-limit",
        str(args.think_word_limit),
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
        "--max-new-after",
        str(args.max_new_after),
        "--branch-mode",
        args.branch_mode,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]

    if args.apply_match_cover:
        cmd.append("--apply-match-cover")
    if args.enable_think_word_limit:
        cmd.append("--enable-think-word-limit")
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
