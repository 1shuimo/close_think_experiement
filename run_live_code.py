#!/usr/bin/env python3
"""
LiveCodeBench 对接脚本（官方 runner 包装层）。

用途：
1) 统一在本仓库里触发 LiveCodeBench 标准评测。
2) 显式固定关键参数（model / release / scenario）。
3) 可选追加 compute_scores 时间窗重评分。

说明：
- 该脚本调用 LiveCodeBench 官方命令，不做“改错”逻辑。
- 若需要中插 <think> 的实验，请在本仓库的 test_close_suite.py 路线下跑。
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run LiveCodeBench official runner from this repo.")

    p.add_argument(
        "--lcb-root",
        default=str((here / "../LiveCodeBench").resolve()),
        help="Path to cloned LiveCodeBench repository.",
    )
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")

    p.add_argument("--model", required=True, help="Model name for lcb_runner.")
    p.add_argument("--local-model-path", default=None, help="Local model path (optional).")
    p.add_argument("--scenario", default="codegeneration", help="LCB scenario, default codegeneration.")
    p.add_argument("--release-version", default="release_v6", help="LiveCodeBench release version.")

    p.add_argument("--evaluate", action="store_true", help="Pass --evaluate to runner.")
    p.add_argument("--n", type=int, default=None, help="Number of samples per problem (if runner supports).")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (if runner supports).")

    p.add_argument(
        "--runner-extra",
        action="append",
        default=[],
        help="Extra raw args forwarded to lcb_runner.runner.main (repeatable).",
    )

    p.add_argument("--compute-scores", action="store_true", help="Run compute_scores after runner.")
    p.add_argument("--eval-all-file", default=None, help="Path to eval_all file for compute_scores.")
    p.add_argument("--start-date", default=None, help="Start date for compute_scores, e.g. 2025-01-01.")
    p.add_argument("--end-date", default=None, help="End date for compute_scores, e.g. 2025-05-01.")
    p.add_argument(
        "--strip-think-before-score",
        action="store_true",
        help="Strip <think>...</think> blocks from eval_all JSON before compute_scores.",
    )
    p.add_argument(
        "--strip-think-output-file",
        default=None,
        help="Output file path for stripped eval_all (default: <eval_all>.strip_think.json).",
    )
    p.add_argument(
        "--strip-think-key-regex",
        default=r"(?i)(output|completion|prediction|pred|response|answer|code)",
        help="Only strip string fields under matching keys (ignored with --strip-think-all-strings).",
    )
    p.add_argument(
        "--strip-think-all-strings",
        action="store_true",
        help="Strip think blocks from every string field in eval_all JSON.",
    )

    p.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    return p.parse_args()


def _expand_extra_args(extras: list[str]) -> list[str]:
    merged: list[str] = []
    for item in extras:
        merged.extend(shlex.split(item))
    return merged


def main() -> None:
    args = parse_args()
    lcb_root = Path(args.lcb_root).resolve()
    if not lcb_root.exists():
        raise FileNotFoundError(f"LiveCodeBench path not found: {lcb_root}")

    # 1) 官方 runner 主命令
    runner_cmd = [
        args.python_bin,
        "-m",
        "lcb_runner.runner.main",
        "--model",
        args.model,
        "--scenario",
        args.scenario,
        "--release_version",
        args.release_version,
    ]

    if args.local_model_path:
        runner_cmd.extend(["--local_model_path", args.local_model_path])
    if args.evaluate:
        runner_cmd.append("--evaluate")
    if args.n is not None:
        runner_cmd.extend(["--n", str(args.n)])
    if args.temperature is not None:
        runner_cmd.extend(["--temperature", str(args.temperature)])

    runner_cmd.extend(_expand_extra_args(args.runner_extra))

    print("[run_live_code] runner command:")
    print(" ".join(runner_cmd))

    if not args.dry_run:
        subprocess.run(runner_cmd, check=True, cwd=str(lcb_root))

    # 2) 可选时间窗重打分
    if args.compute_scores:
        if not args.eval_all_file:
            raise ValueError("--compute-scores requires --eval-all-file")
        eval_all_for_score = args.eval_all_file

        if args.strip_think_before_score:
            strip_out = args.strip_think_output_file
            if not strip_out:
                p = Path(args.eval_all_file)
                strip_out = str(p.with_name(p.stem + ".strip_think" + p.suffix))
            strip_cmd = [
                args.python_bin,
                str((Path(__file__).resolve().parent / "lcb_postprocess_strip_think.py")),
                "--input",
                args.eval_all_file,
                "--output",
                strip_out,
                "--key-regex",
                args.strip_think_key_regex,
            ]
            if args.strip_think_all_strings:
                strip_cmd.append("--strip-all-strings")
            print("[run_live_code] strip_think command:")
            print(" ".join(strip_cmd))
            if not args.dry_run:
                subprocess.run(strip_cmd, check=True)
            eval_all_for_score = strip_out

        score_cmd = [
            args.python_bin,
            "-m",
            "lcb_runner.evaluation.compute_scores",
            "--eval_all_file",
            eval_all_for_score,
        ]
        if args.start_date:
            score_cmd.extend(["--start_date", args.start_date])
        if args.end_date:
            score_cmd.extend(["--end_date", args.end_date])

        print("[run_live_code] compute_scores command:")
        print(" ".join(score_cmd))
        if not args.dry_run:
            subprocess.run(score_cmd, check=True, cwd=str(lcb_root))


if __name__ == "__main__":
    main()
