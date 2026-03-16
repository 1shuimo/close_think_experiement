#!/usr/bin/env python3
from __future__ import annotations

"""
Resume-capable repeated-insertion AIME runner.

Purpose:
- continue an interrupted `run_aime_multi_insert.py` experiment without rerunning finished tasks
- recover completed task ids from existing `results.jsonl` and/or saved `task_texts`
- persist outputs after each task so future interruptions do not lose all progress
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict

torch = None
build_concise_branch_meta = None
run_task_multi_insert = None
summarize = None
load_model = None
load_tasks_jsonl = None
model_label = None
_read_existing_records = None
_recover_records_from_task_texts = None
_ordered_records = None
_write_model_jsonl = None
_write_task_texts = None
_write_summaries = None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Resume an interrupted repeated-insertion AIME run.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument(
        "--tasks-file",
        default=str(here / "data" / "tasks_aime2025.jsonl"),
        help="AIME task jsonl path.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Existing output directory of the interrupted run.",
    )

    p.add_argument(
        "--system-prompt-file",
        default=str(here / "prompts" / "system_math_interleave_v1.txt"),
        help="System prompt file.",
    )
    p.add_argument(
        "--inject-text-file",
        default=str(here / "prompts" / "inject_math_interleave_v1.txt"),
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
    p.add_argument("--min-b-tokens-before-eos", type=int, default=64)
    p.add_argument("--max-reinserts", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--save-task-texts", action="store_true", help="Save branch full texts per task.")
    p.add_argument("--print-full-output", action="store_true", help="Print full B outputs to stdout.")
    p.add_argument("--dry-run", action="store_true", help="Only report completed/remaining tasks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global torch, build_concise_branch_meta, run_task_multi_insert, summarize, load_model
    global load_tasks_jsonl, model_label, _read_existing_records, _recover_records_from_task_texts
    global _ordered_records, _write_model_jsonl, _write_task_texts, _write_summaries

    import torch as _torch

    from test_close_suite import build_concise_branch_meta as _build_concise_branch_meta
    from test_close_suite import summarize as _summarize
    from think_branch_common import load_model as _load_model
    from think_branch_common import load_tasks_jsonl as _load_tasks_jsonl
    from think_branch_common import model_label as _model_label
    from run_aime_multi_insert import run_task_multi_insert as _run_task_multi_insert
    from run_aime_resume import (
        _ordered_records as __ordered_records,
        _read_existing_records as __read_existing_records,
        _recover_records_from_task_texts as __recover_records_from_task_texts,
        _write_model_jsonl as __write_model_jsonl,
        _write_summaries as __write_summaries,
        _write_task_texts as __write_task_texts,
    )

    torch = _torch
    build_concise_branch_meta = _build_concise_branch_meta
    run_task_multi_insert = _run_task_multi_insert
    summarize = _summarize
    load_model = _load_model
    load_tasks_jsonl = _load_tasks_jsonl
    model_label = _model_label
    _read_existing_records = __read_existing_records
    _recover_records_from_task_texts = __recover_records_from_task_texts
    _ordered_records = __ordered_records
    _write_model_jsonl = __write_model_jsonl
    _write_task_texts = __write_task_texts
    _write_summaries = __write_summaries

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = Path(args.tasks_file)
    tasks = load_tasks_jsonl(tasks_path)
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    inject_text = Path(args.inject_text_file).read_text(encoding="utf-8")
    first_think_early_stop_text = Path(args.first_think_early_stop_text_file).read_text(encoding="utf-8")
    first_think_budget_tokens = (
        int(args.first_think_budget_tokens)
        if args.first_think_budget_tokens is not None
        else int(args.max_prefix_tokens)
    )
    effective_max_prefix_tokens = max(
        int(args.max_prefix_tokens),
        int(first_think_budget_tokens) + int(args.checkpoint_mid_max_tokens),
    )
    model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]

    model_summaries: Dict[str, Dict[str, object]] = {}

    for model_path in model_paths:
        print(f"\n===== Resuming model: {model_path} =====")
        model_tag = model_label(model_path)
        jsonl_path = output_dir / f"{model_tag}.results.jsonl"
        summary_path = output_dir / f"{model_tag}.summary.json"
        task_dump_root = output_dir / f"{model_tag}.task_texts"

        existing_records = _read_existing_records(jsonl_path)
        recovered_records = _recover_records_from_task_texts(
            task_dump_root=task_dump_root,
            tasks_by_id=tasks_by_id,
            model_path=model_path,
            existing_ids=set(existing_records.keys()),
        )
        records_by_task: Dict[str, Dict[str, object]] = {}
        records_by_task.update(existing_records)
        records_by_task.update(recovered_records)

        completed_ids = {task_id for task_id in tasks_by_id if task_id in records_by_task}
        remaining_tasks = [task for task in tasks if str(task.get("id") or "") not in completed_ids]

        print(
            f"[resume] completed={len(completed_ids)} "
            f"(jsonl={len(existing_records)}, task_texts_only={len(recovered_records)}) "
            f"remaining={len(remaining_tasks)}"
        )
        if remaining_tasks:
            print("[resume] next tasks:", ", ".join(str(t.get("id")) for t in remaining_tasks[:8]))

        ordered_existing = _ordered_records(tasks, records_by_task)
        if recovered_records or (not jsonl_path.exists() and ordered_existing):
            _write_model_jsonl(jsonl_path, ordered_existing)

        model_summary = summarize(ordered_existing)
        summary_path.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        model_summaries[model_path] = model_summary
        _write_summaries(
            output_dir=output_dir,
            model_summaries=model_summaries,
            task_source=tasks_path,
            n_tasks=len(tasks),
        )

        if args.dry_run or not remaining_tasks:
            continue

        loaded = load_model(model_path, dtype_name=args.dtype)
        tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
        if args.save_task_texts:
            task_dump_root.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks, start=1):
            task_id = str(task.get("id") or "")
            if task_id in records_by_task:
                continue

            print(f"[resume {i}/{len(tasks)}] {task_id}")
            rec = run_task_multi_insert(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_prompt=system_prompt,
                task=task,
                inject_text=inject_text,
                first_think_early_stop_text=first_think_early_stop_text,
                checkpoint_mid_min_tokens=args.checkpoint_mid_min_tokens,
                checkpoint_mid_max_tokens=args.checkpoint_mid_max_tokens,
                max_prefix_tokens=effective_max_prefix_tokens,
                first_think_budget_tokens=first_think_budget_tokens,
                max_new_after=args.max_new_after,
                min_b_tokens_before_eos=args.min_b_tokens_before_eos,
                max_reinserts=args.max_reinserts,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * i,
                chunk_size=args.chunk_size,
                enable_first_think_max_words=bool(args.enable_first_think_max_words),
                first_think_max_words=args.first_think_max_words,
                enable_first_think_smooth_close=bool(args.enable_first_think_smooth_close),
                first_think_smooth_close_text=args.first_think_smooth_close_text,
            )
            rec["model_path"] = model_path
            records_by_task[task_id] = rec

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.save_task_texts:
                _write_task_texts(task_dump_root, rec)

            ordered_now = _ordered_records(tasks, records_by_task)
            model_summary = summarize(ordered_now)
            summary_path.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            model_summaries[model_path] = model_summary
            _write_summaries(
                output_dir=output_dir,
                model_summaries=model_summaries,
                task_source=tasks_path,
                n_tasks=len(tasks),
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.print_full_output:
                print("\n---------------- FULL OUTPUT ----------------")
                print(f"model={model_path}")
                print(f"task={rec.get('task_id')}")
                print(f"insert_meta={json.dumps(rec.get('insert_meta', {}), ensure_ascii=False)}")
                print(f"branch_B_stop_reason={rec['branch_B'].get('stop_reason')}")
                print(f"branch_B_multi_insert={json.dumps(rec['branch_B'].get('multi_insert_info', {}), ensure_ascii=False)}")
                print("\n[Branch B Full Text]\n")
                print(rec["branch_B"]["full_text"])
                print("\n---------------------------------------------\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[resume] saved:\n- {jsonl_path}\n- {summary_path}")

    print(f"\n[resume] done: {output_dir / 'summary_all_models.json'}")


if __name__ == "__main__":
    main()
