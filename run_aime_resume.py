#!/usr/bin/env python3
from __future__ import annotations

"""
Resume-capable AIME insertion runner.

Purpose:
- continue an interrupted `run_aime.py` experiment without rerunning finished tasks
- recover completed task ids from existing `results.jsonl` and/or saved `task_texts`
- persist outputs after each task so future interruptions do not lose all progress
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

torch = None
build_concise_branch_meta = None
run_task_ab = None
summarize = None
load_model = None
load_tasks_jsonl = None
model_label = None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Resume an interrupted AIME insertion run.")
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
    p.add_argument("--print-full-output", action="store_true", help="Print full B outputs to stdout.")
    p.add_argument("--dry-run", action="store_true", help="Only report completed/remaining tasks.")
    return p.parse_args()


def _safe_name(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def _read_existing_records(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    if not jsonl_path.exists():
        return records
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = str(obj.get("task_id") or "").strip()
            if task_id:
                records[task_id] = obj
    return records


def _placeholder_branch(branch_meta: Optional[Dict[str, object]], full_text: Optional[str]) -> Dict[str, object]:
    branch_meta = branch_meta or {}
    trimmed = branch_meta.get("trimmed") or {}
    return {
        "full_text": full_text,
        "metrics": {
            "answer_text": branch_meta.get("answer"),
            "expected_hit": branch_meta.get("answer_hit"),
            "expected_hit_where": branch_meta.get("answer_hit_where"),
            "expected_match_text": branch_meta.get("matched_text"),
            "expected_match_text_raw": branch_meta.get("matched_text"),
            "think_balanced": branch_meta.get("think_closed"),
        },
        "match_cover": {
            "trimmed_chars": int(trimmed.get("match_cover_trimmed_chars", 0) or 0),
        },
        "cross_think_cover": {
            "trimmed_chars": int(trimmed.get("cross_think_trimmed_chars", 0) or 0),
        },
    }


def _recover_records_from_task_texts(
    *,
    task_dump_root: Path,
    tasks_by_id: Dict[str, Dict[str, object]],
    model_path: str,
    existing_ids: set[str],
) -> Dict[str, Dict[str, object]]:
    recovered: Dict[str, Dict[str, object]] = {}
    if not task_dump_root.exists():
        return recovered

    for task_dir in sorted(task_dump_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        if task_id in existing_ids or task_id not in tasks_by_id:
            continue

        meta_path = task_dir / "meta.json"
        branch_b_path = task_dir / "branch_B.full.txt"
        if not branch_b_path.exists():
            continue

        meta_obj: Dict[str, object] = {}
        if meta_path.exists():
            try:
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_obj = {}

        task = tasks_by_id[task_id]
        branch_b_text = branch_b_path.read_text(encoding="utf-8")

        branch_a_obj: Dict[str, object]
        branch_a_path = task_dir / "branch_A.full.txt"
        if branch_a_path.exists():
            branch_a_obj = _placeholder_branch(
                meta_obj.get("branch_A") if isinstance(meta_obj, dict) else None,
                branch_a_path.read_text(encoding="utf-8"),
            )
        else:
            branch_a_obj = {"skipped": True, "reason": "recovered_from_task_texts_only"}

        recovered[task_id] = {
            "task_id": task_id,
            "user_prompt": task.get("user_prompt"),
            "expected_answer": meta_obj.get("expected_answer", task.get("expected_answer")),
            "expected_regex": meta_obj.get("expected_regex", task.get("expected_regex")),
            "reference_output": task.get("reference_output"),
            "prompt_mode": "baseline",
            "system_prompt_used": None,
            "checkpoint_meta": {"recovered_from": "task_texts"},
            "prefix_seen_first_think_end": None,
            "prefix_tokens": None,
            "prefix_think_limit_meta": {"recovered_from": "task_texts"},
            "prefix_step_wait_meta": {"recovered_from": "task_texts"},
            "insert_meta": {"recovered_from": "task_texts"},
            "branch_mode": "b",
            "branch_A": branch_a_obj,
            "branch_B": _placeholder_branch(
                meta_obj.get("branch_B") if isinstance(meta_obj, dict) else None,
                branch_b_text,
            ),
            "model_path": model_path,
        }
    return recovered


def _ordered_records(tasks: List[Dict[str, object]], records_by_task: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    ordered: List[Dict[str, object]] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        rec = records_by_task.get(task_id)
        if rec is not None:
            ordered.append(rec)
    return ordered


def _write_model_jsonl(jsonl_path: Path, records: List[Dict[str, object]]) -> None:
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_task_texts(task_dump_root: Path, rec: Dict[str, object]) -> None:
    global build_concise_branch_meta
    if build_concise_branch_meta is None:
        from test_close_suite import build_concise_branch_meta as _build_concise_branch_meta

        build_concise_branch_meta = _build_concise_branch_meta

    task_name = _safe_name(str(rec.get("task_id", "task")))
    task_dir = task_dump_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    branch_a = rec.get("branch_A", {}) or {}
    branch_b = rec.get("branch_B", {}) or {}
    if branch_a.get("full_text") is not None:
        (task_dir / "branch_A.full.txt").write_text(str(branch_a["full_text"]), encoding="utf-8")
    if branch_b.get("full_text") is not None:
        (task_dir / "branch_B.full.txt").write_text(str(branch_b["full_text"]), encoding="utf-8")

    meta_payload: Dict[str, object] = {
        "task_id": rec.get("task_id"),
        "expected_answer": rec.get("expected_answer"),
        "expected_regex": rec.get("expected_regex"),
    }
    if isinstance(branch_b, dict):
        meta_payload["branch_B"] = build_concise_branch_meta(branch_b)
    if isinstance(branch_a, dict) and branch_a.get("full_text") is not None:
        meta_payload["branch_A"] = build_concise_branch_meta(branch_a)
    (task_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_summaries(
    *,
    output_dir: Path,
    model_summaries: Dict[str, Dict[str, object]],
    task_source: Path,
    n_tasks: int,
) -> None:
    global_summary = {
        "task_source": f"jsonl:{task_source.resolve()}",
        "n_tasks": n_tasks,
        "models": model_summaries,
    }
    (output_dir / "summary_all_models.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    global torch, build_concise_branch_meta, run_task_ab, summarize, load_model, load_tasks_jsonl, model_label

    import torch as _torch

    from test_close_suite import build_concise_branch_meta as _build_concise_branch_meta
    from test_close_suite import run_task_ab as _run_task_ab
    from test_close_suite import summarize as _summarize
    from think_branch_common import load_model as _load_model
    from think_branch_common import load_tasks_jsonl as _load_tasks_jsonl
    from think_branch_common import model_label as _model_label

    torch = _torch
    build_concise_branch_meta = _build_concise_branch_meta
    run_task_ab = _run_task_ab
    summarize = _summarize
    load_model = _load_model
    load_tasks_jsonl = _load_tasks_jsonl
    model_label = _model_label
    here = Path(__file__).resolve().parent
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
        if args.save_task_texts and ordered_existing:
            task_dump_root.mkdir(parents=True, exist_ok=True)
            for rec in ordered_existing:
                _write_task_texts(task_dump_root, rec)
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
            rec = run_task_ab(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_prompt=system_prompt,
                prompt_mode="baseline",
                enable_model_thinking=True,
                think_word_limit=60,
                enable_think_word_limit=False,
                enable_first_think_max_words=bool(args.enable_first_think_max_words),
                first_think_max_words=args.first_think_max_words,
                enable_first_think_smooth_close=bool(args.enable_first_think_smooth_close),
                first_think_smooth_close_text=args.first_think_smooth_close_text,
                first_think_budget_tokens=first_think_budget_tokens,
                first_think_early_stop_text=first_think_early_stop_text,
                math_step_user_guidance="",
                task=task,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * i,
                checkpoint_delay=0,
                checkpoint_mode="think_end_punct",
                checkpoint_regex=None,
                checkpoint_mid_min_tokens=args.checkpoint_mid_min_tokens,
                checkpoint_mid_max_tokens=args.checkpoint_mid_max_tokens,
                checkpoint_mid_avoid_final_regex=r"(?i)\bfinal\s*:|\bfinal answer\b",
                max_prefix_tokens=effective_max_prefix_tokens,
                step_wait_extra_tokens=1200,
                no_step_fallback_offset_tokens=300,
                max_new_after=args.max_new_after,
                branch_mode="b",
                min_b_tokens_before_eos=64,
                b_retry_times=0,
                auto_close_unclosed_think=False,
                chunk_size=2048,
                inject_text=inject_text,
                corrupt_after_first_think=True,
                force_inject_at_corrupt=False,
                force_inject_at_sentence_end=False,
                align_stop_with_insert=False,
                apply_match_cover_flag=bool(args.apply_match_cover),
                apply_cross_think_cover_flag=bool(args.apply_cross_think_cover),
                cover_min_exact_overlap=40,
                cover_fuzzy_min_len=40,
                cover_fuzzy_max_len=140,
                cover_fuzzy_ratio=0.92,
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
                print(f"branch_mode={rec.get('branch_mode')}")
                print(f"insert_meta={json.dumps(rec.get('insert_meta', {}), ensure_ascii=False)}")
                print(f"branch_B_retry={json.dumps(rec['branch_B'].get('retry_info', {}), ensure_ascii=False)}")
                print(f"branch_B_stop_reason={rec['branch_B'].get('stop_reason')}")
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
