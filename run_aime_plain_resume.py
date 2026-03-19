#!/usr/bin/env python3
from __future__ import annotations

"""
Resume-capable plain AIME runner.

Purpose:
- continue an interrupted `run_aime_plain.py` experiment without rerunning finished tasks
- recover completed task ids from existing `results.jsonl` and/or saved `task_texts`
- persist outputs after each task so future interruptions do not lose all progress
"""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List


def _safe_name(s: str) -> str:
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


def _recover_records_from_task_texts(
    *,
    task_dump_root: Path,
    tasks_by_id: Dict[str, Dict[str, object]],
    model_path: str,
) -> Dict[str, Dict[str, object]]:
    recovered: Dict[str, Dict[str, object]] = {}
    if not task_dump_root.exists():
        return recovered

    for task_dir in sorted(task_dump_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        if task_id not in tasks_by_id:
            continue

        full_path = task_dir / "full.txt"
        meta_path = task_dir / "meta.json"
        if not full_path.exists():
            continue

        full_text = full_path.read_text(encoding="utf-8")
        meta_obj: Dict[str, object] = {}
        if meta_path.exists():
            try:
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_obj = {}

        task = tasks_by_id[task_id]
        recovered[task_id] = {
            "task_id": task_id,
            "user_prompt": task.get("user_prompt"),
            "expected_regex": meta_obj.get("expected_regex", task.get("expected_regex")),
            "reference_output": task.get("reference_output"),
            "system_prompt_file": None,
            "disable_model_thinking": None,
            "checkpoint_meta": meta_obj.get("checkpoint_meta", {"recovered_from": "task_texts"}),
            "full_text": full_text,
            "stripped_text": None,
            "new_tokens": meta_obj.get("new_tokens"),
            "stop_reason": meta_obj.get("stop_reason"),
            "metrics": meta_obj.get("metrics", {}),
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
    task_dir = task_dump_root / _safe_name(str(rec.get("task_id", "task")))
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "full.txt").write_text(str(rec.get("full_text") or ""), encoding="utf-8")
    (task_dir / "meta.json").write_text(
        json.dumps(
            {
                "task_id": rec.get("task_id"),
                "user_prompt": rec.get("user_prompt"),
                "expected_regex": rec.get("expected_regex"),
                "stop_reason": rec.get("stop_reason"),
                "new_tokens": rec.get("new_tokens"),
                "checkpoint_meta": rec.get("checkpoint_meta", {}),
                "metrics": rec.get("metrics", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Resume an interrupted plain AIME run.")
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
        help="Existing output directory of the interrupted plain run.",
    )
    p.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional system prompt file. Default is no extra system prompt.",
    )
    p.add_argument(
        "--disable-model-thinking",
        action="store_true",
        help="Disable native model thinking in apply_chat_template.",
    )
    p.add_argument(
        "--first-think-budget-tokens",
        type=int,
        default=None,
        help="Optional Qwen-style budget for the first native <think>. If reached before </think>, force-close it.",
    )
    p.add_argument(
        "--first-think-early-stop-text-file",
        default=str(here / "prompts" / "first_think_early_stop_v1.txt"),
        help="Text inserted immediately before forced </think> when first-think budget is exhausted.",
    )
    p.add_argument("--max-new-tokens", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--save-task-texts", action="store_true")
    p.add_argument("--print-full-output", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from run_aime_plain import _build_prompt_ids, _expected_hit, summarize
    from think_branch_common import (
        generate_until_checkpoint,
        generate_from_state,
        load_model,
        load_tasks_jsonl,
        model_label,
        prefill_kv,
        strip_think_blocks,
        think_balance_ok,
    )

    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        alt = Path(tasks_path.name)
        if alt.exists():
            tasks_path = alt
    tasks = load_tasks_jsonl(tasks_path)
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}

    system_prompt = ""
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    first_think_early_stop_text = ""
    if args.first_think_early_stop_text_file:
        first_think_early_stop_text = Path(args.first_think_early_stop_text_file).read_text(encoding="utf-8")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]

    global_summary: Dict[str, object] = {
        "task_source": f"jsonl:{tasks_path.resolve()}",
        "n_tasks": len(tasks),
        "models": {},
    }

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
        )
        records_by_task: Dict[str, Dict[str, object]] = {}
        records_by_task.update(recovered_records)
        records_by_task.update(existing_records)

        completed_ids = {task_id for task_id in tasks_by_id if task_id in records_by_task}
        remaining_tasks = [task for task in tasks if str(task.get("id") or "") not in completed_ids]

        print(
            f"[resume] completed={len(completed_ids)} "
            f"(jsonl={len(existing_records)}, task_texts_only={max(0, len(recovered_records) - len(existing_records))}) "
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
        model_summary["config"] = {
            "system_prompt_file": args.system_prompt_file,
            "disable_model_thinking": bool(args.disable_model_thinking),
            "first_think_budget_tokens": int(args.first_think_budget_tokens or 0),
            "first_think_early_stop_text_file": args.first_think_early_stop_text_file,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        summary_path.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        global_summary["models"][model_path] = model_summary
        (output_dir / "summary_all_models.json").write_text(
            json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if args.dry_run or not remaining_tasks:
            continue

        loaded = load_model(model_path, dtype_name=args.dtype)
        tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
        if args.save_task_texts:
            task_dump_root.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks, start=1):
            task_id = str(task.get("id", f"task_{i}"))
            if task_id in records_by_task:
                continue

            print(f"[resume {i}/{len(tasks)}] {task_id}")
            user_prompt = str(task["user_prompt"])
            expected_regex = task.get("expected_regex")

            prompt_ids = _build_prompt_ids(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                device=device,
                enable_thinking=not bool(args.disable_model_thinking),
            )
            past = None
            logits = None
            checkpoint_meta: Dict[str, object]
            if bool(args.disable_model_thinking) or not args.first_think_budget_tokens:
                past, logits = prefill_kv(model, prompt_ids, chunk_size=args.chunk_size)
                gen_out = generate_from_state(
                    model=model,
                    tokenizer=tokenizer,
                    past_key_values=past,
                    logits=logits,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + 1000 * i,
                    return_meta=True,
                    print_stream=False,
                )
                generated_ids = gen_out["generated_ids"]
                full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                checkpoint_meta = {
                    "first_think_budget_tokens": 0,
                    "first_think_budget_triggered": False,
                    "first_think_budget_forced_close_applied": False,
                    "first_think_budget_forced_close_text": None,
                    "prefix_tokens_before_plain_continuation": 0,
                }
            else:
                ckpt = generate_until_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    delay_tokens_after_first_think_end=0,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + 1000 * i,
                    checkpoint_mode="think_end",
                    checkpoint_regex=None,
                    first_think_budget_tokens=int(args.first_think_budget_tokens),
                    first_think_early_stop_text=first_think_early_stop_text,
                    chunk_size=args.chunk_size,
                    print_stream=False,
                )
                prefix_ids = list(ckpt.get("generated_ids") or [])
                prefix_text = ckpt.get("generated_text") or tokenizer.decode(prefix_ids, skip_special_tokens=False)
                remaining_tokens = max(0, int(args.max_new_tokens) - len(prefix_ids))
                if remaining_tokens > 0:
                    prefix_ids_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                    full_ids = torch.cat([prompt_ids, prefix_ids_t], dim=1)
                    past, logits = prefill_kv(model, full_ids, chunk_size=args.chunk_size)
                    gen_out = generate_from_state(
                        model=model,
                        tokenizer=tokenizer,
                        past_key_values=past,
                        logits=logits,
                        max_new_tokens=remaining_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed + 1000 * i + 1,
                        return_meta=True,
                        print_stream=False,
                    )
                    continuation_ids = list(gen_out["generated_ids"])
                    full_text = prefix_text + tokenizer.decode(continuation_ids, skip_special_tokens=False)
                    generated_ids = prefix_ids + continuation_ids
                    del prefix_ids_t
                    del full_ids
                else:
                    full_text = prefix_text
                    generated_ids = prefix_ids
                    gen_out = {"stop_reason": "max_new_tokens"}
                checkpoint_meta = {
                    "first_think_budget_tokens": int(ckpt.get("first_think_budget_tokens", 0) or 0),
                    "first_think_budget_triggered": bool(ckpt.get("first_think_budget_triggered", False)),
                    "first_think_budget_forced_close_applied": bool(
                        ckpt.get("first_think_budget_forced_close_applied", False)
                    ),
                    "first_think_budget_forced_close_text": ckpt.get("first_think_budget_forced_close_text"),
                    "prefix_tokens_before_plain_continuation": len(prefix_ids),
                }

            stripped_text = strip_think_blocks(full_text)
            metrics = {
                "think_balanced": think_balance_ok(full_text),
                "expected_hit_raw": _expected_hit(full_text, expected_regex),
                "expected_hit": _expected_hit(stripped_text, expected_regex),
            }

            rec: Dict[str, object] = {
                "task_id": task_id,
                "user_prompt": user_prompt,
                "expected_regex": expected_regex,
                "reference_output": task.get("reference_output"),
                "system_prompt_file": args.system_prompt_file,
                "disable_model_thinking": bool(args.disable_model_thinking),
                "checkpoint_meta": checkpoint_meta,
                "full_text": full_text,
                "stripped_text": stripped_text,
                "new_tokens": len(generated_ids),
                "stop_reason": str(gen_out.get("stop_reason", "unknown")),
                "metrics": metrics,
                "model_path": model_path,
            }
            records_by_task[task_id] = rec

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.save_task_texts:
                _write_task_texts(task_dump_root, rec)

            ordered_now = _ordered_records(tasks, records_by_task)
            model_summary = summarize(ordered_now)
            model_summary["config"] = {
                "system_prompt_file": args.system_prompt_file,
                "disable_model_thinking": bool(args.disable_model_thinking),
                "first_think_budget_tokens": int(args.first_think_budget_tokens or 0),
                "first_think_early_stop_text_file": args.first_think_early_stop_text_file,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            summary_path.write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            global_summary["models"][model_path] = model_summary
            (output_dir / "summary_all_models.json").write_text(
                json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if args.print_full_output:
                print("\n---------------- FULL OUTPUT ----------------")
                print(f"model={model_path}")
                print(f"task={task_id}")
                print(f"stop_reason={rec['stop_reason']}")
                print(f"metrics={json.dumps(metrics, ensure_ascii=False)}")
                print(full_text)
                print("\n---------------------------------------------\n")

            del prompt_ids
            if past is not None:
                del past
            if logits is not None:
                del logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[resume] saved:\n- {jsonl_path}\n- {summary_path}")

    print(f"\n[resume] done: {output_dir / 'summary_all_models.json'}")


if __name__ == "__main__":
    main()
