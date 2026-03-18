#!/usr/bin/env python3
from __future__ import annotations
"""
Direct AIME evaluation without mid-generation insertion or corruption.

Purpose:
- test the model's plain AIME ability
- keep native model thinking on by default
- generate one direct completion per task and score by expected_regex
"""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def _expected_hit(text: str, expected_regex: Optional[str]) -> Optional[bool]:
    if not expected_regex:
        return None
    return re.search(expected_regex, text, re.IGNORECASE | re.DOTALL) is not None


def _build_prompt_ids(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    device: Any,
    enable_thinking: bool,
) -> torch.Tensor:
    from think_branch_common import ensure_input_ids_tensor

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    x = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    return ensure_input_ids_tensor(x, device=device)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run direct AIME evaluation without insertion.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument(
        "--tasks-file",
        default=str(here / "data" / "tasks_aime2025.jsonl"),
        help="AIME task jsonl path.",
    )
    p.add_argument(
        "--output-dir",
        default=str(here / "outputs" / "aime" / "suite_aime2025_plain"),
        help="Output directory for results and summary.",
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
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--save-task-texts", action="store_true")
    p.add_argument("--print-full-output", action="store_true")
    return p.parse_args()


def summarize(records: List[Dict[str, object]]) -> Dict[str, object]:
    def _rate(vals: List[bool]) -> Optional[float]:
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    think_ok: List[bool] = []
    expected_flags: List[bool] = []
    expected_raw_flags: List[bool] = []
    new_tokens: List[int] = []
    budget_triggered: List[bool] = []
    budget_forced_close: List[bool] = []

    for rec in records:
        metrics = rec.get("metrics", {}) or {}
        checkpoint_meta = rec.get("checkpoint_meta", {}) or {}
        think_ok.append(bool(metrics.get("think_balanced", False)))
        hit = metrics.get("expected_hit")
        if hit is not None:
            expected_flags.append(bool(hit))
        hit_raw = metrics.get("expected_hit_raw")
        if hit_raw is not None:
            expected_raw_flags.append(bool(hit_raw))
        new_tokens.append(int(rec.get("new_tokens", 0)))
        if "first_think_budget_triggered" in checkpoint_meta:
            budget_triggered.append(bool(checkpoint_meta.get("first_think_budget_triggered", False)))
            budget_forced_close.append(bool(checkpoint_meta.get("first_think_budget_forced_close_applied", False)))

    return {
        "n_tasks": len(records),
        "think_balanced_rate": _rate(think_ok),
        "expected_hit_rate": _rate(expected_flags) if expected_flags else None,
        "expected_hit_raw_rate": _rate(expected_raw_flags) if expected_raw_flags else None,
        "avg_new_tokens": (sum(new_tokens) / len(new_tokens)) if new_tokens else None,
        "first_think_budget_triggered_rate": _rate(budget_triggered) if budget_triggered else None,
        "first_think_budget_forced_close_rate": _rate(budget_forced_close) if budget_forced_close else None,
    }


def main() -> None:
    args = parse_args()
    import torch
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
        print(f"\n===== Running model: {model_path} =====")
        loaded = load_model(model_path, dtype_name=args.dtype)
        tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

        model_records: List[Dict[str, object]] = []
        model_tag = model_label(model_path)
        task_dump_root = output_dir / f"{model_tag}.task_texts"
        if args.save_task_texts:
            task_dump_root.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks, start=1):
            task_id = str(task.get("id", f"task_{i}"))
            print(f"[{i}/{len(tasks)}] {task_id}")
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
            model_records.append(rec)

            if args.print_full_output:
                print("\n---------------- FULL OUTPUT ----------------")
                print(f"model={model_path}")
                print(f"task={task_id}")
                print(f"stop_reason={rec['stop_reason']}")
                print(f"metrics={json.dumps(metrics, ensure_ascii=False)}")
                print(full_text)
                print("\n---------------------------------------------\n")

            if args.save_task_texts:
                task_dir = task_dump_root / _safe_name(task_id)
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "full.txt").write_text(full_text, encoding="utf-8")
                (task_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "user_prompt": user_prompt,
                            "expected_regex": expected_regex,
                            "stop_reason": rec["stop_reason"],
                            "new_tokens": rec["new_tokens"],
                            "checkpoint_meta": checkpoint_meta,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            del prompt_ids
            if past is not None:
                del past
            if logits is not None:
                del logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        safe_name = model_tag
        jsonl_path = output_dir / f"{safe_name}.results.jsonl"
        summary_path = output_dir / f"{safe_name}.summary.json"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in model_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        model_summary = summarize(model_records)
        model_summary["config"] = {
            "system_prompt_file": args.system_prompt_file,
            "disable_model_thinking": bool(args.disable_model_thinking),
            "first_think_budget_tokens": int(args.first_think_budget_tokens or 0),
            "first_think_early_stop_text_file": args.first_think_early_stop_text_file,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        summary_path.write_text(
            json.dumps(model_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        global_summary["models"][model_path] = model_summary
        print(f"Saved:\n- {jsonl_path}\n- {summary_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_path = output_dir / "summary_all_models.json"
    global_path.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Global summary: {global_path}")


if __name__ == "__main__":
    main()
