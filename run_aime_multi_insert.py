#!/usr/bin/env python3
"""
AIME repeated-insertion runner.

Behavior:
- Let the model produce its native first <think>.
- After that think closes, wait for ~300-400 visible-body tokens and insert <think>.
- After each later inserted think closes, again wait for ~300-400 visible-body tokens and insert again.
- Repeat until EOS or the post-insertion generation budget is exhausted.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import torch

from test_close_suite import (
    _think_balance_delta,
    _truncate_first_think_words,
    build_concise_branch_meta,
    evaluate_branch,
    summarize,
)
from think_branch_common import (
    _ends_with_subseq,
    _piece_hits_sentence_boundary,
    build_prompt_ids,
    compose_system_prompt,
    generate_until_checkpoint,
    load_model,
    load_tasks_jsonl,
    model_label,
    prefill_kv,
    top_p_sample,
)


def _safe_name(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run AIME with repeated mid-generation <think> insertion.")
    p.add_argument("--model-paths", required=True, help="Comma-separated local model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument(
        "--tasks-file",
        default=str(here / "data" / "tasks_aime2025.jsonl"),
        help="AIME task jsonl path.",
    )
    p.add_argument(
        "--output-dir",
        default=str(here / "outputs" / "aime" / "suite_aime2025_multi_insert"),
        help="Output directory for results/summary.",
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
    p.add_argument(
        "--max-new-after",
        type=int,
        default=1200,
        help="Total model-generated token budget after the first injected <think>.",
    )
    p.add_argument(
        "--min-b-tokens-before-eos",
        type=int,
        default=64,
        help="Suppress EOS for at least this many generated tokens after each injection.",
    )
    p.add_argument(
        "--max-reinserts",
        type=int,
        default=64,
        help="Hard cap on the total number of inserted <think> blocks (including the first one).",
    )
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--save-task-texts", action="store_true", help="Save branch full texts per task.")
    p.add_argument("--print-full-output", action="store_true", help="Print full B outputs to stdout.")
    return p.parse_args()


@torch.inference_mode()
def generate_until_next_insert_from_state(
    *,
    model,
    tokenizer,
    past_key_values,
    logits: torch.Tensor,
    max_new_tokens: int,
    checkpoint_mid_min_tokens: int,
    checkpoint_mid_max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    min_tokens_before_eos: int = 0,
) -> Dict[str, object]:
    eos_id = tokenizer.eos_token_id
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    device = model.get_input_embeddings().weight.device

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    seen_current_think_end = False
    think_end_token_pos = -1
    tokens_after_think = 0
    stop_reason = "max_new_tokens"
    punct_stop_piece = None
    punct_stop_kind = None
    insert_ready = False

    generated_ids: List[int] = []
    generated_text_parts: List[str] = []

    lo = max(0, int(checkpoint_mid_min_tokens))
    hi = max(lo, int(checkpoint_mid_max_tokens))

    for _ in range(max_new_tokens):
        sample_logits = logits
        if len(generated_ids) < int(min_tokens_before_eos) and eos_id is not None:
            sample_logits = logits.clone()
            sample_logits[:, eos_id] = float("-inf")

        next_token = top_p_sample(sample_logits, temperature, top_p, gen)
        tid = int(next_token.item())
        if tid == eos_id:
            stop_reason = "eos"
            break

        out = model(next_token.to(device), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

        generated_ids.append(tid)
        piece = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text_parts.append(piece)

        just_seen_current_think_end = False
        if (not seen_current_think_end) and _ends_with_subseq(generated_ids, think_end_ids):
            seen_current_think_end = True
            think_end_token_pos = len(generated_ids)
            tokens_after_think = 0
            just_seen_current_think_end = True

        if seen_current_think_end and think_end_token_pos >= 0:
            tokens_after_think = len(generated_ids) - think_end_token_pos

        if seen_current_think_end and (not just_seen_current_think_end):
            hit_punct = tokens_after_think >= lo and _piece_hits_sentence_boundary(piece)
            hit_upper = tokens_after_think >= hi
            if hit_punct or hit_upper:
                stop_reason = "insert_sentence_boundary" if hit_punct else "insert_max_tokens"
                punct_stop_piece = piece if piece else None
                punct_stop_kind = "sentence_boundary" if hit_punct else "max_tokens"
                insert_ready = True
                break

    generated_text = "".join(generated_text_parts)
    return {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "past_key_values": past_key_values,
        "logits": logits,
        "stop_reason": stop_reason,
        "insert_ready": bool(insert_ready),
        "seen_current_think_end": bool(seen_current_think_end),
        "tokens_after_think": int(tokens_after_think),
        "checkpoint_punct_stop_piece": punct_stop_piece,
        "checkpoint_punct_stop_kind": punct_stop_kind,
    }


@torch.inference_mode()
def append_inject_text_to_state(
    *,
    model,
    tokenizer,
    device,
    past_key_values,
    inject_text: str,
) -> Dict[str, object]:
    inject_ids = tokenizer.encode(inject_text, add_special_tokens=False)
    if not inject_ids:
        raise ValueError("inject_text must encode to at least one token")
    inject_ids_t = torch.tensor([inject_ids], dtype=torch.long, device=device)
    out = model(inject_ids_t, past_key_values=past_key_values, use_cache=True)
    return {
        "past_key_values": out.past_key_values,
        "logits": out.logits[:, -1, :],
        "inject_token_count": len(inject_ids),
    }


def run_task_multi_insert(
    *,
    model,
    tokenizer,
    device,
    system_prompt: str,
    task: Dict[str, object],
    inject_text: str,
    first_think_early_stop_text: str,
    checkpoint_mid_min_tokens: int,
    checkpoint_mid_max_tokens: int,
    max_prefix_tokens: int,
    first_think_budget_tokens: int,
    max_new_after: int,
    min_b_tokens_before_eos: int,
    max_reinserts: int,
    temperature: float,
    top_p: float,
    seed: int,
    chunk_size: int,
    enable_first_think_max_words: bool,
    first_think_max_words: int,
    enable_first_think_smooth_close: bool,
    first_think_smooth_close_text: str,
) -> Dict[str, object]:
    user_prompt = str(task["user_prompt"])
    expected_answer = task.get("expected_answer")
    expected_regex = task.get("expected_regex")
    reference_output = task.get("reference_output")

    used_system_prompt = compose_system_prompt(system_prompt, prompt_mode="baseline")
    prompt_ids = build_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=used_system_prompt,
        user_prompt=user_prompt,
        device=device,
        enable_thinking=True,
    )

    ckpt = generate_until_checkpoint(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        delay_tokens_after_first_think_end=0,
        max_new_tokens=max_prefix_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        checkpoint_mode="think_end_punct",
        checkpoint_regex=None,
        checkpoint_mid_min_tokens=checkpoint_mid_min_tokens,
        checkpoint_mid_max_tokens=checkpoint_mid_max_tokens,
        checkpoint_mid_avoid_final_regex=r"(?i)\bfinal\s*:|\bfinal answer\b",
        first_think_budget_tokens=first_think_budget_tokens,
        first_think_early_stop_text=first_think_early_stop_text,
        chunk_size=chunk_size,
        print_stream=False,
    )

    prefix_ids = ckpt["generated_ids"]
    prefix_text = ckpt.get("generated_text") or tokenizer.decode(prefix_ids, skip_special_tokens=False)
    prefix_think_limit_meta = {"applied": False, "reason": "disabled"}
    if bool(enable_first_think_max_words) and int(first_think_max_words) > 0:
        prefix_text, prefix_think_limit_meta = _truncate_first_think_words(
            prefix_text,
            int(first_think_max_words),
            smooth_close_enabled=bool(enable_first_think_smooth_close),
            smooth_close_text=first_think_smooth_close_text,
        )

    prefix_think_gap = _think_balance_delta(prefix_text)
    inject_opens_think = ("<think>" in inject_text) and ("</think>" not in inject_text)
    forced_close_for_inject = bool(inject_opens_think and prefix_think_gap > 0)
    if forced_close_for_inject:
        prefix_text = prefix_text + ("\n</think>" * prefix_think_gap) + "\n"

    branch_b_prefix_text = prefix_text
    prefix_ids_local = tokenizer.encode(branch_b_prefix_text, add_special_tokens=False)
    prefix_ids_t = torch.tensor([prefix_ids_local], dtype=torch.long, device=device)
    full_ids_base = torch.cat([prompt_ids, prefix_ids_t], dim=1)
    past_b, logits_b = prefill_kv(model, full_ids_base, chunk_size=chunk_size)

    del prefix_ids_t
    del full_ids_base
    del prompt_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inject_state = append_inject_text_to_state(
        model=model,
        tokenizer=tokenizer,
        device=device,
        past_key_values=past_b,
        inject_text=inject_text,
    )
    past_b = inject_state["past_key_values"]
    logits_b = inject_state["logits"]

    continuation_parts: List[str] = [inject_text]
    total_generated_after_first_inject = 0
    remaining_budget = int(max_new_after)
    effective_min_b = max(0, int(min_b_tokens_before_eos)) if inject_opens_think else 0
    insertion_events: List[Dict[str, object]] = [
        {
            "insert_index": 1,
            "kind": "initial_after_first_think",
            "source": "checkpoint_tail",
            "checkpoint_punct_stop_kind": ckpt.get("checkpoint_punct_stop_kind"),
            "prefix_tokens": len(prefix_ids_local),
        }
    ]
    final_stop_reason = "max_new_tokens"

    for cycle in range(max(0, int(max_reinserts))):
        if remaining_budget <= 0:
            final_stop_reason = "max_new_tokens"
            break

        seg = generate_until_next_insert_from_state(
            model=model,
            tokenizer=tokenizer,
            past_key_values=past_b,
            logits=logits_b,
            max_new_tokens=remaining_budget,
            checkpoint_mid_min_tokens=checkpoint_mid_min_tokens,
            checkpoint_mid_max_tokens=checkpoint_mid_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + 2 + 97 * cycle,
            min_tokens_before_eos=effective_min_b,
        )

        seg_text = str(seg.get("generated_text") or "")
        seg_ids = list(seg.get("generated_ids") or [])
        continuation_parts.append(seg_text)
        total_generated_after_first_inject += len(seg_ids)
        remaining_budget -= len(seg_ids)

        past_b = seg["past_key_values"]
        logits_b = seg["logits"]
        final_stop_reason = str(seg.get("stop_reason") or "unknown")

        if not bool(seg.get("insert_ready")):
            break
        if remaining_budget <= 0:
            break
        if len(insertion_events) >= int(max_reinserts):
            final_stop_reason = "max_reinserts"
            break

        inject_state = append_inject_text_to_state(
            model=model,
            tokenizer=tokenizer,
            device=device,
            past_key_values=past_b,
            inject_text=inject_text,
        )
        past_b = inject_state["past_key_values"]
        logits_b = inject_state["logits"]
        continuation_parts.append(inject_text)
        insertion_events.append(
            {
                "insert_index": len(insertion_events) + 1,
                "kind": "reinsert_after_think",
                "stop_reason_before_insert": str(seg.get("stop_reason")),
                "generated_tokens_before_insert": len(seg_ids),
                "seen_current_think_end": bool(seg.get("seen_current_think_end")),
                "tokens_after_think": int(seg.get("tokens_after_think", 0)),
                "checkpoint_punct_stop_kind": seg.get("checkpoint_punct_stop_kind"),
                "checkpoint_punct_stop_piece": seg.get("checkpoint_punct_stop_piece"),
                "remaining_budget_after_insert": int(remaining_budget),
            }
        )

    cont_b_raw = "".join(continuation_parts)
    full_b = branch_b_prefix_text + cont_b_raw
    eval_b = evaluate_branch(branch_b_prefix_text, cont_b_raw, full_b, expected_regex)

    return {
        "task_id": task.get("id", ""),
        "user_prompt": user_prompt,
        "expected_answer": expected_answer,
        "expected_regex": expected_regex,
        "reference_output": reference_output,
        "prompt_mode": "baseline",
        "system_prompt_used": used_system_prompt,
        "checkpoint_meta": {
            "checkpoint_mode": "think_end_punct",
            "checkpoint_regex": None,
            "seen_anchor": bool(ckpt.get("seen_anchor", False)),
            "counter_after_anchor": int(ckpt.get("counter_after_anchor", 0)),
            "tokens_after_first_think": int(ckpt.get("tokens_after_first_think", 0)),
            "checkpoint_mid_min_tokens": int(ckpt.get("checkpoint_mid_min_tokens", 0)),
            "checkpoint_mid_max_tokens": int(ckpt.get("checkpoint_mid_max_tokens", 0)),
            "checkpoint_punct_stop_piece": ckpt.get("checkpoint_punct_stop_piece"),
            "checkpoint_punct_stop_kind": ckpt.get("checkpoint_punct_stop_kind"),
            "first_think_budget_tokens": int(ckpt.get("first_think_budget_tokens", 0) or 0),
            "first_think_budget_triggered": bool(ckpt.get("first_think_budget_triggered", False)),
            "first_think_budget_forced_close_applied": bool(
                ckpt.get("first_think_budget_forced_close_applied", False)
            ),
            "first_think_budget_forced_close_text": ckpt.get("first_think_budget_forced_close_text"),
        },
        "prefix_seen_first_think_end": bool(ckpt["seen_first_think_end"]),
        "prefix_tokens": len(prefix_ids_local),
        "prefix_think_limit_meta": prefix_think_limit_meta,
        "prefix_step_wait_meta": {"applied": False, "reason": "not_used_in_multi_insert"},
        "insert_meta": {
            "inject_text": inject_text,
            "inject_opens_think": bool(inject_opens_think),
            "mode": "multi_insert",
            "prefix_open_think_before_inject": max(0, prefix_think_gap),
            "prefix_forced_close_for_inject": max(0, prefix_think_gap) if forced_close_for_inject else 0,
            "total_insertions": len(insertion_events),
            "max_reinserts": int(max_reinserts),
            "remaining_budget_tokens": int(remaining_budget),
            "events": insertion_events,
        },
        "branch_mode": "b",
        "branch_A": {"skipped": True, "reason": "multi_insert_runner"},
        "branch_B": {
            "prefix_text": branch_b_prefix_text,
            "continuation_text_raw": cont_b_raw,
            "continuation_text": cont_b_raw,
            "full_text": full_b,
            "new_tokens": int(total_generated_after_first_inject),
            "stop_reason": final_stop_reason,
            "retry_info": {
                "retry_times": 0,
                "min_tokens_before_eos": effective_min_b,
                "retry_reasons": [],
                "forced_close_think": 0,
            },
            "multi_insert_info": {
                "total_insertions": len(insertion_events),
                "remaining_budget_tokens": int(remaining_budget),
                "events": insertion_events,
            },
            "match_cover": {"mode": "disabled", "trimmed_chars": 0},
            "cross_think_cover": {"mode": "disabled", "trimmed_chars": 0},
            "metrics": eval_b,
        },
    }


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_jsonl(Path(args.tasks_file))
    model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]

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

    global_summary: Dict[str, object] = {
        "task_source": f"jsonl:{Path(args.tasks_file).resolve()}",
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
            print(f"[{i}/{len(tasks)}] {task.get('id', 'task')}")
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
            model_records.append(rec)

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

            if args.save_task_texts:
                task_name = _safe_name(str(rec.get("task_id", f"task_{i}")))
                task_dir = task_dump_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "branch_B.full.txt").write_text(
                    rec["branch_B"]["full_text"], encoding="utf-8"
                )
                meta_payload: Dict[str, object] = {
                    "task_id": rec.get("task_id"),
                    "expected_answer": rec.get("expected_answer"),
                    "expected_regex": rec.get("expected_regex"),
                    "branch_B": build_concise_branch_meta(rec["branch_B"]),
                    "multi_insert_info": rec["branch_B"].get("multi_insert_info", {}),
                }
                (task_dir / "meta.json").write_text(
                    json.dumps(meta_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        jsonl_path = output_dir / f"{model_tag}.results.jsonl"
        summary_path = output_dir / f"{model_tag}.summary.json"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in model_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        model_summary = summarize(model_records)
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
