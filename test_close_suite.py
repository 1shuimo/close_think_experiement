import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from think_branch_common import (
    apply_match_cover,
    build_prompt_ids,
    clone_past_key_values,
    compose_system_prompt,
    corrupt_numbers_near_anchor,
    corrupt_prefix_text,
    generate_from_state,
    generate_until_checkpoint,
    load_model,
    load_tasks_jsonl,
    longest_suffix_prefix_overlap,
    prefill_kv,
    think_balance_ok,
)


DEFAULT_SYSTEM_PROMPT = """
You are a careful and rigorous solver.
Use <think>...</think> only when needed.
After </think>, continue from exactly where you paused.
Do not repeat text immediately before <think>.
If the user requests a final line format, obey it strictly.
""".strip()

DEFAULT_INJECT_TEXT = (
    "<think>\n"
    "I might be wrong here. Re-check locally and continue from the paused position.\n"
    "</think>\n"
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch A/B suite for close-think experiments.")
    p.add_argument("--model-paths", required=True, help="Comma-separated model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--tasks-file", default="close/tasks_math.jsonl")
    p.add_argument("--output-dir", default="close/suite_outputs")

    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])
    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--checkpoint-delay", type=int, default=200)
    p.add_argument("--checkpoint-mode", default="think_end", choices=["think_end", "regex"])
    p.add_argument("--checkpoint-regex", default=r"(?i)step\s*3")
    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument("--max-new-after", type=int, default=1000)
    p.add_argument("--inject-text", default=DEFAULT_INJECT_TEXT)
    p.add_argument("--apply-match-cover", action="store_true")
    p.add_argument("--cover-min-exact-overlap", type=int, default=40)
    p.add_argument("--cover-fuzzy-min-len", type=int, default=24)
    p.add_argument("--cover-fuzzy-max-len", type=int, default=160)
    p.add_argument("--cover-fuzzy-ratio", type=float, default=0.92)
    p.add_argument("--save-task-texts", action="store_true", help="Save per-task branch full outputs to txt files.")
    p.add_argument("--print-full-output", action="store_true", help="Print full A/B outputs for each task to stdout.")
    p.add_argument(
        "--corrupt-mode",
        default="number_shift",
        choices=["number_shift", "anchor_number_shift", "none"],
        help="Whether to auto-corrupt prefix before branch A/B.",
    )
    p.add_argument("--corrupt-anchor-regex", default=r"(?i)step\s*3")
    p.add_argument("--corrupt-max-changes", type=int, default=2)
    p.add_argument("--corrupt-window-chars", type=int, default=240)
    return p.parse_args()


def _expected_hit(text: str, expected_regex: Optional[str]) -> Optional[bool]:
    if not expected_regex:
        return None
    return re.search(expected_regex, text, re.IGNORECASE | re.DOTALL) is not None


def evaluate_branch(
    edited_prefix_text: str,
    continuation_text: str,
    full_text: str,
    expected_regex: Optional[str],
) -> Dict[str, object]:
    overlap = longest_suffix_prefix_overlap(edited_prefix_text, continuation_text, max_k=300)
    return {
        "think_balanced": think_balance_ok(full_text),
        "overlap_prefix_to_continuation": overlap,
        "repetition_flag": overlap >= 40,
        "expected_hit": _expected_hit(full_text, expected_regex),
    }


def run_task_ab(
    *,
    model,
    tokenizer,
    device,
    system_prompt: str,
    prompt_mode: str,
    think_word_limit: int,
    task: Dict[str, str],
    temperature: float,
    top_p: float,
    seed: int,
    checkpoint_delay: int,
    checkpoint_mode: str,
    checkpoint_regex: Optional[str],
    max_prefix_tokens: int,
    max_new_after: int,
    chunk_size: int,
    inject_text: str,
    corrupt_mode: str,
    corrupt_anchor_regex: str,
    corrupt_max_changes: int,
    corrupt_window_chars: int,
    apply_match_cover_flag: bool,
    cover_min_exact_overlap: int,
    cover_fuzzy_min_len: int,
    cover_fuzzy_max_len: int,
    cover_fuzzy_ratio: float,
) -> Dict[str, object]:
    task_id = task.get("id", "")
    user_prompt = task["user_prompt"]
    expected_regex = task.get("expected_regex")

    used_system_prompt = compose_system_prompt(
        system_prompt,
        prompt_mode=prompt_mode,
        think_word_limit=think_word_limit,
    )

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
        delay_tokens_after_first_think_end=checkpoint_delay,
        max_new_tokens=max_prefix_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        checkpoint_mode=checkpoint_mode,
        checkpoint_regex=checkpoint_regex,
        chunk_size=chunk_size,
        print_stream=False,
    )
    prefix_ids = ckpt["generated_ids"]
    prefix_text = ckpt.get("generated_text") or tokenizer.decode(prefix_ids, skip_special_tokens=False)

    if corrupt_mode == "number_shift":
        edited_text, corrupt_meta = corrupt_prefix_text(prefix_text)
    elif corrupt_mode == "anchor_number_shift":
        edited_text, corrupt_meta = corrupt_numbers_near_anchor(
            prefix_text,
            anchor_regex=corrupt_anchor_regex,
            max_changes=corrupt_max_changes,
            window_chars=corrupt_window_chars,
        )
    else:
        edited_text, corrupt_meta = prefix_text, {"mode": "none", "changed": False}

    edited_ids = tokenizer.encode(edited_text, add_special_tokens=False)
    edited_ids_t = torch.tensor([edited_ids], dtype=torch.long, device=device)

    # 同一次上下文分叉：先 prefill 一次，再复制 KV 给 A/B 两支
    full_ids_base = torch.cat([prompt_ids, edited_ids_t], dim=1)
    past_base, logits_base = prefill_kv(model, full_ids_base, chunk_size=chunk_size)

    past_a = clone_past_key_values(past_base)
    logits_a = logits_base.clone()
    gen_a = generate_from_state(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_a,
        logits=logits_a,
        max_new_tokens=max_new_after,
        temperature=temperature,
        top_p=top_p,
        seed=seed + 1,
        print_stream=False,
    )
    cont_a_raw = tokenizer.decode(gen_a, skip_special_tokens=False)
    if apply_match_cover_flag:
        cont_a, cover_meta_a = apply_match_cover(
            edited_text,
            cont_a_raw,
            min_exact_overlap=cover_min_exact_overlap,
            fuzzy_min_len=cover_fuzzy_min_len,
            fuzzy_max_len=cover_fuzzy_max_len,
            fuzzy_ratio=cover_fuzzy_ratio,
        )
    else:
        cont_a, cover_meta_a = cont_a_raw, {"mode": "disabled", "trimmed_chars": 0}
    full_a = edited_text + cont_a

    past_b = clone_past_key_values(past_base)
    logits_b = logits_base.clone()
    inject_ids = tokenizer.encode(inject_text, add_special_tokens=False)
    inject_ids_t = torch.tensor([inject_ids], dtype=torch.long, device=device)
    out_inj = model(inject_ids_t, past_key_values=past_b, use_cache=True)
    past_b = out_inj.past_key_values
    logits_b = out_inj.logits[:, -1, :]

    gen_b = generate_from_state(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_b,
        logits=logits_b,
        max_new_tokens=max_new_after,
        temperature=temperature,
        top_p=top_p,
        seed=seed + 2,
        print_stream=False,
    )
    cont_b_raw = tokenizer.decode(gen_b, skip_special_tokens=False)
    if apply_match_cover_flag:
        cont_b, cover_meta_b = apply_match_cover(
            edited_text + inject_text,
            cont_b_raw,
            min_exact_overlap=cover_min_exact_overlap,
            fuzzy_min_len=cover_fuzzy_min_len,
            fuzzy_max_len=cover_fuzzy_max_len,
            fuzzy_ratio=cover_fuzzy_ratio,
        )
    else:
        cont_b, cover_meta_b = cont_b_raw, {"mode": "disabled", "trimmed_chars": 0}
    full_b = edited_text + inject_text + cont_b

    eval_a = evaluate_branch(edited_text, cont_a, full_a, expected_regex)
    eval_b = evaluate_branch(edited_text, cont_b, full_b, expected_regex)

    return {
        "task_id": task_id,
        "user_prompt": user_prompt,
        "expected_regex": expected_regex,
        "prompt_mode": prompt_mode,
        "system_prompt_used": used_system_prompt,
        "checkpoint_meta": {
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_regex": checkpoint_regex,
            "seen_anchor": bool(ckpt.get("seen_anchor", False)),
            "counter_after_anchor": int(ckpt.get("counter_after_anchor", 0)),
        },
        "prefix_seen_first_think_end": bool(ckpt["seen_first_think_end"]),
        "prefix_tokens": len(prefix_ids),
        "corrupt_meta": corrupt_meta,
        "branch_A": {
            "continuation_text_raw": cont_a_raw,
            "continuation_text": cont_a,
            "full_text": full_a,
            "new_tokens": len(gen_a),
            "match_cover": cover_meta_a,
            "metrics": eval_a,
        },
        "branch_B": {
            "continuation_text_raw": cont_b_raw,
            "continuation_text": cont_b,
            "full_text": full_b,
            "new_tokens": len(gen_b),
            "match_cover": cover_meta_b,
            "metrics": eval_b,
        },
    }


def summarize(records: List[Dict[str, object]]) -> Dict[str, object]:
    def _rate(vals: List[bool]) -> Optional[float]:
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    s: Dict[str, object] = {"n_tasks": len(records), "branch_A": {}, "branch_B": {}}
    for branch_key in ("branch_A", "branch_B"):
        think_ok: List[bool] = []
        rep_flags: List[bool] = []
        expected_flags: List[bool] = []
        overlaps: List[int] = []
        trimmed_chars: List[int] = []
        cover_modes: Dict[str, int] = {}

        for r in records:
            m = r[branch_key]["metrics"]
            think_ok.append(bool(m["think_balanced"]))
            rep_flags.append(bool(m["repetition_flag"]))
            overlaps.append(int(m["overlap_prefix_to_continuation"]))
            eh = m["expected_hit"]
            if eh is not None:
                expected_flags.append(bool(eh))
            cover = r[branch_key].get("match_cover", {}) or {}
            trimmed = int(cover.get("trimmed_chars", 0) or 0)
            trimmed_chars.append(trimmed)
            mode = str(cover.get("mode", "unknown"))
            cover_modes[mode] = cover_modes.get(mode, 0) + 1

        s[branch_key] = {
            "think_balanced_rate": _rate(think_ok),
            "repetition_rate": _rate(rep_flags),
            "avg_overlap_prefix_to_continuation": (sum(overlaps) / len(overlaps)) if overlaps else None,
            "expected_hit_rate": _rate(expected_flags) if expected_flags else None,
            "avg_trimmed_chars_by_match_cover": (sum(trimmed_chars) / len(trimmed_chars)) if trimmed_chars else None,
            "match_cover_mode_counts": cover_modes,
        }
    return s


def main() -> None:
    args = parse_args()
    tasks = load_tasks_jsonl(Path(args.tasks_file))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]
    global_summary: Dict[str, object] = {
        "tasks_file": str(Path(args.tasks_file).resolve()),
        "n_tasks": len(tasks),
        "models": {},
    }

    for model_path in model_paths:
        print(f"\n===== Running model: {model_path} =====")
        loaded = load_model(model_path, dtype_name=args.dtype)
        tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

        model_records: List[Dict[str, object]] = []
        task_dump_root = output_dir / f"{_safe_name(model_path)}.task_texts"
        if args.save_task_texts:
            task_dump_root.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks, start=1):
            print(f"[{i}/{len(tasks)}] {task.get('id', 'task')}")
            rec = run_task_ab(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_prompt=args.system_prompt,
                prompt_mode=args.prompt_mode,
                think_word_limit=args.think_word_limit,
                task=task,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * i,
                checkpoint_delay=args.checkpoint_delay,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_regex=args.checkpoint_regex if args.checkpoint_mode == "regex" else None,
                max_prefix_tokens=args.max_prefix_tokens,
                max_new_after=args.max_new_after,
                chunk_size=args.chunk_size,
                inject_text=args.inject_text,
                corrupt_mode=args.corrupt_mode,
                corrupt_anchor_regex=args.corrupt_anchor_regex,
                corrupt_max_changes=args.corrupt_max_changes,
                corrupt_window_chars=args.corrupt_window_chars,
                apply_match_cover_flag=bool(args.apply_match_cover),
                cover_min_exact_overlap=args.cover_min_exact_overlap,
                cover_fuzzy_min_len=args.cover_fuzzy_min_len,
                cover_fuzzy_max_len=args.cover_fuzzy_max_len,
                cover_fuzzy_ratio=args.cover_fuzzy_ratio,
            )
            rec["model_path"] = model_path
            model_records.append(rec)

            if args.print_full_output:
                print("\n---------------- FULL OUTPUT ----------------")
                print(f"model={model_path}")
                print(f"task={rec.get('task_id')}")
                print("\n[Branch A Full Text]\n")
                print(rec["branch_A"]["full_text"])
                print("\n[Branch B Full Text]\n")
                print(rec["branch_B"]["full_text"])
                print("\n---------------------------------------------\n")

            if args.save_task_texts:
                task_name = _safe_name(str(rec.get("task_id", f"task_{i}")))
                task_dir = task_dump_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "branch_A.full.txt").write_text(
                    rec["branch_A"]["full_text"], encoding="utf-8"
                )
                (task_dir / "branch_B.full.txt").write_text(
                    rec["branch_B"]["full_text"], encoding="utf-8"
                )
                (task_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "task_id": rec.get("task_id"),
                            "user_prompt": rec.get("user_prompt"),
                            "expected_regex": rec.get("expected_regex"),
                            "checkpoint_meta": rec.get("checkpoint_meta", {}),
                            "corrupt_meta": rec.get("corrupt_meta", {}),
                            "branch_A_metrics": rec["branch_A"]["metrics"],
                            "branch_B_metrics": rec["branch_B"]["metrics"],
                            "branch_A_match_cover": rec["branch_A"].get("match_cover", {}),
                            "branch_B_match_cover": rec["branch_B"].get("match_cover", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", model_path)
        jsonl_path = output_dir / f"{safe_name}.results.jsonl"
        summary_path = output_dir / f"{safe_name}.summary.json"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in model_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        model_summary = summarize(model_records)
        model_summary["config"] = {
            "prompt_mode": args.prompt_mode,
            "think_word_limit": args.think_word_limit,
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint_regex": args.checkpoint_regex if args.checkpoint_mode == "regex" else None,
            "corrupt_mode": args.corrupt_mode,
            "corrupt_anchor_regex": args.corrupt_anchor_regex,
            "corrupt_max_changes": args.corrupt_max_changes,
            "corrupt_window_chars": args.corrupt_window_chars,
            "apply_match_cover": bool(args.apply_match_cover),
            "cover_min_exact_overlap": args.cover_min_exact_overlap,
            "cover_fuzzy_min_len": args.cover_fuzzy_min_len,
            "cover_fuzzy_max_len": args.cover_fuzzy_max_len,
            "cover_fuzzy_ratio": args.cover_fuzzy_ratio,
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
