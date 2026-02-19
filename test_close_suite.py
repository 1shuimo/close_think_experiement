import argparse
import gc
import json
import re
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from think_branch_common import (
    apply_match_cover,
    apply_cross_think_match_cover,
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
    strip_think_blocks,
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
    "I am not fully confident. Re-check and decide again.\n"
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def _load_longproc_data(
    dataset_name: str,
    data_path: str,
    code_path: str,
) -> Tuple[List[Dict[str, object]], Callable]:
    cp = str(Path(code_path).resolve())
    if cp not in sys.path:
        sys.path.insert(0, cp)
    from longproc.longproc_data import load_longproc_data  # type: ignore

    return load_longproc_data(dataset_name, data_path)


def infer_task_family(longproc_task: Optional[str]) -> Optional[str]:
    if not longproc_task:
        return None
    return longproc_task.rsplit("_", 1)[0]


def build_task_format_guidance(task_family: Optional[str]) -> str:
    if task_family == "tom_tracking":
        return (
            "Output format requirement: after reasoning, output belief trace lines beginning with '- Step X:' "
            "and include a final answer line matching the task examples."
        )
    if task_family == "path_traversal":
        return (
            "Output format requirement: output ONLY one route block wrapped with <Route>...</Route>, "
            "one route step per line in the exact sentence template."
        )
    if task_family == "pseudo_to_code":
        return (
            "Output format requirement: output ONLY one C++ code block wrapped with ```cpp and ```."
        )
    if task_family == "html_to_tsv":
        return (
            "Output format requirement: output ONLY one TSV block wrapped with ```tsv and ```."
        )
    if task_family == "countdown":
        return (
            "Output format requirement: include '# Search Procedure' and mark final equations with "
            "<Solution>...</Solution>."
        )
    if task_family == "travel_planning":
        return (
            "Output format requirement: include <Solving Procedure>...</Solving Procedure> and "
            "<Plan>...</Plan>."
        )
    return ""


def format_regex_for_task_family(task_family: Optional[str]) -> Optional[str]:
    if task_family == "tom_tracking":
        return r"(?m)^-\s*Step\s+\d+:"
    if task_family == "path_traversal":
        return r"(?s)<Route>.*</Route>"
    if task_family == "pseudo_to_code":
        return r"(?s)```cpp.*```"
    if task_family == "html_to_tsv":
        return r"(?s)```tsv.*```"
    if task_family == "countdown":
        return r"(?s)<Solution>.*</Solution>"
    if task_family == "travel_planning":
        return r"(?s)<Plan>.*</Plan>"
    return None


def auto_checkpoint_regex(task_family: Optional[str]) -> str:
    if task_family == "tom_tracking":
        return r"(?i)-\s*Step\s*3:"
    if task_family == "path_traversal":
        return r"(?i)<Route>"
    if task_family == "pseudo_to_code":
        return r"(?i)```cpp"
    if task_family == "html_to_tsv":
        return r"(?i)```tsv"
    if task_family == "countdown":
        return r"(?i)#\s*Search Procedure"
    if task_family == "travel_planning":
        return r"(?i)<Solving Procedure>"
    return r"(?i)step\s*3"


def auto_corrupt_anchor_regex(task_family: Optional[str]) -> str:
    if task_family == "tom_tracking":
        return r"(?i)-\s*Step\s*3:"
    if task_family == "path_traversal":
        return r"(?i)<Route>"
    if task_family == "pseudo_to_code":
        return r"(?i)```cpp"
    if task_family == "html_to_tsv":
        return r"(?i)```tsv"
    if task_family == "countdown":
        return r"(?i)#\s*Search Procedure"
    if task_family == "travel_planning":
        return r"(?i)<Solving Procedure>"
    return r"(?i)step\s*3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch A/B suite for close-think experiments.")
    p.add_argument("--model-paths", required=True, help="Comma-separated model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--tasks-file", default="tasks_math.jsonl")
    p.add_argument("--longproc-task", default=None, help="e.g. tom_tracking_0.5k / path_traversal_0.5k")
    p.add_argument("--longproc-data-path", default="../LongProc/data")
    p.add_argument("--longproc-code-path", default="../LongProc")
    p.add_argument("--n-samples", type=int, default=None, help="Sample count limit (for longproc mode).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle longproc samples before slicing.")
    p.add_argument("--output-dir", default="suite_outputs")

    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--system-prompt-file", default=None, help="Load system prompt text from file.")
    p.add_argument("--no-task-format-guidance", action="store_true", help="Disable auto task-specific format guidance.")
    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])
    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--checkpoint-delay", type=int, default=200)
    p.add_argument(
        "--checkpoint-mode",
        default="think_end",
        choices=["think_end", "regex", "think_end_then_regex", "think_end_mid"],
    )
    p.add_argument("--checkpoint-regex", default="__auto__")
    p.add_argument("--checkpoint-mid-min-tokens", type=int, default=80)
    p.add_argument("--checkpoint-mid-max-tokens", type=int, default=220)
    p.add_argument(
        "--checkpoint-mid-avoid-final-regex",
        default=r"(?i)\bfinal\s*:|\bfinal answer\b",
        help="In think_end_mid mode, stop early if this regex appears (set empty string to disable).",
    )
    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument("--max-new-after", type=int, default=1000)
    p.add_argument("--branch-mode", default="ab", choices=["ab", "b"], help="Generate both branches or only branch B.")
    p.add_argument("--min-b-tokens-before-eos", type=int, default=64)
    p.add_argument("--b-retry-times", type=int, default=2, help="Retry branch B if empty or unclosed think.")
    p.add_argument("--auto-close-unclosed-think", action="store_true", help="Force close unmatched <think> tags in prefix/final output.")
    p.add_argument("--inject-text", default=DEFAULT_INJECT_TEXT)
    p.add_argument("--apply-match-cover", action="store_true")
    p.add_argument("--cover-min-exact-overlap", type=int, default=40)
    p.add_argument("--cover-fuzzy-min-len", type=int, default=24)
    p.add_argument("--cover-fuzzy-max-len", type=int, default=160)
    p.add_argument("--cover-fuzzy-ratio", type=float, default=0.92)
    p.add_argument("--apply-cross-think-cover", action="store_true", help="Trim overlap between pre-think body tail and post-think body head in branch B.")
    p.add_argument("--no-eval-strip-think", action="store_true", help="Do not strip <think>...</think> before LongProc evaluator.")
    p.add_argument("--save-task-texts", action="store_true", help="Save per-task branch full outputs to txt files.")
    p.add_argument("--print-full-output", action="store_true", help="Print full A/B outputs for each task to stdout.")
    p.add_argument(
        "--corrupt-mode",
        default="number_shift",
        choices=["number_shift", "anchor_number_shift", "none"],
        help="Whether to auto-corrupt prefix before branch A/B.",
    )
    p.add_argument("--corrupt-anchor-regex", default="__auto__")
    p.add_argument("--corrupt-max-changes", type=int, default=2)
    p.add_argument("--corrupt-window-chars", type=int, default=240)
    p.add_argument(
        "--corrupt-after-first-think",
        action="store_true",
        help="Apply corruption only to text after the first </think> in prefix.",
    )
    p.add_argument(
        "--corrupt-prefer-sign-flip",
        action="store_true",
        help="Try flipping + / - first; fallback to numeric corruption if no operator found.",
    )
    return p.parse_args()


def _expected_hit(text: str, expected_regex: Optional[str]) -> Optional[bool]:
    if not expected_regex:
        return None
    return re.search(expected_regex, text, re.IGNORECASE | re.DOTALL) is not None


def _think_balance_delta(text: str) -> int:
    opens = len(re.findall(r"<think>", text))
    closes = len(re.findall(r"</think>", text))
    return opens - closes


def _split_after_first_think_close(text: str) -> Tuple[str, str, bool]:
    m = re.search(r"</think>", text, re.IGNORECASE)
    if not m:
        return text, "", False
    return text[: m.end()], text[m.end() :], True


def _flip_first_plus_minus_operator(text: str) -> Tuple[str, Dict[str, object]]:
    if not text:
        return text, {"mode": "sign_flip", "changed": False, "reason": "empty_text"}
    for i, ch in enumerate(text):
        if ch not in "+-":
            continue
        j = i - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        k = i + 1
        while k < len(text) and text[k].isspace():
            k += 1
        if j < 0 or k >= len(text):
            continue
        left = text[j]
        right = text[k]
        if (left.isalnum() or left in ")]}") and (right.isalnum() or right in "([{"):
            repl = "-" if ch == "+" else "+"
            edited = text[:i] + repl + text[i + 1 :]
            return edited, {
                "mode": "sign_flip",
                "changed": True,
                "from": ch,
                "to": repl,
                "pos": i,
            }
    return text, {"mode": "sign_flip", "changed": False, "reason": "no_operator_found"}


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
    task_family: Optional[str],
    task_format_guidance: str,
    task: Dict[str, object],
    eval_fn: Optional[Callable],
    temperature: float,
    top_p: float,
    seed: int,
    checkpoint_delay: int,
    checkpoint_mode: str,
    checkpoint_regex: Optional[str],
    checkpoint_mid_min_tokens: int,
    checkpoint_mid_max_tokens: int,
    checkpoint_mid_avoid_final_regex: Optional[str],
    max_prefix_tokens: int,
    max_new_after: int,
    branch_mode: str,
    min_b_tokens_before_eos: int,
    b_retry_times: int,
    auto_close_unclosed_think: bool,
    eval_strip_think: bool,
    chunk_size: int,
    inject_text: str,
    corrupt_mode: str,
    corrupt_anchor_regex: str,
    corrupt_max_changes: int,
    corrupt_window_chars: int,
    corrupt_after_first_think: bool,
    corrupt_prefer_sign_flip: bool,
    apply_match_cover_flag: bool,
    apply_cross_think_cover_flag: bool,
    cover_min_exact_overlap: int,
    cover_fuzzy_min_len: int,
    cover_fuzzy_max_len: int,
    cover_fuzzy_ratio: float,
) -> Dict[str, object]:
    task_id = task.get("id", "")
    user_prompt = task["user_prompt"]
    expected_regex = task.get("expected_regex")
    eval_item = task.get("eval_item")
    reference_output = task.get("reference_output")

    used_system_prompt = compose_system_prompt(
        system_prompt,
        prompt_mode=prompt_mode,
        think_word_limit=think_word_limit,
    )
    if task_format_guidance:
        used_system_prompt = (
            used_system_prompt
            + "\n\nTask-specific output format (MUST follow):\n"
            + task_format_guidance
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
        checkpoint_mid_min_tokens=checkpoint_mid_min_tokens,
        checkpoint_mid_max_tokens=checkpoint_mid_max_tokens,
        checkpoint_mid_avoid_final_regex=checkpoint_mid_avoid_final_regex,
        chunk_size=chunk_size,
        print_stream=False,
    )
    prefix_ids = ckpt["generated_ids"]
    prefix_text = ckpt.get("generated_text") or tokenizer.decode(prefix_ids, skip_special_tokens=False)

    target_prefix = prefix_text
    head_prefix = ""
    scope_meta: Dict[str, object] = {
        "corrupt_after_first_think": bool(corrupt_after_first_think),
        "first_think_closed_found": False,
    }
    if corrupt_after_first_think:
        head_prefix, target_prefix, found_close = _split_after_first_think_close(prefix_text)
        scope_meta["first_think_closed_found"] = bool(found_close)
        if not found_close:
            head_prefix = ""
            target_prefix = prefix_text

    edited_target = target_prefix
    corrupt_meta: Dict[str, object] = {"mode": "none", "changed": False}
    if corrupt_prefer_sign_flip:
        sign_edited, sign_meta = _flip_first_plus_minus_operator(edited_target)
        if bool(sign_meta.get("changed")):
            edited_target = sign_edited
            corrupt_meta = sign_meta

    if not bool(corrupt_meta.get("changed")):
        if corrupt_mode == "number_shift":
            edited_target, corrupt_meta = corrupt_prefix_text(edited_target)
        elif corrupt_mode == "anchor_number_shift":
            edited_target, corrupt_meta = corrupt_numbers_near_anchor(
                edited_target,
                anchor_regex=corrupt_anchor_regex,
                max_changes=corrupt_max_changes,
                window_chars=corrupt_window_chars,
            )
        else:
            edited_target, corrupt_meta = edited_target, {"mode": "none", "changed": False}

    edited_text = head_prefix + edited_target
    corrupt_meta.update(scope_meta)
    corrupt_meta["corrupt_prefer_sign_flip"] = bool(corrupt_prefer_sign_flip)
    corrupt_meta["corrupt_region_start"] = len(head_prefix)
    # Diagnose whether checkpoint cut inside an open <think> block.
    prefix_think_gap = _think_balance_delta(edited_text)
    if auto_close_unclosed_think and prefix_think_gap > 0:
        edited_text = edited_text + ("\n</think>" * prefix_think_gap) + "\n"
    corrupt_meta["prefix_open_think_before_inject"] = max(0, prefix_think_gap)
    corrupt_meta["prefix_auto_closed_think"] = max(0, prefix_think_gap) if auto_close_unclosed_think else 0

    edited_ids = tokenizer.encode(edited_text, add_special_tokens=False)
    edited_ids_t = torch.tensor([edited_ids], dtype=torch.long, device=device)

    # 同一次上下文分叉：先 prefill 一次，再复制 KV 给分支
    full_ids_base = torch.cat([prompt_ids, edited_ids_t], dim=1)
    past_base, logits_base = prefill_kv(model, full_ids_base, chunk_size=chunk_size)

    branch_a_rec: Dict[str, object] = {"skipped": True, "reason": "branch_mode=b"}
    full_a: Optional[str] = None
    if branch_mode == "ab":
        past_a = clone_past_key_values(past_base)
        logits_a = logits_base.clone()
        gen_a_out = generate_from_state(
            model=model,
            tokenizer=tokenizer,
            past_key_values=past_a,
            logits=logits_a,
            max_new_tokens=max_new_after,
            temperature=temperature,
            top_p=top_p,
            seed=seed + 1,
            return_meta=True,
            print_stream=False,
        )
        gen_a = gen_a_out["generated_ids"]
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
        eval_a = evaluate_branch(edited_text, cont_a, full_a, expected_regex)
        branch_a_rec = {
            "continuation_text_raw": cont_a_raw,
            "continuation_text": cont_a,
            "full_text": full_a,
            "new_tokens": len(gen_a),
            "stop_reason": str(gen_a_out.get("stop_reason")),
            "match_cover": cover_meta_a,
            "metrics": eval_a,
            "longproc_eval": None,
        }
        # Release branch A KV/logits before branch B to reduce peak VRAM.
        del past_a
        del logits_a
        del gen_a_out
        del gen_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    past_b = clone_past_key_values(past_base)
    logits_b = logits_base.clone()
    # Base KV is no longer needed after cloning B branch seed.
    del past_base
    del logits_base
    del full_ids_base
    del edited_ids_t
    del prompt_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inject_ids = tokenizer.encode(inject_text, add_special_tokens=False)
    inject_ids_t = torch.tensor([inject_ids], dtype=torch.long, device=device)
    out_inj = model(inject_ids_t, past_key_values=past_b, use_cache=True)
    past_b = out_inj.past_key_values
    logits_b = out_inj.logits[:, -1, :]

    inject_opens_think = ("<think>" in inject_text) and ("</think>" not in inject_text)
    effective_min_b = max(0, int(min_b_tokens_before_eos)) if inject_opens_think else 0
    gen_b: List[int] = []
    gen_b_stop_reason = "unknown"
    retry_reasons: List[str] = []
    past_b_seed = past_b
    logits_b_seed = logits_b
    for attempt in range(max(0, int(b_retry_times)) + 1):
        past_b_try = clone_past_key_values(past_b_seed)
        logits_b_try = logits_b_seed.clone()
        gen_b_out = generate_from_state(
            model=model,
            tokenizer=tokenizer,
            past_key_values=past_b_try,
            logits=logits_b_try,
            max_new_tokens=max_new_after,
            temperature=temperature,
            top_p=top_p,
            seed=seed + 2 + 97 * attempt,
            min_tokens_before_eos=effective_min_b,
            return_meta=True,
            print_stream=False,
        )
        gen_b = gen_b_out["generated_ids"]
        gen_b_stop_reason = str(gen_b_out.get("stop_reason", "unknown"))
        cont_try = tokenizer.decode(gen_b, skip_special_tokens=False)
        if len(gen_b) == 0:
            retry_reasons.append(f"attempt_{attempt}:{gen_b_stop_reason}:empty_generation")
            del past_b_try
            del logits_b_try
            continue
        if inject_opens_think and ("</think>" not in cont_try):
            retry_reasons.append(f"attempt_{attempt}:{gen_b_stop_reason}:unclosed_think")
            del past_b_try
            del logits_b_try
            continue
        del past_b_try
        del logits_b_try
        break

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
    if apply_cross_think_cover_flag:
        cont_b, cross_cover_meta_b = apply_cross_think_match_cover(
            edited_text,
            cont_b,
            min_exact_overlap=max(16, cover_min_exact_overlap // 2),
            fuzzy_min_len=max(12, cover_fuzzy_min_len // 2),
            fuzzy_max_len=max(cover_fuzzy_max_len, 200),
            fuzzy_ratio=cover_fuzzy_ratio,
        )
    else:
        cross_cover_meta_b = {"mode": "disabled", "trimmed_chars": 0}
    full_b = edited_text + inject_text + cont_b
    full_b_think_gap = _think_balance_delta(full_b)
    if auto_close_unclosed_think and full_b_think_gap > 0:
        closer = ("\n</think>" * full_b_think_gap) + "\n"
        cont_b = cont_b + closer
        full_b = full_b + closer

    eval_b = evaluate_branch(edited_text, cont_b, full_b, expected_regex)
    format_re = format_regex_for_task_family(task_family)
    if format_re:
        if full_a is not None and "metrics" in branch_a_rec:
            branch_a_rec["metrics"]["format_hit"] = bool(
                re.search(format_re, full_a, re.IGNORECASE | re.DOTALL)
            )
        eval_b["format_hit"] = bool(re.search(format_re, full_b, re.IGNORECASE | re.DOTALL))

    longproc_eval_b = None
    if eval_fn is not None and eval_item is not None:
        if full_a is not None and "metrics" in branch_a_rec:
            try:
                eval_text_a = strip_think_blocks(full_a) if eval_strip_think else full_a
                m_a, info_a = eval_fn(eval_text_a, eval_item)
                branch_a_rec["longproc_eval"] = {
                    "metrics": m_a,
                    "info": info_a,
                    "eval_strip_think": bool(eval_strip_think),
                }
            except Exception as e:
                branch_a_rec["longproc_eval"] = {"metrics": None, "info": {"error": str(e)}}
        try:
            eval_text_b = strip_think_blocks(full_b) if eval_strip_think else full_b
            m_b, info_b = eval_fn(eval_text_b, eval_item)
            longproc_eval_b = {
                "metrics": m_b,
                "info": info_b,
                "eval_strip_think": bool(eval_strip_think),
            }
        except Exception as e:
            longproc_eval_b = {"metrics": None, "info": {"error": str(e)}}

    return {
        "task_id": task_id,
        "user_prompt": user_prompt,
        "expected_regex": expected_regex,
        "task_family": task_family,
        "reference_output": reference_output,
        "prompt_mode": prompt_mode,
        "system_prompt_used": used_system_prompt,
        "checkpoint_meta": {
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_regex": checkpoint_regex,
            "seen_anchor": bool(ckpt.get("seen_anchor", False)),
            "counter_after_anchor": int(ckpt.get("counter_after_anchor", 0)),
            "tokens_after_first_think": int(ckpt.get("tokens_after_first_think", 0)),
            "checkpoint_mid_min_tokens": int(ckpt.get("checkpoint_mid_min_tokens", 0)),
            "checkpoint_mid_max_tokens": int(ckpt.get("checkpoint_mid_max_tokens", 0)),
            "checkpoint_mid_target_tokens": ckpt.get("checkpoint_mid_target_tokens"),
            "checkpoint_mid_avoid_final_regex": ckpt.get("checkpoint_mid_avoid_final_regex"),
            "checkpoint_mid_early_stop_on_final": bool(
                ckpt.get("checkpoint_mid_early_stop_on_final", False)
            ),
        },
        "prefix_seen_first_think_end": bool(ckpt["seen_first_think_end"]),
        "prefix_tokens": len(prefix_ids),
        "corrupt_meta": corrupt_meta,
        "branch_mode": branch_mode,
        "branch_A": branch_a_rec,
        "branch_B": {
            "continuation_text_raw": cont_b_raw,
            "continuation_text": cont_b,
            "full_text": full_b,
            "new_tokens": len(gen_b),
            "stop_reason": gen_b_stop_reason,
            "retry_info": {
                "retry_times": max(0, int(b_retry_times)),
                "min_tokens_before_eos": effective_min_b,
                "prefix_open_think_before_inject": max(0, prefix_think_gap),
                "retry_reasons": retry_reasons,
                "forced_close_think": max(0, full_b_think_gap) if auto_close_unclosed_think else 0,
            },
            "match_cover": cover_meta_b,
            "cross_think_cover": cross_cover_meta_b,
            "metrics": eval_b,
            "longproc_eval": longproc_eval_b,
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
        format_flags: List[bool] = []
        overlaps: List[int] = []
        trimmed_chars: List[int] = []
        cover_modes: Dict[str, int] = {}
        cross_trimmed_chars: List[int] = []
        cross_modes: Dict[str, int] = {}
        cross_hits: List[bool] = []
        longproc_metric_values: Dict[str, List[float]] = {}

        for r in records:
            branch_obj = r.get(branch_key)
            if not isinstance(branch_obj, dict):
                continue
            m = branch_obj.get("metrics")
            if not isinstance(m, dict):
                continue
            think_ok.append(bool(m["think_balanced"]))
            rep_flags.append(bool(m["repetition_flag"]))
            overlaps.append(int(m["overlap_prefix_to_continuation"]))
            eh = m["expected_hit"]
            if eh is not None:
                expected_flags.append(bool(eh))
            fh = m.get("format_hit")
            if fh is not None:
                format_flags.append(bool(fh))
            cover = branch_obj.get("match_cover", {}) or {}
            trimmed = int(cover.get("trimmed_chars", 0) or 0)
            trimmed_chars.append(trimmed)
            mode = str(cover.get("mode", "unknown"))
            cover_modes[mode] = cover_modes.get(mode, 0) + 1
            cross = branch_obj.get("cross_think_cover", {}) or {}
            c_trimmed = int(cross.get("trimmed_chars", 0) or 0)
            c_mode = str(cross.get("mode", "none"))
            cross_trimmed_chars.append(c_trimmed)
            cross_modes[c_mode] = cross_modes.get(c_mode, 0) + 1
            cross_hits.append(c_mode in {"exact", "fuzzy", "anchor_exact"})

            lp = branch_obj.get("longproc_eval")
            if isinstance(lp, dict):
                mm = lp.get("metrics")
                if isinstance(mm, dict):
                    for k, v in mm.items():
                        if isinstance(v, (int, float)):
                            longproc_metric_values.setdefault(str(k), []).append(float(v))

        branch_summary = {
            "think_balanced_rate": _rate(think_ok),
            "repetition_rate": _rate(rep_flags),
            "avg_overlap_prefix_to_continuation": (sum(overlaps) / len(overlaps)) if overlaps else None,
            "expected_hit_rate": _rate(expected_flags) if expected_flags else None,
            "format_hit_rate": _rate(format_flags) if format_flags else None,
            "avg_trimmed_chars_by_match_cover": (sum(trimmed_chars) / len(trimmed_chars)) if trimmed_chars else None,
            "match_cover_mode_counts": cover_modes,
            "cross_think_match_rate": _rate(cross_hits) if cross_hits else None,
            "avg_trimmed_chars_by_cross_think_cover": (sum(cross_trimmed_chars) / len(cross_trimmed_chars)) if cross_trimmed_chars else None,
            "cross_think_cover_mode_counts": cross_modes,
        }
        if longproc_metric_values:
            branch_summary["longproc_avg_metrics"] = {
                k: (sum(vs) / len(vs)) for k, vs in longproc_metric_values.items() if vs
            }
        s[branch_key] = branch_summary
    return s


def main() -> None:
    args = parse_args()
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    eval_fn: Optional[Callable] = None
    task_source: str
    task_family: Optional[str] = infer_task_family(args.longproc_task)
    task_format_guidance = ""
    if args.longproc_task and not args.no_task_format_guidance:
        task_format_guidance = build_task_format_guidance(task_family)

    resolved_checkpoint_regex: Optional[str]
    if args.checkpoint_mode in {"regex", "think_end_then_regex"}:
        if args.checkpoint_regex == "__auto__":
            resolved_checkpoint_regex = auto_checkpoint_regex(task_family)
        else:
            resolved_checkpoint_regex = args.checkpoint_regex
    else:
        resolved_checkpoint_regex = None
    if args.corrupt_anchor_regex == "__auto__":
        resolved_corrupt_anchor_regex = auto_corrupt_anchor_regex(task_family)
    else:
        resolved_corrupt_anchor_regex = args.corrupt_anchor_regex
    resolved_checkpoint_mid_avoid_final_regex: Optional[str] = (
        args.checkpoint_mid_avoid_final_regex if args.checkpoint_mid_avoid_final_regex else None
    )

    if args.longproc_task:
        longproc_data, eval_fn = _load_longproc_data(
            dataset_name=args.longproc_task,
            data_path=args.longproc_data_path,
            code_path=args.longproc_code_path,
        )
        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(longproc_data)
        if args.n_samples is not None:
            longproc_data = longproc_data[: args.n_samples]
        tasks: List[Dict[str, object]] = []
        for i, d in enumerate(longproc_data, start=1):
            tasks.append(
                {
                    "id": f"{args.longproc_task}_{i}",
                    "user_prompt": d["input_prompt"],
                    "reference_output": d.get("reference_output"),
                    "eval_item": d,
                }
            )
        task_source = f"longproc:{args.longproc_task}"
        print(
            f"[TaskSpec] family={task_family} "
            f"checkpoint_regex={resolved_checkpoint_regex} "
            f"corrupt_anchor_regex={resolved_corrupt_anchor_regex}"
        )
        if task_format_guidance:
            print(f"[TaskSpec] format_guidance={task_format_guidance}")
    else:
        tasks_path = Path(args.tasks_file)
        if not tasks_path.exists():
            alt = Path(tasks_path.name)
            if alt.exists():
                tasks_path = alt
        tasks = load_tasks_jsonl(tasks_path)
        task_source = f"jsonl:{tasks_path.resolve()}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [x.strip() for x in args.model_paths.split(",") if x.strip()]
    global_summary: Dict[str, object] = {
        "task_source": task_source,
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
                task_family=task_family,
                task_format_guidance=task_format_guidance,
                task=task,
                eval_fn=eval_fn,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * i,
                checkpoint_delay=args.checkpoint_delay,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_regex=resolved_checkpoint_regex,
                checkpoint_mid_min_tokens=args.checkpoint_mid_min_tokens,
                checkpoint_mid_max_tokens=args.checkpoint_mid_max_tokens,
                checkpoint_mid_avoid_final_regex=resolved_checkpoint_mid_avoid_final_regex,
                max_prefix_tokens=args.max_prefix_tokens,
                max_new_after=args.max_new_after,
                branch_mode=args.branch_mode,
                min_b_tokens_before_eos=args.min_b_tokens_before_eos,
                b_retry_times=args.b_retry_times,
                auto_close_unclosed_think=bool(args.auto_close_unclosed_think),
                eval_strip_think=not bool(args.no_eval_strip_think),
                chunk_size=args.chunk_size,
                inject_text=args.inject_text,
                corrupt_mode=args.corrupt_mode,
                corrupt_anchor_regex=resolved_corrupt_anchor_regex,
                corrupt_max_changes=args.corrupt_max_changes,
                corrupt_window_chars=args.corrupt_window_chars,
                corrupt_after_first_think=bool(args.corrupt_after_first_think),
                corrupt_prefer_sign_flip=bool(args.corrupt_prefer_sign_flip),
                apply_match_cover_flag=bool(args.apply_match_cover),
                apply_cross_think_cover_flag=bool(args.apply_cross_think_cover),
                cover_min_exact_overlap=args.cover_min_exact_overlap,
                cover_fuzzy_min_len=args.cover_fuzzy_min_len,
                cover_fuzzy_max_len=args.cover_fuzzy_max_len,
                cover_fuzzy_ratio=args.cover_fuzzy_ratio,
            )
            rec["model_path"] = model_path
            model_records.append(rec)
            # Keep model loaded, but force GC between tasks to avoid KV leftovers.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.print_full_output:
                print("\n---------------- FULL OUTPUT ----------------")
                print(f"model={model_path}")
                print(f"task={rec.get('task_id')}")
                print(f"branch_mode={rec.get('branch_mode')}")
                print(f"corrupt_meta={json.dumps(rec.get('corrupt_meta', {}), ensure_ascii=False)}")
                print(f"branch_B_retry={json.dumps(rec['branch_B'].get('retry_info', {}), ensure_ascii=False)}")
                print(f"branch_B_stop_reason={rec['branch_B'].get('stop_reason')}")
                print(f"branch_B_cross_think_cover={json.dumps(rec['branch_B'].get('cross_think_cover', {}), ensure_ascii=False)}")
                if rec["branch_A"].get("full_text") is not None:
                    print("\n[Branch A Full Text]\n")
                    print(rec["branch_A"]["full_text"])
                print("\n[Branch B Full Text]\n")
                print(rec["branch_B"]["full_text"])
                print("\n---------------------------------------------\n")

            if args.save_task_texts:
                task_name = _safe_name(str(rec.get("task_id", f"task_{i}")))
                task_dir = task_dump_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                if rec["branch_A"].get("full_text") is not None:
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
                            "reference_output": rec.get("reference_output"),
                            "expected_regex": rec.get("expected_regex"),
                            "checkpoint_meta": rec.get("checkpoint_meta", {}),
                            "corrupt_meta": rec.get("corrupt_meta", {}),
                            "branch_mode": rec.get("branch_mode"),
                            "branch_A_metrics": rec["branch_A"].get("metrics"),
                            "branch_B_metrics": rec["branch_B"]["metrics"],
                            "branch_B_retry": rec["branch_B"].get("retry_info", {}),
                            "branch_B_stop_reason": rec["branch_B"].get("stop_reason"),
                            "branch_A_match_cover": rec["branch_A"].get("match_cover", {}),
                            "branch_B_match_cover": rec["branch_B"].get("match_cover", {}),
                            "branch_B_cross_think_cover": rec["branch_B"].get("cross_think_cover", {}),
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
            "longproc_task": args.longproc_task,
            "longproc_data_path": args.longproc_data_path if args.longproc_task else None,
            "longproc_code_path": args.longproc_code_path if args.longproc_task else None,
            "n_samples": args.n_samples if args.longproc_task else None,
            "shuffle": bool(args.shuffle) if args.longproc_task else None,
            "task_family": task_family,
            "task_format_guidance": task_format_guidance if args.longproc_task else None,
            "system_prompt_file": args.system_prompt_file,
            "no_task_format_guidance": bool(args.no_task_format_guidance),
            "prompt_mode": args.prompt_mode,
            "think_word_limit": args.think_word_limit,
            "branch_mode": args.branch_mode,
            "min_b_tokens_before_eos": args.min_b_tokens_before_eos,
            "b_retry_times": args.b_retry_times,
            "auto_close_unclosed_think": bool(args.auto_close_unclosed_think),
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint_regex": resolved_checkpoint_regex,
            "checkpoint_mid_min_tokens": args.checkpoint_mid_min_tokens,
            "checkpoint_mid_max_tokens": args.checkpoint_mid_max_tokens,
            "checkpoint_mid_avoid_final_regex": resolved_checkpoint_mid_avoid_final_regex,
            "corrupt_mode": args.corrupt_mode,
            "corrupt_anchor_regex": resolved_corrupt_anchor_regex,
            "corrupt_max_changes": args.corrupt_max_changes,
            "corrupt_window_chars": args.corrupt_window_chars,
            "corrupt_after_first_think": bool(args.corrupt_after_first_think),
            "corrupt_prefer_sign_flip": bool(args.corrupt_prefer_sign_flip),
            "apply_match_cover": bool(args.apply_match_cover),
            "apply_cross_think_cover": bool(args.apply_cross_think_cover),
            "eval_strip_think": not bool(args.no_eval_strip_think),
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
