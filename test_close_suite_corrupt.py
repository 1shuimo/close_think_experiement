#!/usr/bin/env python3
"""
A/B insertion + corruption runner (AIME/jsonl only)

目标：
- 保留改错/扰动能力，供改错实验使用。
- 移除所有 LongProc 相关逻辑。
- 输入来源固定为 jsonl 任务文件。
"""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from think_branch_common import (
    apply_cross_think_match_cover,
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
    "I am not fully confident. Re-check and decide again.\n"
)

DEFAULT_MATH_STEP_USER_GUIDANCE = (
    "Output format requirement (MUST follow): after reasoning, output one step per line using "
    "'Step 0:', 'Step 1:', 'Step 2:', ...; then output one final line that contains ONLY the final "
    "answer value (no extra words)."
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def auto_checkpoint_regex() -> str:
    return r"(?i)step\s*3"


def auto_corrupt_anchor_regex() -> str:
    return auto_checkpoint_regex()


def maybe_append_math_step_guidance(user_prompt: str, math_step_user_guidance: str) -> str:
    if not math_step_user_guidance:
        return user_prompt
    if re.search(r"(?i)step\s*0\s*:", user_prompt):
        return user_prompt
    return user_prompt.rstrip() + "\n\n" + math_step_user_guidance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B insertion runner with corruption support for jsonl tasks (no LongProc)."
    )
    p.add_argument("--model-paths", required=True, help="Comma-separated model paths.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--tasks-file", default="tasks_aime2025.jsonl")
    p.add_argument("--output-dir", default="suite_outputs_corrupt")

    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--system-prompt-file", default=None, help="Load system prompt text from file.")
    p.add_argument(
        "--no-math-step-format-guidance",
        action="store_true",
        help="Disable auto step-format guidance injected into prompts.",
    )
    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])
    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument(
        "--enable-think-word-limit",
        action="store_true",
        help="Enable soft think length hint in enhanced system prompt.",
    )
    p.add_argument(
        "--enable-first-think-max-words",
        action="store_true",
        help="Enable hard truncation for the first <think> block.",
    )
    p.add_argument(
        "--first-think-max-words",
        type=int,
        default=120,
        help="Hard cap on first <think> words in prefix (0 disables).",
    )
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk-size", type=int, default=2048)

    p.add_argument("--checkpoint-delay", type=int, default=0)
    p.add_argument(
        "--checkpoint-mode",
        default="think_end_mid",
        choices=["think_end", "regex", "think_end_then_regex", "think_end_mid"],
    )
    p.add_argument("--checkpoint-regex", default="__auto__")
    p.add_argument("--checkpoint-mid-min-tokens", type=int, default=20)
    p.add_argument("--checkpoint-mid-max-tokens", type=int, default=30)
    p.add_argument(
        "--checkpoint-mid-avoid-final-regex",
        default=r"(?i)\bfinal\s*:|\bfinal answer\b",
        help="In think_end_mid mode, stop early if this regex appears (set empty string to disable).",
    )
    p.add_argument("--max-prefix-tokens", type=int, default=3500)
    p.add_argument(
        "--step-wait-extra-tokens",
        type=int,
        default=1200,
        help="If no Step line appears before inject point, continue prefix generation by this many tokens.",
    )
    p.add_argument(
        "--no-step-fallback-offset-tokens",
        type=int,
        default=300,
        help="If no Step line is found after first think, corrupt near this token offset (<=0 disables).",
    )
    p.add_argument("--max-new-after", type=int, default=1200)
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

    # 改错相关参数
    p.add_argument(
        "--corrupt-mode",
        default="number_shift",
        choices=[
            "number_shift",
            "anchor_number_shift",
            "sign_flip",
            "sign_then_number",
            "sign_and_number",
            "none",
        ],
        help="Whether to auto-corrupt prefix before branch A/B.",
    )
    p.add_argument("--corrupt-anchor-regex", default="__auto__")
    p.add_argument(
        "--corrupt-step-select",
        default="anchor",
        choices=["anchor", "middle"],
        help="For math step-body corruption: use anchor step or pick middle step.",
    )
    p.add_argument("--corrupt-max-changes", type=int, default=2)
    p.add_argument("--corrupt-window-chars", type=int, default=240)
    p.add_argument(
        "--corrupt-min-step",
        type=int,
        default=0,
        help="Only corrupt at or after Step N (e.g. 2 skips Step 0/1).",
    )
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
    p.add_argument(
        "--force-inject-at-corrupt",
        action="store_true",
        help="Force Branch B to inject <think> immediately at the corruption point.",
    )
    p.add_argument(
        "--force-inject-at-sentence-end",
        action="store_true",
        help="When forcing inject at corruption, shift insert point to nearest sentence end after it.",
    )

    p.add_argument("--save-task-texts", action="store_true", help="Save per-task branch full outputs to txt files.")
    p.add_argument("--print-full-output", action="store_true", help="Print full A/B outputs for each task to stdout.")
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


def _collect_step_lines(text: str) -> List[Tuple[int, int, str, int]]:
    step_re = re.compile(r"^\s*Step\s+(\d+)\s*:", re.IGNORECASE)
    spans: List[Tuple[int, int, str, int]] = []
    off = 0
    for line in text.splitlines(keepends=True):
        ln = line.rstrip("\r\n")
        st = off
        ed = off + len(ln)
        m_step = step_re.match(ln)
        if m_step:
            spans.append((st, ed, ln, int(m_step.group(1))))
        off += len(line)
    return spans


def _has_step_marker(text: str) -> bool:
    return re.search(r"(?im)^\s*step\s+\d+\s*:", text or "") is not None


def _truncate_first_think_words(text: str, max_words: int) -> Tuple[str, Dict[str, object]]:
    limit = max(0, int(max_words))
    if limit <= 0:
        return text, {"applied": False, "reason": "disabled"}

    m_open = re.search(r"<think>", text or "", re.IGNORECASE)
    if not m_open:
        return text, {"applied": False, "reason": "no_first_think_open"}

    start = m_open.end()
    m_close_rel = re.search(r"</think>", (text or "")[start:], re.IGNORECASE)
    if m_close_rel:
        close_start = start + m_close_rel.start()
        close_end = start + m_close_rel.end()
        body = text[start:close_start]
        suffix = text[close_end:]
        close_found = True
    else:
        body = text[start:]
        suffix = ""
        close_found = False

    words = re.findall(r"\S+", body)
    if len(words) <= limit and close_found:
        return text, {
            "applied": False,
            "reason": "within_limit",
            "orig_words": len(words),
            "limit_words": limit,
        }

    kept = " ".join(words[:limit]).strip()
    if kept:
        kept = "\n" + kept + "\n"
    new_text = text[: m_open.start()] + "<think>" + kept + "</think>" + suffix
    return new_text, {
        "applied": True,
        "reason": "truncated",
        "orig_words": len(words),
        "new_words": len(words[:limit]),
        "limit_words": limit,
        "close_found": close_found,
    }


def _extend_prefix_for_step(
    *,
    model,
    tokenizer,
    device,
    prompt_ids: torch.Tensor,
    prefix_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    chunk_size: int,
) -> Tuple[str, Dict[str, object]]:
    extra = max(0, int(max_new_tokens))
    if extra <= 0:
        return prefix_text, {"applied": False, "reason": "disabled", "new_tokens": 0, "step_found": _has_step_marker(prefix_text)}

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    prefix_ids_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids, prefix_ids_t], dim=1)
    past, logits = prefill_kv(model, full_ids, chunk_size=chunk_size)
    out = generate_from_state(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past,
        logits=logits,
        max_new_tokens=extra,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        return_meta=True,
        print_stream=False,
    )
    gen_ids = out["generated_ids"]
    ext_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    new_prefix = prefix_text + ext_text
    return new_prefix, {
        "applied": True,
        "reason": "extended_for_step",
        "new_tokens": len(gen_ids),
        "stop_reason": str(out.get("stop_reason", "unknown")),
        "step_found": _has_step_marker(new_prefix),
    }


def _char_pos_from_token_offset(text: str, tokenizer, token_offset: int) -> Tuple[int, int]:
    if not text:
        return 0, 0
    ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids)
    if n <= 0:
        return 0, 0
    k = min(max(0, int(token_offset)), n)
    if k <= 0:
        return 0, n
    prefix = tokenizer.decode(ids[:k], skip_special_tokens=False)
    return min(len(text), len(prefix)), n


def _nearest_sentence_end_near(text: str, pos: int, radius: int = 220) -> int:
    if not text:
        return 0
    p = max(0, min(int(pos), len(text)))
    r = max(0, int(radius))
    lo = max(0, p - r)
    hi = min(len(text), p + r)
    stops = set(".!?。！？;；")

    best_idx: Optional[int] = None
    best_dist: Optional[int] = None
    for i in range(lo, hi):
        if text[i] not in stops:
            continue
        d = abs(i - p)
        if best_idx is None or d < best_dist or (d == best_dist and i >= p):
            best_idx = i
            best_dist = d
    if best_idx is not None:
        return best_idx + 1

    # Fallback: prefer next sentence end direction when nearby punctuation is absent.
    return _advance_to_sentence_end(text, p, max_lookahead=r if r > 0 else 220)


def _corrupt_number_near_token_offset(
    text: str,
    *,
    tokenizer,
    token_offset: int,
    window_chars: int = 260,
) -> Tuple[str, Dict[str, object]]:
    if not text:
        return text, {"mode": "no_step_fallback_number_shift", "changed": False, "reason": "empty_text", "inject_pos": None}

    center_char, total_tokens = _char_pos_from_token_offset(text, tokenizer, token_offset)
    sentence_anchor_char = _nearest_sentence_end_near(
        text,
        center_char,
        radius=max(120, int(window_chars)),
    )
    num_re = re.compile(r"(?<![A-Za-z0-9_.-])(-?\d+)(?![A-Za-z0-9_.-])")
    matches = list(num_re.finditer(text))
    if not matches:
        return text, {
            "mode": "no_step_fallback_number_shift",
            "changed": False,
            "reason": "no_number_found",
            "target_token_offset": int(token_offset),
            "target_char_offset": int(center_char),
            "sentence_anchor_char_offset": int(sentence_anchor_char),
            "total_tokens": int(total_tokens),
            "inject_pos": None,
        }

    w = max(0, int(window_chars))
    candidates = []
    for m in matches:
        mid = (m.start() + m.end()) // 2
        dist = abs(mid - sentence_anchor_char)
        if w == 0 or dist <= w:
            candidates.append((dist, m.start(), m))
    if not candidates:
        candidates = [
            (abs(((m.start() + m.end()) // 2) - sentence_anchor_char), m.start(), m)
            for m in matches
        ]
    candidates.sort(key=lambda x: (x[0], x[1]))
    chosen = candidates[0][2]

    src = int(chosen.group(1))
    dst = src + 1 if src >= 0 else src - 1
    dst_s = str(dst)
    a, b = chosen.start(1), chosen.end(1)
    edited = text[:a] + dst_s + text[b:]
    delta = len(dst_s) - (b - a)
    inject_pos = sentence_anchor_char + (delta if a < sentence_anchor_char else 0)
    inject_pos = max(0, min(len(edited), inject_pos))
    return edited, {
        "mode": "no_step_fallback_number_shift",
        "changed": True,
        "from": src,
        "to": dst,
        "edit_start": int(a),
        "edit_end": int(a + len(dst_s)),
        "inject_pos": int(inject_pos),
        "inject_pos_strategy": "nearest_sentence_end",
        "target_token_offset": int(token_offset),
        "target_char_offset": int(center_char),
        "sentence_anchor_char_offset": int(sentence_anchor_char),
        "total_tokens": int(total_tokens),
        "window_chars": int(w),
    }


def _corrupt_step_body_number(
    text: str,
    *,
    anchor_regex: Optional[str] = None,
    step_select: str = "anchor",
    min_step_no: int = 0,
) -> Tuple[str, Dict[str, object]]:
    if not text:
        return text, {"mode": "step_body_number_shift", "changed": False, "reason": "empty_text", "inject_pos": None}

    num_re = re.compile(r"(?<![A-Za-z0-9_.-])(-?\d+)(?![A-Za-z0-9_.-])")
    anchor_pos = None
    if anchor_regex:
        m_anchor = re.search(anchor_regex, text, re.IGNORECASE | re.DOTALL)
        if m_anchor:
            anchor_pos = m_anchor.start()

    spans = _collect_step_lines(text)
    min_step = max(0, int(min_step_no))
    spans = [sp for sp in spans if sp[3] >= min_step]

    if not spans:
        return text, {
            "mode": "step_body_number_shift",
            "changed": False,
            "reason": "no_eligible_step_lines",
            "min_step_no": min_step,
            "inject_pos": None,
        }

    ordered_spans = spans
    if step_select == "middle":
        n = len(spans)
        mid = n // 2
        idx_order: List[int] = [mid]
        for d in range(1, n):
            r = mid + d
            l = mid - d
            if r < n:
                idx_order.append(r)
            if l >= 0:
                idx_order.append(l)
        ordered_spans = [spans[i] for i in idx_order]
    elif anchor_pos is not None:
        after = [sp for sp in spans if sp[0] >= anchor_pos]
        before = [sp for sp in spans if sp[0] < anchor_pos]
        ordered_spans = after + before

    for st, ed, ln, step_no in ordered_spans:
        colon = ln.find(":")
        if colon < 0:
            continue
        body_start_local = colon + 1
        body = ln[body_start_local:]
        m_num = num_re.search(body)
        if not m_num:
            continue
        a = st + body_start_local + m_num.start()
        b = st + body_start_local + m_num.end()
        src = int(text[a:b])
        dst = src + 1 if src >= 0 else src - 1
        dst_s = str(dst)
        edited = text[:a] + dst_s + text[b:]
        return edited, {
            "mode": "step_body_number_shift",
            "changed": True,
            "from": src,
            "to": dst,
            "edit_start": int(a),
            "edit_end": int(a + len(dst_s)),
            "inject_pos": int(a + len(dst_s)),
            "step_line_start": int(st),
            "step_line_end": int(ed),
            "step_no": int(step_no),
            "min_step_no": min_step,
        }

    return text, {
        "mode": "step_body_number_shift",
        "changed": False,
        "reason": "no_number_in_step_body",
        "min_step_no": min_step,
        "inject_pos": None,
    }


def _advance_to_sentence_end(text: str, start: int, max_lookahead: int = 220) -> int:
    if start < 0:
        return 0
    if start >= len(text):
        return len(text)

    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    if re.match(r"^\s*Step\s+\d+\s*:", line, re.IGNORECASE):
        if line_end < len(text) and text[line_end] == "\n":
            return line_end + 1
        return line_end

    end = min(len(text), start + max(0, int(max_lookahead)))
    stops = set(".!?。！？;；")
    for i in range(start, end):
        if text[i] in stops:
            return i + 1
    nl = text.find("\n", start, end)
    if nl >= 0:
        return nl + 1
    return start


def _flip_first_plus_minus_operator(
    text: str,
    *,
    min_step_no: int = 0,
    anchor_regex: Optional[str] = None,
    step_only: bool = False,
) -> Tuple[str, Dict[str, object]]:
    min_step = max(0, int(min_step_no))
    if not text:
        return text, {"mode": "sign_flip", "changed": False, "reason": "empty_text", "inject_pos": None}

    # If requested, only scan step lines (optionally at/after min_step).
    if step_only or min_step > 0:
        spans = [sp for sp in _collect_step_lines(text) if sp[3] >= min_step]
        if not spans:
            return text, {
                "mode": "sign_flip",
                "changed": False,
                "reason": "no_eligible_step_lines",
                "min_step_no": min_step,
                "step_only": bool(step_only),
                "inject_pos": None,
            }
        if anchor_regex:
            m_anchor = re.search(anchor_regex, text, re.IGNORECASE | re.DOTALL)
            if m_anchor:
                anchor_pos = m_anchor.start()
                after = [sp for sp in spans if sp[0] >= anchor_pos]
                before = [sp for sp in spans if sp[0] < anchor_pos]
                spans = after + before

        for st, ed, ln, step_no in spans:
            colon = ln.find(":")
            if colon < 0:
                continue
            body_start = st + colon + 1
            body_end = ed
            body_text = text[body_start:body_end]
            edited_body, meta = _flip_first_plus_minus_operator(
                body_text, min_step_no=0, anchor_regex=None, step_only=False
            )
            if not bool(meta.get("changed")):
                continue
            edited = text[:body_start] + edited_body + text[body_end:]
            pos_local = int(meta.get("pos", 0))
            inject_local = int(meta.get("inject_pos", pos_local))
            meta_out = {
                "mode": "sign_flip",
                "changed": True,
                "from": meta.get("from"),
                "to": meta.get("to"),
                "pos": int(body_start + pos_local),
                "inject_pos": int(body_start + inject_local),
                "step_line_start": int(st),
                "step_line_end": int(ed),
                "step_no": int(step_no),
                "min_step_no": min_step,
                "step_only": bool(step_only),
            }
            return edited, meta_out

        return text, {
            "mode": "sign_flip",
            "changed": False,
            "reason": "no_operator_found",
            "min_step_no": min_step,
            "step_only": bool(step_only),
            "inject_pos": None,
        }

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
                "inject_pos": i + 1,
            }

    # Fallback: flip one relation operator when +/- is unavailable.
    rel_patterns = [
        (r"<=", ">="),
        (r">=", "<="),
        (r"==", "!="),
        (r"!=", "=="),
        (r"(?<![<>=!])<(?![=])", ">"),
        (r"(?<![<>=!])>(?![=])", "<"),
    ]
    for pat, repl in rel_patterns:
        m = re.search(pat, text)
        if not m:
            continue
        edited = text[: m.start()] + repl + text[m.end() :]
        return edited, {
            "mode": "sign_flip",
            "changed": True,
            "from": text[m.start() : m.end()],
            "to": repl,
            "pos": int(m.start()),
            "inject_pos": int(m.start() + len(repl)),
        }

    return text, {"mode": "sign_flip", "changed": False, "reason": "no_operator_found", "inject_pos": None}


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
    enable_think_word_limit: bool,
    enable_first_think_max_words: bool,
    first_think_max_words: int,
    math_step_user_guidance: str,
    task: Dict[str, object],
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
    step_wait_extra_tokens: int,
    no_step_fallback_offset_tokens: int,
    max_new_after: int,
    branch_mode: str,
    min_b_tokens_before_eos: int,
    b_retry_times: int,
    auto_close_unclosed_think: bool,
    chunk_size: int,
    inject_text: str,
    corrupt_mode: str,
    corrupt_anchor_regex: str,
    corrupt_max_changes: int,
    corrupt_window_chars: int,
    corrupt_min_step: int,
    corrupt_step_select: str,
    corrupt_after_first_think: bool,
    corrupt_prefer_sign_flip: bool,
    force_inject_at_corrupt: bool,
    force_inject_at_sentence_end: bool,
    apply_match_cover_flag: bool,
    apply_cross_think_cover_flag: bool,
    cover_min_exact_overlap: int,
    cover_fuzzy_min_len: int,
    cover_fuzzy_max_len: int,
    cover_fuzzy_ratio: float,
) -> Dict[str, object]:
    task_id = task.get("id", "")
    user_prompt = maybe_append_math_step_guidance(task["user_prompt"], math_step_user_guidance)
    expected_regex = task.get("expected_regex")
    reference_output = task.get("reference_output")

    task_corrupt_note = task.get("corrupt_note")
    task_corrupt_plan = str(task.get("corrupt_plan", "")).strip().lower()
    task_corrupt_anchor_regex = task.get("corrupt_anchor_regex")
    task_corrupt_after_first_think = task.get("corrupt_after_first_think")
    task_corrupt_min_step = task.get("corrupt_min_step")

    effective_corrupt_mode = corrupt_mode
    effective_corrupt_anchor_regex = corrupt_anchor_regex
    effective_corrupt_after_first_think = bool(corrupt_after_first_think)
    effective_corrupt_min_step = max(0, int(corrupt_min_step))
    effective_corrupt_prefer_sign_flip = bool(corrupt_prefer_sign_flip)
    force_sign_only = False
    require_number_after_sign = False

    if task_corrupt_plan in {"sign_flip", "sign_only"}:
        effective_corrupt_prefer_sign_flip = True
        effective_corrupt_mode = "sign_flip"
        force_sign_only = True
    elif task_corrupt_plan in {
        "number_shift",
        "anchor_number_shift",
        "sign_flip",
        "sign_then_number",
        "sign_and_number",
        "none",
    }:
        effective_corrupt_prefer_sign_flip = False
        effective_corrupt_mode = task_corrupt_plan
    elif task_corrupt_plan in {"auto", ""}:
        pass
    else:
        task_corrupt_plan = ""

    if isinstance(task_corrupt_anchor_regex, str) and task_corrupt_anchor_regex:
        effective_corrupt_anchor_regex = task_corrupt_anchor_regex
    if isinstance(task_corrupt_after_first_think, bool):
        effective_corrupt_after_first_think = task_corrupt_after_first_think
    if task_corrupt_min_step is not None:
        try:
            effective_corrupt_min_step = max(0, int(task_corrupt_min_step))
        except Exception:
            pass

    used_system_prompt = compose_system_prompt(
        system_prompt,
        prompt_mode=prompt_mode,
        think_word_limit=think_word_limit,
        enable_think_word_limit=enable_think_word_limit,
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
    prefix_think_limit_meta = {"applied": False, "reason": "disabled"}
    if bool(enable_first_think_max_words) and int(first_think_max_words) > 0:
        prefix_text, prefix_think_limit_meta = _truncate_first_think_words(
            prefix_text, int(first_think_max_words)
        )

    # Fallback: if checkpoint misses step anchor (often due long first think),
    # append extra prefix generation until step marker appears or extra budget is used.
    prefix_step_wait_meta = {
        "applied": False,
        "reason": "not_needed",
        "step_found_before": _has_step_marker(prefix_text),
        "step_found_after": _has_step_marker(prefix_text),
        "new_tokens": 0,
    }
    if (
        checkpoint_mode == "think_end_then_regex"
        and bool(effective_corrupt_after_first_think)
        and (not _has_step_marker(prefix_text))
        and int(step_wait_extra_tokens) > 0
    ):
        prefix_text, ext_meta = _extend_prefix_for_step(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_ids=prompt_ids,
            prefix_text=prefix_text,
            max_new_tokens=int(step_wait_extra_tokens),
            temperature=temperature,
            top_p=top_p,
            seed=seed + 43,
            chunk_size=chunk_size,
        )
        prefix_step_wait_meta = {
            "applied": True,
            "reason": ext_meta.get("reason"),
            "step_found_before": False,
            "step_found_after": bool(ext_meta.get("step_found")),
            "new_tokens": int(ext_meta.get("new_tokens", 0)),
            "stop_reason": ext_meta.get("stop_reason"),
        }

    prefix_tokens_effective = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    target_prefix = prefix_text
    head_prefix = ""
    scope_meta: Dict[str, object] = {
        "corrupt_after_first_think": bool(effective_corrupt_after_first_think),
        "first_think_closed_found": False,
    }
    if effective_corrupt_after_first_think:
        head_prefix, target_prefix, found_close = _split_after_first_think_close(prefix_text)
        scope_meta["first_think_closed_found"] = bool(found_close)
        if not found_close:
            # User requested "after first </think>" corruption:
            # if not found, do NOT corrupt inside the unfinished think.
            head_prefix = prefix_text
            target_prefix = ""
            scope_meta["first_think_forced_close_before_inject"] = True

    edited_target = target_prefix
    corrupt_meta: Dict[str, object] = {"mode": "none", "changed": False}
    sign_meta: Optional[Dict[str, object]] = None
    number_meta: Optional[Dict[str, object]] = None
    sign_changed = False
    step_lines_exist_initial = _has_step_marker(edited_target)
    step_only_for_corrupt = bool(effective_corrupt_after_first_think)

    prefer_sign_first = bool(effective_corrupt_prefer_sign_flip) or effective_corrupt_mode in {
        "sign_flip",
        "sign_then_number",
        "sign_and_number",
    }
    if prefer_sign_first:
        sign_edited, sign_meta = _flip_first_plus_minus_operator(
            edited_target,
            min_step_no=effective_corrupt_min_step,
            anchor_regex=effective_corrupt_anchor_regex,
            step_only=step_only_for_corrupt,
        )
        sign_changed = bool(sign_meta.get("changed"))
        if sign_changed:
            edited_target = sign_edited
            corrupt_meta = sign_meta

    if effective_corrupt_mode == "sign_flip":
        force_sign_only = True
    if effective_corrupt_mode == "sign_and_number":
        require_number_after_sign = True

    if not bool(corrupt_meta.get("changed")) and force_sign_only and sign_meta is not None:
        corrupt_meta = sign_meta

    no_step_fallback_meta: Optional[Dict[str, object]] = None
    if (
        step_only_for_corrupt
        and (not step_lines_exist_initial)
        and int(no_step_fallback_offset_tokens) > 0
    ):
        edited_target, no_step_fallback_meta = _corrupt_number_near_token_offset(
            edited_target,
            tokenizer=tokenizer,
            token_offset=int(no_step_fallback_offset_tokens),
            window_chars=max(120, int(corrupt_window_chars)),
        )
        if bool(no_step_fallback_meta.get("changed")):
            if not bool(corrupt_meta.get("changed")):
                corrupt_meta = no_step_fallback_meta
            else:
                corrupt_meta["no_step_fallback"] = no_step_fallback_meta
    if step_only_for_corrupt and (not step_lines_exist_initial) and no_step_fallback_meta is None:
        no_step_fallback_meta = {
            "mode": "step_scoped_corrupt_guard",
            "changed": False,
            "reason": "no_step_lines_after_first_think",
            "min_step_no": effective_corrupt_min_step,
            "target_token_offset": int(no_step_fallback_offset_tokens),
            "inject_pos": None,
        }
        if not bool(corrupt_meta.get("changed")):
            corrupt_meta = no_step_fallback_meta

    need_number_edit = (not bool(corrupt_meta.get("changed")) and not force_sign_only) or require_number_after_sign
    if step_only_for_corrupt and (not step_lines_exist_initial):
        need_number_edit = False
    if need_number_edit:
        number_mode = effective_corrupt_mode
        if number_mode in {"sign_then_number", "sign_and_number"}:
            number_mode = "anchor_number_shift"

        step_lines_exist = bool(re.search(r"(?im)^\s*step\s+\d+\s*:", edited_target))
        prefer_step_body = step_lines_exist and (effective_corrupt_min_step > 0 or bool(effective_corrupt_after_first_think))

        if number_mode == "number_shift":
            if prefer_step_body:
                edited_target, number_meta = _corrupt_step_body_number(
                    edited_target,
                    anchor_regex=effective_corrupt_anchor_regex,
                    step_select=corrupt_step_select,
                    min_step_no=effective_corrupt_min_step,
                )
                if not bool(number_meta.get("changed")):
                    if effective_corrupt_min_step > 0 or step_only_for_corrupt:
                        number_meta = {
                            "mode": "number_shift",
                            "changed": False,
                            "reason": "no_eligible_step_number_for_min_step",
                            "min_step_no": effective_corrupt_min_step,
                            "inject_pos": None,
                        }
                    else:
                        edited_target, number_meta = corrupt_prefix_text(edited_target)
            else:
                edited_target, number_meta = corrupt_prefix_text(edited_target)
        elif number_mode == "anchor_number_shift":
            if prefer_step_body:
                edited_target, number_meta = _corrupt_step_body_number(
                    edited_target,
                    anchor_regex=effective_corrupt_anchor_regex,
                    step_select=corrupt_step_select,
                    min_step_no=effective_corrupt_min_step,
                )
                if not bool(number_meta.get("changed")):
                    if effective_corrupt_min_step > 0 or step_only_for_corrupt:
                        number_meta = {
                            "mode": "anchor_number_shift",
                            "changed": False,
                            "reason": "no_eligible_step_number_for_min_step",
                            "min_step_no": effective_corrupt_min_step,
                            "inject_pos": None,
                        }
                    else:
                        edited_target, number_meta = corrupt_numbers_near_anchor(
                            edited_target,
                            anchor_regex=effective_corrupt_anchor_regex,
                            max_changes=corrupt_max_changes,
                            window_chars=corrupt_window_chars,
                        )
            else:
                edited_target, number_meta = corrupt_numbers_near_anchor(
                    edited_target,
                    anchor_regex=effective_corrupt_anchor_regex,
                    max_changes=corrupt_max_changes,
                    window_chars=corrupt_window_chars,
                )
        else:
            number_meta = {"mode": number_mode, "changed": False, "reason": "number_mode_skipped", "inject_pos": None}

        if require_number_after_sign:
            number_changed = bool(number_meta and number_meta.get("changed"))
            if sign_changed and number_changed:
                corrupt_meta = {
                    "mode": "sign_and_number",
                    "changed": True,
                    "sign_change": sign_meta,
                    "number_change": number_meta,
                    "inject_pos": number_meta.get("inject_pos"),
                }
            elif sign_changed:
                corrupt_meta = sign_meta if sign_meta is not None else {"mode": "sign_flip", "changed": True}
            elif number_meta is not None:
                corrupt_meta = number_meta
        elif not bool(corrupt_meta.get("changed")) and number_meta is not None:
            corrupt_meta = number_meta

    edited_text = head_prefix + edited_target
    corrupt_meta.update(scope_meta)
    corrupt_meta["corrupt_prefer_sign_flip"] = bool(effective_corrupt_prefer_sign_flip)
    corrupt_meta["corrupt_mode_effective"] = effective_corrupt_mode
    corrupt_meta["corrupt_anchor_regex_effective"] = effective_corrupt_anchor_regex
    corrupt_meta["corrupt_min_step_effective"] = int(effective_corrupt_min_step)
    corrupt_meta["task_corrupt_plan"] = task_corrupt_plan or None
    corrupt_meta["task_corrupt_note"] = task_corrupt_note
    corrupt_meta["force_sign_only"] = bool(force_sign_only)
    corrupt_meta["corrupt_region_start"] = len(head_prefix)

    inject_opens_think = ("<think>" in inject_text) and ("</think>" not in inject_text)
    prefix_think_gap = _think_balance_delta(edited_text)
    forced_close_for_inject = bool(inject_opens_think and prefix_think_gap > 0)
    if forced_close_for_inject:
        edited_text = edited_text + ("\n</think>" * prefix_think_gap) + "\n"
    elif auto_close_unclosed_think and prefix_think_gap > 0:
        edited_text = edited_text + ("\n</think>" * prefix_think_gap) + "\n"

    corrupt_meta["prefix_open_think_before_inject"] = max(0, prefix_think_gap)
    corrupt_meta["prefix_auto_closed_think"] = max(0, prefix_think_gap) if auto_close_unclosed_think else 0
    corrupt_meta["prefix_forced_close_for_inject"] = max(0, prefix_think_gap) if forced_close_for_inject else 0

    branch_b_inject_pos = len(edited_text)
    branch_b_force_overlap_applied = False
    inject_pos_local_raw = corrupt_meta.get("inject_pos")
    if (
        force_inject_at_corrupt
        and (not forced_close_for_inject)
        and bool(corrupt_meta.get("changed"))
        and inject_pos_local_raw is not None
    ):
        try:
            inject_pos_local = int(inject_pos_local_raw)
            inject_pos_global = len(head_prefix) + inject_pos_local
            if 0 <= inject_pos_global <= len(edited_text):
                raw_inject_pos_global = inject_pos_global
                if force_inject_at_sentence_end:
                    inject_pos_global = _advance_to_sentence_end(edited_text, inject_pos_global)
                branch_b_inject_pos = inject_pos_global
                branch_b_force_overlap_applied = True
                corrupt_meta["branch_b_inject_pos_raw"] = int(raw_inject_pos_global)
        except Exception:
            pass

    branch_b_prefix_text = edited_text[:branch_b_inject_pos]
    branch_b_suffix_text = edited_text[branch_b_inject_pos:]
    corrupt_meta["force_inject_at_corrupt"] = bool(force_inject_at_corrupt)
    corrupt_meta["branch_b_force_overlap_applied"] = bool(branch_b_force_overlap_applied)
    corrupt_meta["branch_b_inject_pos"] = int(branch_b_inject_pos)
    corrupt_meta["branch_b_suffix_len"] = int(len(branch_b_suffix_text))
    corrupt_meta["force_inject_at_sentence_end"] = bool(force_inject_at_sentence_end)
    corrupt_meta["inject_clamped_to_tail_due_unclosed_think"] = bool(forced_close_for_inject and force_inject_at_corrupt)

    need_base_prefill = (branch_mode == "ab") or (branch_b_inject_pos == len(edited_text))
    past_base = None
    logits_base = None
    full_ids_base = None

    if need_base_prefill:
        edited_ids = tokenizer.encode(edited_text, add_special_tokens=False)
        edited_ids_t = torch.tensor([edited_ids], dtype=torch.long, device=device)
        full_ids_base = torch.cat([prompt_ids, edited_ids_t], dim=1)
        past_base, logits_base = prefill_kv(model, full_ids_base, chunk_size=chunk_size)
        del edited_ids_t

    branch_a_rec: Dict[str, object] = {"skipped": True, "reason": "branch_mode=b"}
    if branch_mode == "ab":
        if past_base is None or logits_base is None:
            raise RuntimeError("Branch A requires base prefill state, but it is missing.")

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
        }

        del past_a
        del logits_a
        del gen_a_out
        del gen_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if branch_b_inject_pos == len(edited_text):
        if past_base is None or logits_base is None:
            raise RuntimeError("Branch B requires base prefill state, but it is missing.")
        past_b = clone_past_key_values(past_base)
        logits_b = logits_base.clone()
    else:
        branch_b_ids = tokenizer.encode(branch_b_prefix_text, add_special_tokens=False)
        branch_b_ids_t = torch.tensor([branch_b_ids], dtype=torch.long, device=device)
        full_ids_b = torch.cat([prompt_ids, branch_b_ids_t], dim=1)
        past_b, logits_b = prefill_kv(model, full_ids_b, chunk_size=chunk_size)
        del branch_b_ids_t
        del full_ids_b

    if past_base is not None:
        del past_base
    if logits_base is not None:
        del logits_base
    if full_ids_base is not None:
        del full_ids_base
    del prompt_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inject_ids = tokenizer.encode(inject_text, add_special_tokens=False)
    inject_ids_t = torch.tensor([inject_ids], dtype=torch.long, device=device)
    out_inj = model(inject_ids_t, past_key_values=past_b, use_cache=True)
    past_b = out_inj.past_key_values
    logits_b = out_inj.logits[:, -1, :]

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
            branch_b_prefix_text + inject_text,
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
            branch_b_prefix_text,
            cont_b,
            min_exact_overlap=max(16, cover_min_exact_overlap // 2),
            fuzzy_min_len=max(12, cover_fuzzy_min_len // 2),
            fuzzy_max_len=max(cover_fuzzy_max_len, 200),
            fuzzy_ratio=cover_fuzzy_ratio,
        )
    else:
        cross_cover_meta_b = {"mode": "disabled", "trimmed_chars": 0}

    full_b = branch_b_prefix_text + inject_text + cont_b
    full_b_think_gap = _think_balance_delta(full_b)
    if auto_close_unclosed_think and full_b_think_gap > 0:
        closer = ("\n</think>" * full_b_think_gap) + "\n"
        cont_b = cont_b + closer
        full_b = full_b + closer

    eval_b = evaluate_branch(branch_b_prefix_text, cont_b, full_b, expected_regex)

    return {
        "task_id": task_id,
        "user_prompt": user_prompt,
        "expected_regex": expected_regex,
        "task_corrupt_note": task_corrupt_note,
        "task_corrupt_plan": task_corrupt_plan or None,
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
        "prefix_think_limit_meta": prefix_think_limit_meta,
        "prefix_step_wait_meta": prefix_step_wait_meta,
        "prefix_seen_first_think_end": bool(ckpt["seen_first_think_end"]),
        "prefix_tokens": int(prefix_tokens_effective),
        "corrupt_meta": corrupt_meta,
        "branch_mode": branch_mode,
        "branch_A": branch_a_rec,
        "branch_B": {
            "prefix_text": branch_b_prefix_text,
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
        cross_trimmed_chars: List[int] = []
        cross_modes: Dict[str, int] = {}
        cross_hits: List[bool] = []

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

        s[branch_key] = {
            "think_balanced_rate": _rate(think_ok),
            "repetition_rate": _rate(rep_flags),
            "avg_overlap_prefix_to_continuation": (sum(overlaps) / len(overlaps)) if overlaps else None,
            "expected_hit_rate": _rate(expected_flags) if expected_flags else None,
            "avg_trimmed_chars_by_match_cover": (sum(trimmed_chars) / len(trimmed_chars)) if trimmed_chars else None,
            "match_cover_mode_counts": cover_modes,
            "cross_think_match_rate": _rate(cross_hits) if cross_hits else None,
            "avg_trimmed_chars_by_cross_think_cover": (sum(cross_trimmed_chars) / len(cross_trimmed_chars)) if cross_trimmed_chars else None,
            "cross_think_cover_mode_counts": cross_modes,
        }
    return s


def _summarize_corrupt_meta(corrupt_meta: Dict[str, object]) -> Dict[str, object]:
    mode = str(corrupt_meta.get("mode", "none"))
    changed = bool(corrupt_meta.get("changed", False))
    out: Dict[str, object] = {"mode": mode, "changed": changed}

    if not changed:
        out["reason"] = corrupt_meta.get("reason")
        return out

    if mode in {"step_body_number_shift", "number_shift", "anchor_fallback_number_shift", "no_step_fallback_number_shift"}:
        out["from"] = corrupt_meta.get("from")
        out["to"] = corrupt_meta.get("to")
        out["edit_start"] = corrupt_meta.get("edit_start")
        out["edit_end"] = corrupt_meta.get("edit_end")
        out["target_token_offset"] = corrupt_meta.get("target_token_offset")
        return out

    if mode == "anchor_number_shift":
        changes = corrupt_meta.get("changes")
        if isinstance(changes, list):
            out["changes"] = changes
            out["n_changes"] = len(changes)
        out["window"] = corrupt_meta.get("window")
        return out

    if mode == "sign_flip":
        out["from"] = corrupt_meta.get("from")
        out["to"] = corrupt_meta.get("to")
        out["pos"] = corrupt_meta.get("pos")
        return out

    if mode == "sign_and_number":
        out["sign_change"] = corrupt_meta.get("sign_change")
        out["number_change"] = corrupt_meta.get("number_change")
        return out

    out["raw"] = corrupt_meta
    return out


def _build_branch_b_view_record(rec: Dict[str, object]) -> Dict[str, object]:
    branch_b = rec.get("branch_B", {}) or {}
    metrics = branch_b.get("metrics", {}) or {}
    corrupt_meta = rec.get("corrupt_meta", {}) or {}
    return {
        "task_id": rec.get("task_id"),
        "corrupt_summary": _summarize_corrupt_meta(corrupt_meta),
        "corrupt_meta": corrupt_meta,
        "branch_B": {
            "stop_reason": branch_b.get("stop_reason"),
            "new_tokens": branch_b.get("new_tokens"),
            "think_balanced": metrics.get("think_balanced"),
            "expected_hit": metrics.get("expected_hit"),
            "full_text": branch_b.get("full_text", ""),
        },
    }


def _write_branch_b_view_markdown(path: Path, model_path: str, records: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Branch B View")
    lines.append("")
    lines.append(f"- model_path: {model_path}")
    lines.append(f"- n_tasks: {len(records)}")
    lines.append("")

    for rec in records:
        view = _build_branch_b_view_record(rec)
        b = view["branch_B"]
        cm = view["corrupt_summary"]
        lines.append(f"## {view.get('task_id', 'task')}")
        lines.append(f"- stop_reason: {b.get('stop_reason')}")
        lines.append(f"- new_tokens: {b.get('new_tokens')}")
        lines.append(f"- think_balanced: {b.get('think_balanced')}")
        lines.append(f"- expected_hit: {b.get('expected_hit')}")
        lines.append(f"- corrupt_summary: {json.dumps(cm, ensure_ascii=False)}")
        lines.append("")
        lines.append("```text")
        lines.append(str(b.get("full_text", "")))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        alt = Path(tasks_path.name)
        if alt.exists():
            tasks_path = alt
    tasks = load_tasks_jsonl(tasks_path)

    math_step_user_guidance = ""
    if not args.no_math_step_format_guidance:
        math_step_user_guidance = DEFAULT_MATH_STEP_USER_GUIDANCE

    resolved_checkpoint_regex: Optional[str]
    if args.checkpoint_mode in {"regex", "think_end_then_regex"}:
        if args.checkpoint_regex == "__auto__":
            resolved_checkpoint_regex = auto_checkpoint_regex()
        else:
            resolved_checkpoint_regex = args.checkpoint_regex
    else:
        resolved_checkpoint_regex = None

    if args.corrupt_anchor_regex == "__auto__":
        resolved_corrupt_anchor_regex = auto_corrupt_anchor_regex()
    else:
        resolved_corrupt_anchor_regex = args.corrupt_anchor_regex

    resolved_checkpoint_mid_avoid_final_regex: Optional[str] = (
        args.checkpoint_mid_avoid_final_regex if args.checkpoint_mid_avoid_final_regex else None
    )

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
                enable_think_word_limit=bool(args.enable_think_word_limit),
                enable_first_think_max_words=bool(args.enable_first_think_max_words),
                first_think_max_words=args.first_think_max_words,
                math_step_user_guidance=math_step_user_guidance,
                task=task,
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
                step_wait_extra_tokens=args.step_wait_extra_tokens,
                no_step_fallback_offset_tokens=args.no_step_fallback_offset_tokens,
                max_new_after=args.max_new_after,
                branch_mode=args.branch_mode,
                min_b_tokens_before_eos=args.min_b_tokens_before_eos,
                b_retry_times=args.b_retry_times,
                auto_close_unclosed_think=bool(args.auto_close_unclosed_think),
                chunk_size=args.chunk_size,
                inject_text=args.inject_text,
                corrupt_mode=args.corrupt_mode,
                corrupt_anchor_regex=resolved_corrupt_anchor_regex,
                corrupt_max_changes=args.corrupt_max_changes,
                corrupt_window_chars=args.corrupt_window_chars,
                corrupt_min_step=args.corrupt_min_step,
                corrupt_step_select=args.corrupt_step_select,
                corrupt_after_first_think=bool(args.corrupt_after_first_think),
                corrupt_prefer_sign_flip=bool(args.corrupt_prefer_sign_flip),
                force_inject_at_corrupt=bool(args.force_inject_at_corrupt),
                force_inject_at_sentence_end=bool(args.force_inject_at_sentence_end),
                apply_match_cover_flag=bool(args.apply_match_cover),
                apply_cross_think_cover_flag=bool(args.apply_cross_think_cover),
                cover_min_exact_overlap=args.cover_min_exact_overlap,
                cover_fuzzy_min_len=args.cover_fuzzy_min_len,
                cover_fuzzy_max_len=args.cover_fuzzy_max_len,
                cover_fuzzy_ratio=args.cover_fuzzy_ratio,
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
                print(f"branch_mode={rec.get('branch_mode')}")
                print(f"task_corrupt_plan={rec.get('task_corrupt_plan')}")
                print(f"task_corrupt_note={rec.get('task_corrupt_note')}")
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
                            "task_corrupt_plan": rec.get("task_corrupt_plan"),
                            "task_corrupt_note": rec.get("task_corrupt_note"),
                            "reference_output": rec.get("reference_output"),
                            "expected_regex": rec.get("expected_regex"),
                            "checkpoint_meta": rec.get("checkpoint_meta", {}),
                            "prefix_think_limit_meta": rec.get("prefix_think_limit_meta", {}),
                            "prefix_step_wait_meta": rec.get("prefix_step_wait_meta", {}),
                            "corrupt_meta": rec.get("corrupt_meta", {}),
                            "branch_mode": rec.get("branch_mode"),
                            "branch_A_metrics": rec["branch_A"].get("metrics"),
                            "branch_B_metrics": rec["branch_B"]["metrics"],
                            "branch_B_prefix_text": rec["branch_B"].get("prefix_text", ""),
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
                branch_b_view = _build_branch_b_view_record(rec)
                (task_dir / "branch_B.view.md").write_text(
                    "\n".join(
                        [
                            f"# {branch_b_view.get('task_id', 'task')}",
                            "",
                            f"- corrupt_summary: {json.dumps(branch_b_view.get('corrupt_summary', {}), ensure_ascii=False)}",
                            f"- stop_reason: {branch_b_view['branch_B'].get('stop_reason')}",
                            f"- new_tokens: {branch_b_view['branch_B'].get('new_tokens')}",
                            f"- think_balanced: {branch_b_view['branch_B'].get('think_balanced')}",
                            f"- expected_hit: {branch_b_view['branch_B'].get('expected_hit')}",
                            "",
                            "```text",
                            str(branch_b_view["branch_B"].get("full_text", "")),
                            "```",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", model_path)
        jsonl_path = output_dir / f"{safe_name}.results.jsonl"
        summary_path = output_dir / f"{safe_name}.summary.json"
        branch_b_view_jsonl_path = output_dir / f"{safe_name}.branch_b_view.jsonl"
        branch_b_view_md_path = output_dir / f"{safe_name}.branch_b_view.md"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in model_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with branch_b_view_jsonl_path.open("w", encoding="utf-8") as f:
            for r in model_records:
                f.write(json.dumps(_build_branch_b_view_record(r), ensure_ascii=False) + "\n")

        _write_branch_b_view_markdown(branch_b_view_md_path, model_path, model_records)

        model_summary = summarize(model_records)
        model_summary["config"] = {
            "system_prompt_file": args.system_prompt_file,
            "no_math_step_format_guidance": bool(args.no_math_step_format_guidance),
            "prompt_mode": args.prompt_mode,
            "think_word_limit": args.think_word_limit,
            "enable_think_word_limit": bool(args.enable_think_word_limit),
            "enable_first_think_max_words": bool(args.enable_first_think_max_words),
            "first_think_max_words": args.first_think_max_words,
            "branch_mode": args.branch_mode,
            "min_b_tokens_before_eos": args.min_b_tokens_before_eos,
            "b_retry_times": args.b_retry_times,
            "auto_close_unclosed_think": bool(args.auto_close_unclosed_think),
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint_regex": resolved_checkpoint_regex,
            "checkpoint_mid_min_tokens": args.checkpoint_mid_min_tokens,
            "checkpoint_mid_max_tokens": args.checkpoint_mid_max_tokens,
            "checkpoint_mid_avoid_final_regex": resolved_checkpoint_mid_avoid_final_regex,
            "step_wait_extra_tokens": args.step_wait_extra_tokens,
            "no_step_fallback_offset_tokens": args.no_step_fallback_offset_tokens,
            "corrupt_mode": args.corrupt_mode,
            "corrupt_anchor_regex": resolved_corrupt_anchor_regex,
            "corrupt_max_changes": args.corrupt_max_changes,
            "corrupt_window_chars": args.corrupt_window_chars,
            "corrupt_min_step": args.corrupt_min_step,
            "corrupt_step_select": args.corrupt_step_select,
            "corrupt_after_first_think": bool(args.corrupt_after_first_think),
            "corrupt_prefer_sign_flip": bool(args.corrupt_prefer_sign_flip),
            "force_inject_at_corrupt": bool(args.force_inject_at_corrupt),
            "force_inject_at_sentence_end": bool(args.force_inject_at_sentence_end),
            "apply_match_cover": bool(args.apply_match_cover),
            "apply_cross_think_cover": bool(args.apply_cross_think_cover),
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

        print(
            "Saved:\n"
            f"- {jsonl_path}\n"
            f"- {summary_path}\n"
            f"- {branch_b_view_jsonl_path}\n"
            f"- {branch_b_view_md_path}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_path = output_dir / "summary_all_models.json"
    global_path.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll done. Global summary: {global_path}")


if __name__ == "__main__":
    main()
