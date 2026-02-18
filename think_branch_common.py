import json
import re
import copy
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def parse_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or "").strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model(model_path: str, dtype_name: str = "bf16") -> LoadedModel:
    dtype = parse_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    return LoadedModel(tokenizer=tokenizer, model=model, device=device)


def compose_system_prompt(
    base_prompt: str,
    *,
    prompt_mode: str = "baseline",
    think_word_limit: int = 60,
) -> str:
    if prompt_mode == "baseline":
        return base_prompt.strip()
    if prompt_mode != "enhanced":
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    appendix = f"""

Add-on protocol for uncertainty handling:
1) If uncertainty appears, enter <think> with first-person local reasoning.
2) Keep <think> concise (about {think_word_limit} words max).
3) End reasoning with </think> explicitly.
4) After </think>, continue exactly from the interrupted position.
5) Do not repeat the immediate previous text; do not restart the answer.
6) Preserve output format and language style.
""".strip()
    return (base_prompt.strip() + "\n\n" + appendix).strip()


def top_p_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    mask = cum > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_idx.gather(-1, sampled)


def ensure_input_ids_tensor(input_ids: object, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    兼容不同 transformers 版本：
    - Tensor
    - BatchEncoding (含 .input_ids 或 data["input_ids"])
    - dict["input_ids"]
    """
    if torch.is_tensor(input_ids):
        t = input_ids
    elif hasattr(input_ids, "input_ids"):
        t = getattr(input_ids, "input_ids")
    elif isinstance(input_ids, dict) and "input_ids" in input_ids:
        t = input_ids["input_ids"]
    elif hasattr(input_ids, "data") and isinstance(getattr(input_ids, "data"), dict) and "input_ids" in input_ids.data:
        t = input_ids.data["input_ids"]
    else:
        raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")

    if isinstance(t, list):
        t = torch.tensor(t, dtype=torch.long)

    if not torch.is_tensor(t):
        raise TypeError(f"input_ids is not a Tensor after extraction: {type(t)}")

    if t.dim() == 1:
        t = t.unsqueeze(0)

    if device is not None:
        t = t.to(device)
    return t


@torch.inference_mode()
def prefill_kv(
    model: AutoModelForCausalLM,
    input_ids: object,
    chunk_size: int = 2048,
) -> Tuple[object, torch.Tensor]:
    past_key_values = None
    logits = None
    model_device = model.get_input_embeddings().weight.device
    input_ids = ensure_input_ids_tensor(input_ids, device=model_device)
    _, seqlen = input_ids.shape

    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        chunk = input_ids[:, start:end]
        out = model(chunk, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

    if logits is None:
        raise RuntimeError("Prefill failed: no logits returned.")
    return past_key_values, logits


def clone_past_key_values(past_key_values: object) -> object:
    """
    为分叉续写复制 KV 缓存，尽量避免不同分支互相污染。
    """
    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        pass

    if isinstance(past_key_values, tuple):
        layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                layers.append(
                    tuple(x.clone() if torch.is_tensor(x) else x for x in layer)
                )
            else:
                layers.append(layer.clone() if torch.is_tensor(layer) else layer)
        return tuple(layers)

    if isinstance(past_key_values, list):
        layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                layers.append(
                    tuple(x.clone() if torch.is_tensor(x) else x for x in layer)
                )
            else:
                layers.append(layer.clone() if torch.is_tensor(layer) else layer)
        return layers

    return past_key_values


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    device: torch.device,
    enable_thinking: bool = True,
) -> torch.Tensor:
    x = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    return ensure_input_ids_tensor(x, device=device)


def _ends_with_subseq(tokens: List[int], subseq: List[int]) -> bool:
    if not subseq:
        return False
    if len(tokens) < len(subseq):
        return False
    return tokens[-len(subseq):] == subseq


@torch.inference_mode()
def generate_until_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    *,
    delay_tokens_after_first_think_end: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    checkpoint_mode: str = "think_end",
    checkpoint_regex: Optional[str] = None,
    chunk_size: int = 2048,
    print_stream: bool = False,
) -> Dict[str, object]:
    eos_id = tokenizer.eos_token_id
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    device = model.get_input_embeddings().weight.device

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    past_key_values, logits = prefill_kv(model, prompt_ids, chunk_size=chunk_size)

    seen_first_think_end = False
    seen_anchor = False
    counter_after_anchor = 0
    generated_ids: List[int] = []
    generated_text_parts: List[str] = []
    regex_obj = re.compile(checkpoint_regex, re.IGNORECASE | re.DOTALL) if checkpoint_regex else None

    for _ in range(max_new_tokens):
        next_token = top_p_sample(logits, temperature, top_p, gen)
        tid = int(next_token.item())
        if tid == eos_id:
            break

        out = model(next_token.to(device), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

        generated_ids.append(tid)
        piece = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text_parts.append(piece)
        if print_stream:
            print(piece, end="", flush=True)

        if (not seen_first_think_end) and _ends_with_subseq(generated_ids, think_end_ids):
            seen_first_think_end = True

        if not seen_anchor:
            joined_text = "".join(generated_text_parts)
            if checkpoint_mode == "think_end":
                if seen_first_think_end:
                    seen_anchor = True
                    counter_after_anchor = 0
            elif checkpoint_mode == "regex":
                if regex_obj and regex_obj.search(joined_text):
                    seen_anchor = True
                    counter_after_anchor = 0
            elif checkpoint_mode == "think_end_then_regex":
                if seen_first_think_end and regex_obj and regex_obj.search(joined_text):
                    seen_anchor = True
                    counter_after_anchor = 0
            else:
                raise ValueError(f"Unsupported checkpoint_mode: {checkpoint_mode}")
        else:
            counter_after_anchor += 1
            if counter_after_anchor >= delay_tokens_after_first_think_end:
                break

    generated_text = "".join(generated_text_parts)
    return {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "seen_first_think_end": seen_first_think_end,
        "counter_after_think_end": counter_after_anchor,
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_regex": checkpoint_regex,
        "seen_anchor": seen_anchor,
        "counter_after_anchor": counter_after_anchor,
    }


@torch.inference_mode()
def generate_from_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    past_key_values: object,
    logits: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    min_tokens_before_eos: int = 0,
    return_meta: bool = False,
    print_stream: bool = False,
) -> Any:
    eos_id = tokenizer.eos_token_id
    device = model.get_input_embeddings().weight.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    generated_ids: List[int] = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        sample_logits = logits
        if len(generated_ids) < min_tokens_before_eos and eos_id is not None:
            # Prevent immediate termination right after branch injection.
            sample_logits = logits.clone()
            sample_logits[:, eos_id] = float("-inf")

        next_token = top_p_sample(sample_logits, temperature, top_p, gen)
        tid = int(next_token.item())
        if tid == eos_id:
            stop_reason = "eos"
            break

        generated_ids.append(tid)
        if print_stream:
            print(tokenizer.decode(next_token[0], skip_special_tokens=False), end="", flush=True)

        out = model(next_token.to(device), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

    if return_meta:
        return {"generated_ids": generated_ids, "stop_reason": stop_reason}
    return generated_ids


def longest_suffix_prefix_overlap(left: str, right: str, max_k: int = 400) -> int:
    if not left or not right:
        return 0
    m = min(max_k, len(left), len(right))
    for k in range(m, 0, -1):
        if left[-k:] == right[:k]:
            return k
    return 0


def apply_match_cover(
    prefix_text: str,
    continuation_text: str,
    *,
    min_exact_overlap: int = 40,
    fuzzy_min_len: int = 24,
    fuzzy_max_len: int = 160,
    fuzzy_ratio: float = 0.92,
) -> Tuple[str, Dict[str, object]]:
    """
    匹配覆盖方案（todo 对应）：
    1) 先做 exact suffix-prefix overlap，足够长则直接裁掉 continuation 开头重复段。
    2) 再做 fuzzy overlap（处理“很像但不完全一样”的重复），满足阈值则裁掉。
    """
    if not continuation_text:
        return continuation_text, {"mode": "none", "trimmed_chars": 0}
    if not prefix_text:
        return continuation_text, {"mode": "none", "trimmed_chars": 0}

    exact_k = longest_suffix_prefix_overlap(prefix_text, continuation_text, max_k=600)
    if exact_k >= min_exact_overlap:
        return continuation_text[exact_k:], {
            "mode": "exact",
            "trimmed_chars": exact_k,
            "exact_overlap": exact_k,
        }

    max_len = min(fuzzy_max_len, len(prefix_text), len(continuation_text))
    best_len = 0
    best_ratio = 0.0
    for l in range(max_len, fuzzy_min_len - 1, -1):
        suffix = prefix_text[-l:]
        head = continuation_text[:l]
        ratio = SequenceMatcher(None, suffix, head).ratio()
        if ratio >= fuzzy_ratio:
            best_len = l
            best_ratio = ratio
            break

    if best_len > 0:
        return continuation_text[best_len:], {
            "mode": "fuzzy",
            "trimmed_chars": best_len,
            "fuzzy_ratio": round(best_ratio, 4),
            "fuzzy_len": best_len,
        }

    return continuation_text, {"mode": "none", "trimmed_chars": 0}


def apply_cross_think_match_cover(
    prefix_text: str,
    continuation_text: str,
    *,
    min_exact_overlap: int = 24,
    fuzzy_min_len: int = 16,
    fuzzy_max_len: int = 200,
    fuzzy_ratio: float = 0.9,
) -> Tuple[str, Dict[str, object]]:
    """
    Match/cover between:
    - tail of body after the first </think> in prefix_text
    - head of body after the next </think> in continuation_text
    """
    if not prefix_text or not continuation_text:
        return continuation_text, {"mode": "none", "trimmed_chars": 0, "reason": "empty_text"}

    first_close = prefix_text.find("</think>")
    if first_close < 0:
        return continuation_text, {"mode": "none", "trimmed_chars": 0, "reason": "no_first_think_close_in_prefix"}
    prefix_body = prefix_text[first_close + len("</think>") :]
    if not prefix_body:
        return continuation_text, {"mode": "none", "trimmed_chars": 0, "reason": "empty_prefix_body"}

    second_close = continuation_text.find("</think>")
    if second_close < 0:
        return continuation_text, {"mode": "none", "trimmed_chars": 0, "reason": "no_second_think_close_in_continuation"}
    head = continuation_text[: second_close + len("</think>")]
    post_body = continuation_text[second_close + len("</think>") :]
    if not post_body:
        return continuation_text, {"mode": "none", "trimmed_chars": 0, "reason": "empty_post_think_body"}

    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    exact_k = longest_suffix_prefix_overlap(prefix_body, post_body, max_k=600)
    if exact_k >= min_exact_overlap:
        return head + post_body[exact_k:], {
            "mode": "exact",
            "trimmed_chars": exact_k,
            "exact_overlap": exact_k,
        }

    max_len = min(fuzzy_max_len, len(prefix_body), len(post_body))
    best_len = 0
    best_ratio = 0.0
    for l in range(max_len, fuzzy_min_len - 1, -1):
        suffix = prefix_body[-l:]
        prefix = post_body[:l]
        ratio = SequenceMatcher(None, suffix, prefix).ratio()
        if ratio >= fuzzy_ratio:
            best_len = l
            best_ratio = ratio
            break

    if best_len > 0:
        return head + post_body[best_len:], {
            "mode": "fuzzy",
            "trimmed_chars": best_len,
            "fuzzy_ratio": round(best_ratio, 4),
            "fuzzy_len": best_len,
        }

    # Fallback: anchor-start duplicate match (not limited to suffix-prefix).
    m = re.search(r"\S", post_body)
    if m:
        post_core = post_body[m.start() :]
        line_end = post_core.find("\n")
        first_line = (post_core if line_end < 0 else post_core[:line_end]).strip()
        if len(first_line) >= 10:
            idx = prefix_body.rfind(first_line)
            if idx >= 0:
                k = _common_prefix_len(prefix_body[idx:], post_core)
                if k >= min_exact_overlap:
                    return head + post_body[: m.start()] + post_core[k:], {
                        "mode": "anchor_exact",
                        "trimmed_chars": k,
                        "anchor": first_line,
                    }

    return continuation_text, {"mode": "none", "trimmed_chars": 0}


def think_balance_ok(text: str) -> bool:
    opens = len(re.findall(r"<think>", text))
    closes = len(re.findall(r"</think>", text))
    return opens == closes


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    out = text
    # Remove closed think blocks first.
    out = re.sub(r"(?is)<think>.*?</think>", "", out)
    # Remove trailing unclosed think blocks.
    out = re.sub(r"(?is)<think>.*$", "", out)
    # Remove dangling close tags if any.
    out = out.replace("</think>", "")
    return out


def corrupt_prefix_text(text: str) -> Tuple[str, Dict[str, object]]:
    m = re.search(r"(?<![A-Za-z0-9_.-])(-?\d+)(?![A-Za-z0-9_.-])", text)
    if not m:
        fallback = text + "\n[Injected error marker: claim above may be wrong.]"
        return fallback, {"mode": "append_marker", "changed": False}

    src = int(m.group(1))
    dst = src + 1 if src >= 0 else src - 1
    edited = text[: m.start()] + str(dst) + text[m.end() :]
    return edited, {"mode": "number_shift", "changed": True, "from": src, "to": dst}


def corrupt_numbers_near_anchor(
    text: str,
    *,
    anchor_regex: str,
    max_changes: int = 2,
    window_chars: int = 240,
) -> Tuple[str, Dict[str, object]]:
    """
    在锚点（如 Step 3）附近改若干数字，模拟“中间步骤出错”。
    """
    if not text:
        return text, {"mode": "anchor_number_shift", "changed": False, "reason": "empty_text"}

    m_anchor = re.search(anchor_regex, text, re.IGNORECASE | re.DOTALL)
    if not m_anchor:
        fallback, meta = corrupt_prefix_text(text)
        meta["mode"] = "anchor_fallback_number_shift"
        meta["anchor_found"] = False
        return fallback, meta

    st = max(0, m_anchor.start() - window_chars // 4)
    ed = min(len(text), m_anchor.end() + window_chars)
    seg = text[st:ed]

    matches = list(re.finditer(r"(?<![A-Za-z0-9_.-])(-?\d+)(?![A-Za-z0-9_.-])", seg))
    if not matches:
        return text, {
            "mode": "anchor_number_shift",
            "changed": False,
            "anchor_found": True,
            "reason": "no_number_in_window",
        }

    take = min(max_changes, len(matches))
    offset = 0
    changed = []
    seg_edit = seg
    for i in range(take):
        m = matches[i]
        a = m.start() + offset
        b = m.end() + offset
        src = int(seg_edit[a:b])
        dst = src + 1 if src >= 0 else src - 1
        seg_edit = seg_edit[:a] + str(dst) + seg_edit[b:]
        offset += len(str(dst)) - (b - a)
        changed.append({"from": src, "to": dst})

    edited = text[:st] + seg_edit + text[ed:]
    return edited, {
        "mode": "anchor_number_shift",
        "changed": True,
        "anchor_found": True,
        "anchor_regex": anchor_regex,
        "changes": changed,
        "window": {"start": st, "end": ed},
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_tasks_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
