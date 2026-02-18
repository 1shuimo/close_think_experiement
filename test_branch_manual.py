import argparse
import json
from pathlib import Path

import torch

from think_branch_common import (
    build_prompt_ids,
    compose_system_prompt,
    generate_from_state,
    generate_until_checkpoint,
    load_model,
    prefill_kv,
    read_text,
    think_balance_ok,
    write_text,
)


DEFAULT_SYSTEM_PROMPT = """
You are a careful and rigorous problem solver with a unique internal reasoning capability.

Core instruction:
When uncertainty appears, you may use <think>...</think>.
After </think>, continue exactly where you paused.

Strict constraints:
1) Do not repeat text right before <think>.
2) If interrupted in the middle of a sentence, continue from that point.
3) Keep reasoning concise and local.
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual branch test after prefix editing.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--prompt-mode", default="enhanced", choices=["baseline", "enhanced"])
    p.add_argument("--think-word-limit", type=int, default=60)
    p.add_argument("--user-prompt", default="Solve for x: (x-1)(x+1)=35.")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--checkpoint-delay", type=int, default=200)
    p.add_argument("--max-prefix-tokens", type=int, default=5000)
    p.add_argument("--max-new-after", type=int, default=1200)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--inject-text", default="<think>")
    p.add_argument("--output-dir", default="close/manual_outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.model_path, dtype_name=args.dtype)
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

    prompt_ids = build_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=compose_system_prompt(
            args.system_prompt,
            prompt_mode=args.prompt_mode,
            think_word_limit=args.think_word_limit,
        ),
        user_prompt=args.user_prompt,
        device=device,
        enable_thinking=True,
    )

    print("\n===== Stage 1: generate prefix until checkpoint =====\n")
    stage1 = generate_until_checkpoint(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        delay_tokens_after_first_think_end=args.checkpoint_delay,
        max_new_tokens=args.max_prefix_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        chunk_size=args.chunk_size,
        print_stream=True,
    )
    print("\n\n===== Stage 1 done =====\n")

    prefix_ids = stage1["generated_ids"]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

    orig_path = out_dir / "generated_prefix.ORIG.txt"
    edit_path = out_dir / "generated_prefix.EDIT_ME.txt"
    meta_path = out_dir / "stage1_meta.json"
    write_text(orig_path, prefix_text)
    write_text(edit_path, prefix_text)
    meta_path.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "dtype": args.dtype,
                "prompt_mode": args.prompt_mode,
                "think_word_limit": args.think_word_limit,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "checkpoint_delay": args.checkpoint_delay,
                "max_prefix_tokens": args.max_prefix_tokens,
                "seen_first_think_end": bool(stage1["seen_first_think_end"]),
                "counter_after_think_end": int(stage1["counter_after_think_end"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved:\n- {orig_path}\n- {edit_path}\n- {meta_path}")
    input("\nEdit generated_prefix.EDIT_ME.txt, then press ENTER to run Branch A/B...")

    edited_text = read_text(edit_path)
    edited_ids = tokenizer.encode(edited_text, add_special_tokens=False)
    edited_ids_t = torch.tensor([edited_ids], dtype=torch.long, device=device)

    print("\n===== Branch A: direct continuation =====\n")
    full_ids_a = torch.cat([prompt_ids, edited_ids_t], dim=1)
    past_a, logits_a = prefill_kv(model, full_ids_a, chunk_size=args.chunk_size)
    gen_a_ids = generate_from_state(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_a,
        logits=logits_a,
        max_new_tokens=args.max_new_after,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + 1,
        print_stream=True,
    )
    print("\n")

    print("\n===== Branch B: inject then continuation =====\n")
    full_ids_b = torch.cat([prompt_ids, edited_ids_t], dim=1)
    past_b, logits_b = prefill_kv(model, full_ids_b, chunk_size=args.chunk_size)
    print(args.inject_text, end="", flush=True)

    inject_ids = tokenizer.encode(args.inject_text, add_special_tokens=False)
    inject_ids_t = torch.tensor([inject_ids], dtype=torch.long, device=device)
    out_inj = model(inject_ids_t, past_key_values=past_b, use_cache=True)
    past_b = out_inj.past_key_values
    logits_b = out_inj.logits[:, -1, :]

    gen_b_ids = generate_from_state(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_b,
        logits=logits_b,
        max_new_tokens=args.max_new_after,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + 2,
        print_stream=True,
    )
    print("\n")

    cont_a = tokenizer.decode(gen_a_ids, skip_special_tokens=False)
    cont_b = tokenizer.decode(gen_b_ids, skip_special_tokens=False)
    full_a = edited_text + cont_a
    full_b = edited_text + args.inject_text + cont_b

    out_a = out_dir / "branch_A.txt"
    out_b = out_dir / "branch_B.txt"
    sum_path = out_dir / "branch_summary.json"
    write_text(out_a, full_a)
    write_text(out_b, full_b)
    sum_path.write_text(
        json.dumps(
            {
                "branch_A_think_balanced": think_balance_ok(full_a),
                "branch_B_think_balanced": think_balance_ok(full_b),
                "branch_A_new_tokens": len(gen_a_ids),
                "branch_B_new_tokens": len(gen_b_ids),
                "inject_text": args.inject_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved:\n- {out_a}\n- {out_b}\n- {sum_path}")


if __name__ == "__main__":
    main()
