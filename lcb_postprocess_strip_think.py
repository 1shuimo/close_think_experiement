#!/usr/bin/env python3
"""
Post-process LiveCodeBench eval_all JSON:
- remove all <think>...</think> blocks from model outputs before scoring.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strip <think> blocks from eval_all JSON outputs.")
    p.add_argument("--input", required=True, help="Input eval_all JSON path.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument(
        "--key-regex",
        default=r"(?i)(output|completion|prediction|pred|response|answer|code)",
        help="Only strip string values under keys matching this regex.",
    )
    p.add_argument(
        "--strip-all-strings",
        action="store_true",
        help="Strip every string field recursively (ignore key filtering).",
    )
    return p.parse_args()


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    s = text
    while True:
        m_open = re.search(r"<think>", s, re.IGNORECASE)
        if not m_open:
            break
        m_close_rel = re.search(r"</think>", s[m_open.end() :], re.IGNORECASE)
        if not m_close_rel:
            s = s[: m_open.start()]
            break
        close_end = m_open.end() + m_close_rel.end()
        s = s[: m_open.start()] + s[close_end:]
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    return s


def transform(obj: Any, *, key_regex: re.Pattern[str], strip_all_strings: bool, stats: Dict[str, int], key: str = "") -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = transform(
                v,
                key_regex=key_regex,
                strip_all_strings=strip_all_strings,
                stats=stats,
                key=str(k),
            )
        return out
    if isinstance(obj, list):
        return [
            transform(
                x,
                key_regex=key_regex,
                strip_all_strings=strip_all_strings,
                stats=stats,
                key=key,
            )
            for x in obj
        ]
    if isinstance(obj, str):
        should_strip = strip_all_strings or bool(key_regex.search(key or ""))
        if not should_strip:
            return obj
        stripped = strip_think_blocks(obj)
        if stripped != obj:
            stats["fields_changed"] += 1
        stats["fields_seen"] += 1
        return stripped
    return obj


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    stats = {"fields_seen": 0, "fields_changed": 0}
    key_regex = re.compile(args.key_regex)
    cleaned = transform(
        data,
        key_regex=key_regex,
        strip_all_strings=bool(args.strip_all_strings),
        stats=stats,
    )
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "fields_seen": stats["fields_seen"],
                "fields_changed": stats["fields_changed"],
                "strip_all_strings": bool(args.strip_all_strings),
                "key_regex": args.key_regex,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

