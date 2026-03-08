#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Extract one task from a jsonl file into a new single-task jsonl."
    )
    p.add_argument(
        "--tasks-file",
        default=str(here / "data" / "tasks_aime2025.jsonl"),
        help="Source jsonl file.",
    )
    p.add_argument(
        "--task-id",
        default="aime2025_ii_15",
        help="Task id to extract. Default is aime2025_ii_15.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. Default is ./tmp/<task-id>.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        raise SystemExit(f"Tasks file not found: {tasks_path}")

    output_path = Path(args.output) if args.output else Path("tmp") / f"{args.task_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    found = None
    for raw_line in tasks_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("id", "")) == args.task_id:
            found = row
            break

    if found is None:
        raise SystemExit(f"Task id not found: {args.task_id}")

    output_path.write_text(json.dumps(found, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
