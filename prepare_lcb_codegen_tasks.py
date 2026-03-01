#!/usr/bin/env python3
"""
Prepare LiveCodeBench code-generation tasks as jsonl for close runner.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LiveCodeBench codegen tasks to close jsonl format.")
    p.add_argument("--release-version", default="release_v6")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--question-id", default=None, help="Export only this question_id.")
    p.add_argument("--limit", type=int, default=1, help="Max number of tasks to export after filtering.")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument(
        "--output-problems-json",
        default=None,
        help="Save selected original problems for later evaluation (default: <output-jsonl>.problems.json).",
    )
    p.add_argument(
        "--dataset-name",
        default="livecodebench/code_generation_lite",
        help="HF dataset repo name.",
    )
    p.add_argument("--split", default="test")
    return p.parse_args()


def _build_prompt(problem: Dict[str, Any]) -> str:
    title = str(problem.get("question_title", "")).strip()
    content = str(problem.get("question_content", "")).strip()
    starter = str(problem.get("starter_code", "")).rstrip()
    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    parts.append("Problem:")
    parts.append(content)
    if starter:
        parts.append("")
        parts.append("Starter Code:")
        parts.append("```python")
        parts.append(starter)
        parts.append("```")
    parts.append("")
    parts.append("Return only the final Python solution code in one markdown code block.")
    return "\n".join(parts).strip()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "datasets package is required. Run this script in LiveCodeBench environment."
        ) from e

    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        version_tag=args.release_version,
        trust_remote_code=True,
    )
    rows: List[Dict[str, Any]] = [dict(x) for x in ds]

    if args.start_date:
        rows = [x for x in rows if str(x.get("contest_date", "")) >= args.start_date]
    if args.end_date:
        rows = [x for x in rows if str(x.get("contest_date", "")) <= args.end_date]
    if args.question_id:
        rows = [x for x in rows if str(x.get("question_id", "")) == args.question_id]

    rows = sorted(rows, key=lambda x: str(x.get("question_id", "")))
    if args.limit > 0:
        rows = rows[: args.limit]

    out_jsonl = Path(args.output_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_problem_json = (
        Path(args.output_problems_json).resolve()
        if args.output_problems_json
        else out_jsonl.with_suffix(out_jsonl.suffix + ".problems.json")
    )

    tasks: List[Dict[str, Any]] = []
    for p in rows:
        qid = str(p.get("question_id", "")).strip()
        task = {
            "id": qid,
            "question_id": qid,
            "user_prompt": _build_prompt(p),
            "expected_regex": None,
            "reference_output": None,
            "lcb_meta": {
                "question_title": p.get("question_title"),
                "platform": p.get("platform"),
                "contest_date": p.get("contest_date"),
                "difficulty": p.get("difficulty"),
            },
            # enforce your desired insertion behavior
            "locator_only": True,
            "corrupt_after_first_think": True,
            "corrupt_min_step": 2,
        }
        tasks.append(task)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    out_problem_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "tasks_jsonl": str(out_jsonl),
                "problems_json": str(out_problem_json),
                "n_tasks": len(tasks),
                "question_ids": [t["id"] for t in tasks],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

