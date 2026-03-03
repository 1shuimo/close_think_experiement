#!/usr/bin/env python3
"""
Prepare LiveCodeBench code-generation tasks as jsonl for close runner.
"""

import argparse
import json
import re
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
    p.add_argument(
        "--source-mode",
        default="auto",
        choices=["auto", "jsonl", "parquet"],
        help="Dataset loading strategy. auto: jsonl -> parquet.",
    )
    p.add_argument(
        "--hf-jsonl-file",
        default=None,
        help="Optional explicit JSONL file path in HF dataset repo, e.g. test6.jsonl.",
    )
    p.add_argument(
        "--local-jsonl-file",
        default=None,
        help="Optional local JSONL file path (highest priority).",
    )
    return p.parse_args()


OFFICIAL_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
OFFICIAL_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _build_prompt(problem: Dict[str, Any]) -> str:
    """
    Match LiveCodeBench official `format_prompt_generation` user-message structure
    as closely as possible for local close runner tasks.
    """
    content = str(problem.get("question_content", "")).strip()
    starter = str(problem.get("starter_code", "")).rstrip()
    parts: List[str] = []
    parts.append("### Question:")
    parts.append(content)
    parts.append("")
    if starter:
        parts.append(f"### Format: {OFFICIAL_FORMAT_WITH_STARTER}")
        parts.append("```python")
        parts.append(starter)
        parts.append("```")
    else:
        parts.append(f"### Format: {OFFICIAL_FORMAT_WITHOUT_STARTER}")
        parts.append("```python")
        parts.append("# YOUR CODE HERE")
        parts.append("```")
    parts.append("")
    parts.append("### Answer: (use the provided format with backticks)")
    parts.append("")
    return "\n".join(parts).strip()


def _list_release_parquet_files(dataset_name: str, release_version: str, split: str) -> List[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    split_named = [f for f in parquet_files if Path(f).name.startswith(f"{split}-")]
    if not split_named:
        split_named = parquet_files

    release_prefixes = [
        f"{release_version}/",
        f"data/{release_version}/",
    ]
    release_scoped = [
        f
        for f in split_named
        if any(f.startswith(pref) for pref in release_prefixes) or f"/{release_version}/" in f
    ]
    picked = sorted(release_scoped if release_scoped else split_named)
    return picked


def _pick_release_jsonl_file(files: List[str], release_version: str, split: str) -> Optional[str]:
    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    if not jsonl_files:
        return None

    split = split.lower()
    split_named = [f for f in jsonl_files if Path(f).name.lower().startswith(split)]
    if not split_named:
        split_named = jsonl_files

    # release_v6 -> 6
    m = re.search(r"(\d+)$", release_version or "")
    rel_num = m.group(1) if m else None

    if rel_num:
        exact_candidates = [
            f"{split}{rel_num}.jsonl",
            f"{split}_{rel_num}.jsonl",
            f"{split}-{rel_num}.jsonl",
            f"{split}v{rel_num}.jsonl",
            f"{split}_v{rel_num}.jsonl",
            f"test{rel_num}.jsonl",
        ]
        lowered = {Path(f).name.lower(): f for f in split_named}
        for cand in exact_candidates:
            hit = lowered.get(cand.lower())
            if hit:
                return hit

        # fallback: filename stem contains the release number
        weak_hits = [
            f for f in split_named if re.search(rf"(?<!\d){rel_num}(?!\d)", Path(f).stem)
        ]
        if weak_hits:
            return sorted(weak_hits)[0]

    return sorted(split_named)[0]


def _load_rows_from_jsonl_fallback(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    if args.local_jsonl_file:
        local_jsonl = Path(args.local_jsonl_file).expanduser().resolve()
        if not local_jsonl.exists():
            raise FileNotFoundError(f"local JSONL file not found: {local_jsonl}")
        rows: List[Dict[str, Any]] = []
        with local_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if args.hf_jsonl_file:
        jsonl_file = args.hf_jsonl_file
    else:
        files = list_repo_files(repo_id=args.dataset_name, repo_type="dataset")
        jsonl_file = _pick_release_jsonl_file(files, args.release_version, args.split)
        if not jsonl_file:
            raise RuntimeError(
                f"No JSONL file found for dataset={args.dataset_name}, release={args.release_version}, split={args.split}"
            )

    local_path = hf_hub_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        filename=jsonl_file,
    )

    rows: List[Dict[str, Any]] = []
    with Path(local_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_rows_from_parquet_fallback(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    parquet_files = _list_release_parquet_files(args.dataset_name, args.release_version, args.split)
    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found for dataset={args.dataset_name}, release={args.release_version}, split={args.split}"
        )

    hf_urls = [f"hf://datasets/{args.dataset_name}/{p}" for p in parquet_files]
    try:
        ds = load_dataset("parquet", data_files=hf_urls, split="train")
        return [dict(x) for x in ds]
    except Exception:
        local_paths = [
            hf_hub_download(repo_id=args.dataset_name, repo_type="dataset", filename=p)
            for p in parquet_files
        ]
        ds = load_dataset("parquet", data_files=local_paths, split="train")
        return [dict(x) for x in ds]


def _load_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], str]:
    mode = (args.source_mode or "auto").lower()
    tried: List[str] = []

    if mode == "jsonl":
        name = "local_jsonl" if args.local_jsonl_file else "jsonl_fallback"
        return _load_rows_from_jsonl_fallback(args), name
    if mode == "parquet":
        return _load_rows_from_parquet_fallback(args), "parquet_fallback"

    # auto: prefer script-free loaders only.
    auto_loaders = []
    if args.local_jsonl_file:
        auto_loaders.append(("local_jsonl", _load_rows_from_jsonl_fallback))
    auto_loaders.extend(
        [
            ("jsonl_fallback", _load_rows_from_jsonl_fallback),
            ("parquet_fallback", _load_rows_from_parquet_fallback),
        ]
    )
    for name, fn in auto_loaders:
        try:
            return fn(args), name
        except Exception as e:
            tried.append(f"{name}: {e}")
            continue

    raise RuntimeError("All dataset loading strategies failed:\n" + "\n".join(tried))


def main() -> None:
    args = parse_args()

    if args.source_mode in ("auto", "parquet"):
        try:
            import datasets  # noqa: F401
        except Exception as e:
            if args.source_mode == "parquet":
                raise RuntimeError("datasets package is required in source-mode=parquet.") from e
            # auto mode can still succeed via JSONL path; keep going.

    rows, loader_mode = _load_rows(args)

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
                "loader_mode": loader_mode,
                "n_tasks": len(tasks),
                "question_ids": [t["id"] for t in tasks],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
