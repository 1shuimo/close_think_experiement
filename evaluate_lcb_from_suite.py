#!/usr/bin/env python3
"""
Bridge close suite outputs to LiveCodeBench code-generation evaluation.

Workflow:
1) Read `*.results.jsonl` (or `*.branch_b_view.jsonl`) from close runner.
2) Take branch full text (A or B), strip all <think>...</think> blocks.
3) Extract code from markdown block (fallback to raw text).
4) Evaluate with LiveCodeBench codegen metrics (subset by question_id is supported).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LiveCodeBench from close suite outputs.")
    p.add_argument("--suite-results-jsonl", required=True, help="Path to close results jsonl.")
    p.add_argument("--branch", default="b", choices=["a", "b"], help="Use branch A or B full text.")
    p.add_argument(
        "--question-ids",
        default=None,
        help="Comma-separated question_id filter (default: evaluate all tasks found in suite jsonl).",
    )
    p.add_argument("--limit", type=int, default=0, help="Limit number of tasks after filtering (0 means no limit).")

    p.add_argument(
        "--strip-think",
        action="store_true",
        help="Strip all <think>...</think> blocks before code extraction.",
    )
    p.add_argument(
        "--extract-mode",
        default="markdown_or_raw",
        choices=["markdown_or_raw", "markdown_only"],
        help="How to extract code from cleaned text.",
    )

    p.add_argument(
        "--problems-json",
        default=None,
        help="Problems JSON exported by prepare_lcb_codegen_tasks.py (recommended).",
    )
    p.add_argument(
        "--release-version",
        default="release_v6",
        help="Used only when problems are loaded from LiveCodeBench dataset directly.",
    )

    p.add_argument("--k-list", default="1", help="Comma-separated pass@k list, e.g. '1,5'.")
    p.add_argument("--num-process-evaluate", type=int, default=12)
    p.add_argument("--timeout", type=int, default=6)
    p.add_argument("--debug-eval", action="store_true")

    p.add_argument("--output-dir", required=True, help="Directory for extracted/eval outputs.")
    p.add_argument("--output-extracted-jsonl", default=None)
    p.add_argument("--output-eval-all", default=None)
    p.add_argument("--output-metrics-json", default=None)

    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only extract cleaned text/code; skip LCB grading.",
    )
    p.add_argument(
        "--compute-scores",
        action="store_true",
        help="After eval_all is produced, call lcb_runner.evaluation.compute_scores.",
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--python-bin", default=sys.executable)
    return p.parse_args()


def _strip_all_think_blocks(text: str) -> Tuple[str, Dict[str, Any]]:
    if not text:
        return text, {"removed_blocks": 0, "unclosed_tail_trimmed": False, "stray_tags_removed": 0}

    s = text
    removed = 0
    unclosed_tail_trimmed = False
    while True:
        m_open = re.search(r"<think>", s, re.IGNORECASE)
        if not m_open:
            break
        m_close_rel = re.search(r"</think>", s[m_open.end() :], re.IGNORECASE)
        if not m_close_rel:
            s = s[: m_open.start()]
            removed += 1
            unclosed_tail_trimmed = True
            break
        close_end = m_open.end() + m_close_rel.end()
        s = s[: m_open.start()] + s[close_end:]
        removed += 1

    s2 = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    return s2, {
        "removed_blocks": int(removed),
        "unclosed_tail_trimmed": bool(unclosed_tail_trimmed),
        "stray_tags_removed": int(s2 != s),
    }


def _extract_code(cleaned_text: str, extract_mode: str) -> Tuple[str, str]:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n(.*?)```", cleaned_text or "", flags=re.DOTALL)
    if blocks:
        for b in reversed(blocks):
            code = b.strip()
            if code:
                return code, "markdown_last_block"
        return "", "markdown_empty_block"
    if extract_mode == "markdown_only":
        return "", "no_markdown_block"
    return (cleaned_text or "").strip(), "raw_fallback"


def _pick_branch_full_text(rec: Dict[str, Any], branch: str) -> Optional[str]:
    key = "branch_B" if branch == "b" else "branch_A"
    obj = rec.get(key)
    if isinstance(obj, dict):
        v = obj.get("full_text")
        if isinstance(v, str):
            return v

    # Backward compatibility for flattened/debug exports.
    flat_key = f"{key}_full_text"
    v2 = rec.get(flat_key)
    if isinstance(v2, str):
        return v2
    return None


def _parse_k_list(s: str) -> List[int]:
    vals = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals or [1]


def _load_suite_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    suite_path = Path(args.suite_results_jsonl).resolve()
    lines = suite_path.read_text(encoding="utf-8").splitlines()
    qid_filter = None
    if args.question_ids:
        qid_filter = {x.strip() for x in args.question_ids.split(",") if x.strip()}

    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        task_id = str(rec.get("task_id", "")).strip()
        if not task_id:
            continue
        if qid_filter is not None and task_id not in qid_filter:
            continue

        full_text = _pick_branch_full_text(rec, args.branch)
        if full_text is None:
            continue

        cleaned = full_text
        strip_meta = {"removed_blocks": 0, "unclosed_tail_trimmed": False, "stray_tags_removed": 0}
        if args.strip_think:
            cleaned, strip_meta = _strip_all_think_blocks(full_text)
        code, code_source = _extract_code(cleaned, args.extract_mode)

        out.append(
            {
                "task_id": task_id,
                "model_path": rec.get("model_path"),
                "stop_reason": (rec.get("branch_B", {}) or {}).get("stop_reason")
                if args.branch == "b"
                else (rec.get("branch_A", {}) or {}).get("stop_reason"),
                "full_text": full_text,
                "cleaned_text": cleaned,
                "strip_meta": strip_meta,
                "code": code,
                "code_source": code_source,
                "source_line_index": idx,
            }
        )

    if args.limit > 0:
        out = out[: args.limit]
    return out


def _load_problems(
    *,
    question_ids: List[str],
    problems_json: Optional[str],
    release_version: str,
) -> List[Any]:
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
    except Exception as e:
        raise RuntimeError(
            "lcb_runner not available. Activate LiveCodeBench env first (pip install -e LiveCodeBench)."
        ) from e

    qid_set = set(question_ids)

    if problems_json:
        p = Path(problems_json).resolve()
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if isinstance(raw.get("problems"), list):
                rows = raw["problems"]
            elif isinstance(raw.get("rows"), list):
                rows = raw["rows"]
            else:
                raise ValueError(f"Unsupported problems JSON structure: {p}")
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"Unsupported problems JSON type: {type(raw)}")

        by_qid: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            qid = str(row.get("question_id", "")).strip()
            if qid and qid in qid_set:
                by_qid[qid] = row
        missing = [qid for qid in question_ids if qid not in by_qid]
        if missing:
            raise ValueError(f"Missing question_ids in problems JSON: {missing[:5]}")
        return [CodeGenerationProblem(**by_qid[qid]) for qid in question_ids]

    all_problems = load_code_generation_dataset(release_version)
    by_qid = {str(p.question_id): p for p in all_problems}
    missing = [qid for qid in question_ids if qid not in by_qid]
    if missing:
        raise ValueError(f"Missing question_ids in release {release_version}: {missing[:5]}")
    return [by_qid[qid] for qid in question_ids]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_jsonl = (
        Path(args.output_extracted_jsonl).resolve()
        if args.output_extracted_jsonl
        else out_dir / f"extracted_branch_{args.branch}.jsonl"
    )
    eval_all_path = (
        Path(args.output_eval_all).resolve() if args.output_eval_all else out_dir / "eval_all.json"
    )
    metrics_path = (
        Path(args.output_metrics_json).resolve()
        if args.output_metrics_json
        else out_dir / "metrics.json"
    )

    rows = _load_suite_rows(args)
    if not rows:
        raise ValueError("No usable tasks found in suite results JSONL.")

    with extracted_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepare_only": True,
                    "n_tasks": len(rows),
                    "extracted_jsonl": str(extracted_jsonl),
                    "strip_think": bool(args.strip_think),
                    "extract_mode": args.extract_mode,
                },
                ensure_ascii=False,
            )
        )
        return

    try:
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
        from lcb_runner.evaluation.pass_k_utils import extract_instance_results
    except Exception as e:
        raise RuntimeError(
            "lcb_runner evaluation modules not available. Use --prepare-only or activate LiveCodeBench env."
        ) from e

    question_ids = [r["task_id"] for r in rows]
    problems = _load_problems(
        question_ids=question_ids,
        problems_json=args.problems_json,
        release_version=args.release_version,
    )

    samples_list = [p.get_evaluation_sample() for p in problems]
    generations_list = [[r["code"]] for r in rows]
    k_list = _parse_k_list(args.k_list)

    metrics, results_by_task_id, metadatas = codegen_metrics(
        samples_list=samples_list,
        generations_list=generations_list,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug_eval,
        k_list=k_list,
    )
    graded_list = extract_instance_results(results_by_task_id)

    if len(graded_list) != len(rows):
        raise RuntimeError(
            f"graded_list length mismatch: got {len(graded_list)}, expected {len(rows)}"
        )

    eval_all: List[Dict[str, Any]] = []
    for i, problem in enumerate(problems):
        row = rows[i]
        metadata = metadatas[i] if i < len(metadatas) else None
        eval_all.append(
            problem.insert_output_evaluation(
                output_list=[row["cleaned_text"]],
                code_list=[row["code"]],
                graded_list=graded_list[i],
                metadata=metadata,
                source_task_id=row["task_id"],
                source_line_index=row["source_line_index"],
                strip_think=bool(args.strip_think),
                code_source=row["code_source"],
            )
        )

    eval_all_path.write_text(json.dumps(eval_all, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "k_list": k_list,
                "n_tasks": len(rows),
                "suite_results_jsonl": str(Path(args.suite_results_jsonl).resolve()),
                "extracted_jsonl": str(extracted_jsonl),
                "eval_all_file": str(eval_all_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.compute_scores:
        cmd = [
            args.python_bin,
            "-m",
            "lcb_runner.evaluation.compute_scores",
            "--eval_all_file",
            str(eval_all_path),
        ]
        if args.start_date:
            cmd.extend(["--start_date", args.start_date])
        if args.end_date:
            cmd.extend(["--end_date", args.end_date])
        subprocess.run(cmd, check=True)

    print(
        json.dumps(
            {
                "prepare_only": False,
                "n_tasks": len(rows),
                "extracted_jsonl": str(extracted_jsonl),
                "eval_all_file": str(eval_all_path),
                "metrics_json": str(metrics_path),
                "pass_at": {k: metrics.get(f"pass@{k}") for k in k_list},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
