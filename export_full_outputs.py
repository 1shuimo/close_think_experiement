import argparse
import json
import re
from pathlib import Path


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "task")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export branch full outputs from results.jsonl")
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--print-to-stdout", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.results_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [x.strip() for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, line in enumerate(lines, start=1):
        rec = json.loads(line)
        task_id = safe_name(str(rec.get("task_id", f"task_{i}")))
        d = out_dir / task_id
        d.mkdir(parents=True, exist_ok=True)

        a = rec["branch_A"]["full_text"]
        b = rec["branch_B"]["full_text"]
        (d / "branch_A.full.txt").write_text(a, encoding="utf-8")
        (d / "branch_B.full.txt").write_text(b, encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.print_to_stdout:
            print("\n========================================")
            print(f"task={rec.get('task_id')}")
            print("\n[Branch A Full Text]\n")
            print(a)
            print("\n[Branch B Full Text]\n")
            print(b)
            print("========================================\n")


if __name__ == "__main__":
    main()

