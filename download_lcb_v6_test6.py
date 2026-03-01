#!/usr/bin/env python3
"""
Download LiveCodeBench code_generation_lite release_v6 JSONL file (test6.jsonl).
"""

from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("Please install huggingface_hub first: pip install -U huggingface_hub") from e

    out_dir = Path("data/livecodebench_code_generation_lite")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        repo_type="dataset",
        filename="test6.jsonl",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()
