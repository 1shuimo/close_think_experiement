# Outputs Layout

Historical experiment outputs are grouped here to keep the repo root readable.

Current runners also default to writing future results under `outputs/`.
Per-model result files now use the model directory basename such as `Qwen3-32B`,
instead of embedding the full `/scratch-ssd/...` path in filenames.

- `outputs/aime`
  - AIME-focused runs and AIME-specific reports
- `outputs/math`
  - math-only experiment suites
- `outputs/longproc`
  - legacy LongProc outputs kept for reference
- `outputs/lcb`
  - LiveCodeBench-related suites
- `outputs/smoke`
  - smoke / debugging runs
