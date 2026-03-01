# close_think_experiement

这个仓库当前聚焦两个数据集：
- AIME（含 AIME2025 与 4 题 hard 集）
- LiveCodeBench

核心实验目标：验证“同一条生成轨迹中，中间插入 `<think>` 后”的稳定性与效果。

## 当前脚本

- `test_close_suite.py`
  - 仅插入版（A/B 分叉：A=直接续写，B=中插 `<think>` 后续写）
  - 不含改错逻辑
  - 不含 LongProc 逻辑
- `test_close_suite_corrupt.py`
  - 改错版（在前缀中做扰动 + 中插 `<think>` + A/B）
  - 不含 LongProc 逻辑
- `run_aime.py`
  - AIME 一键入口（调用 `test_close_suite.py`）
- `run_aime_corrupt.py`
  - AIME 改错一键入口（调用 `test_close_suite_corrupt.py`）
- `run_live_code.py`
  - LiveCodeBench 官方 runner 对接脚本

## 任务文件

- `tasks_aime2025.jsonl`：AIME2025（30题）
- `tasks_math_hard_steps.jsonl`：4题 hard 集（Aya/Hyperbola/Complex/Token game）

## 快速开始

详细命令见：
- [`RUN_COMMANDS.md`](./RUN_COMMANDS.md)

你可以直接按以下最小步骤执行：

```bash
cd /path/to/close_think_experiement
MODEL=/scratch-ssd/guoeng/huggingface/models/Qwen3-32B
```

四题改错测试：

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_math_hard_steps.jsonl \
  --output-dir suite_math_hard_4q_corrupt \
  --corrupt-mode anchor_number_shift \
  --corrupt-after-first-think \
  --corrupt-prefer-sign-flip \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --apply-match-cover
```

LiveCodeBench 对接：

```bash
python run_live_code.py \
  --lcb-root /path/to/LiveCodeBench \
  --model Qwen3-32B \
  --local-model-path "$MODEL" \
  --scenario codegeneration \
  --release-version release_v6 \
  --evaluate \
  --n 10 \
  --temperature 0.2
```

## 输出结果

每次运行会生成：
- `*.results.jsonl`：逐题 A/B 详细记录
- `*.summary.json`：单模型汇总
- `summary_all_models.json`：本次所有模型总汇总
- （若开 `--save-task-texts`）每题 `branch_A.full.txt` / `branch_B.full.txt` / `meta.json`

## 说明

- 目前主流程已移除 LongProc 相关分支和参数。
- 如果需要保留历史 LongProc 结果目录，可继续保留在仓库中，但不再是当前脚本的运行路径。
