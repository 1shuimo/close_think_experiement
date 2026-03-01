# Run Commands (AIME + LiveCodeBench)

本文件给出服务器可直接复制的命令。

已知路径：
- 项目目录：`close_think_experiement`
- 模型目录：`/scratch-ssd/guoeng/huggingface/models/Qwen3-32B`

```bash
cd /path/to/close_think_experiement
MODEL=/scratch-ssd/guoeng/huggingface/models/Qwen3-32B
```

## 1) AIME 四题改错测试（你之前那 4 题）

任务文件：`tasks_math_hard_steps.jsonl`

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_math_hard_steps.jsonl \
  --output-dir suite_math_hard_4q_corrupt \
  --checkpoint-mode think_end_then_regex \
  --checkpoint-regex '(?i)step\s*3' \
  --enable-first-think-max-words \
  --first-think-max-words 120 \
  --step-wait-extra-tokens 2000 \
  --no-step-fallback-offset-tokens 300 \
  --corrupt-mode anchor_number_shift \
  --corrupt-after-first-think \
  --corrupt-prefer-sign-flip \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --apply-match-cover \
  --save-task-texts \
  --print-full-output
```

可选：如果要重新打开软提示约束，再加

```bash
--enable-think-word-limit --think-word-limit 60
```

可选：如果要关闭首段 think 的硬截断，去掉

```bash
--enable-first-think-max-words
```

## 2) AIME 四题对照（只插入，不改错）

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_math_hard_steps.jsonl \
  --output-dir suite_math_hard_4q_insert_only \
  --corrupt-mode none \
  --apply-match-cover \
  --save-task-texts
```

## 3) AIME2025 全量 30 题（只插入）

任务文件：`tasks_aime2025.jsonl`

```bash
python run_aime.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_aime2025.jsonl \
  --output-dir suite_aime2025_insert \
  --apply-match-cover \
  --save-task-texts
```

## 4) AIME2025 全量 30 题（改错版）

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_aime2025.jsonl \
  --output-dir suite_aime2025_corrupt \
  --corrupt-mode anchor_number_shift \
  --corrupt-after-first-think \
  --corrupt-prefer-sign-flip \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --apply-match-cover \
  --save-task-texts
```

## 5) 查看 AIME 汇总

```bash
cat suite_math_hard_4q_corrupt/summary_all_models.json
cat suite_math_hard_4q_insert_only/summary_all_models.json
cat suite_aime2025_insert/summary_all_models.json
cat suite_aime2025_corrupt/summary_all_models.json
```

## 6) LiveCodeBench 标准评测

先准备 LiveCodeBench 仓库（只需一次）：

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

回到本项目后运行：

```bash
cd /path/to/close_think_experiement
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

## 7) LiveCodeBench 时间窗重评分（可选）

```bash
python run_live_code.py \
  --lcb-root /path/to/LiveCodeBench \
  --model Qwen3-32B \
  --compute-scores \
  --eval-all-file /path/to/eval_all.json \
  --start-date 2025-01-01 \
  --end-date 2025-05-01
```

## 8) 只看将执行什么命令（不真正执行）

```bash
python run_live_code.py \
  --lcb-root /path/to/LiveCodeBench \
  --model Qwen3-32B \
  --dry-run
```
