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

## 为什么用 `(?i)step\s*3`

- `(?i)`：大小写不敏感，`Step` / `step` 都能匹配。
- `step\s*3`：把 checkpoint 放在模型已经输出到 `Step 3` 附近之后。
- 目的：避免注入点太早落在 `Step 0/1`，给“先写步骤、再插入 `<think>`”留足上下文。

如果你希望更晚再插入，可以改成 `(?i)step\s*4` 或 `(?i)step\s*5`。

## 新增改错能力

`run_aime_corrupt.py` / `test_close_suite_corrupt.py` 现支持：

- `--corrupt-mode sign_flip`
  - 仅做符号翻转（`+/-`，以及比较符 fallback：`<= >= == != < >`）。
- `--locator-only`
  - 只做插入点定位，不做任何改错（数字/符号都不改）。
  - 规则：`Step优先`（受 `--corrupt-min-step` 控制）→ `Token兜底`（`--no-step-fallback-offset-tokens`）。
- `--corrupt-mode sign_then_number`
  - 先尝试符号翻转，失败再做数字改动。
- `--corrupt-mode sign_and_number`
  - 同一题优先做“符号 + 数字”两次改动（若符号不可改，会至少尝试数字）。
- `--corrupt-min-step N`
  - 仅在 `Step N` 及之后改错。  
  - 例如 `--corrupt-min-step 2` 表示跳过 `Step 0/1`，从 `Step 2+` 开始改。
- `--enable-first-think-max-words`
  - 开启后才会启用 `--first-think-max-words` 这条硬截断。
  - 默认关闭。
- `--first-think-max-words N`
  - 仅在开启 `--enable-first-think-max-words` 时生效。
  - 对第一段 `<think>` 做硬截断（并强制补 `</think>`），避免第一段过长占满 prefix。
- `--step-wait-extra-tokens N`
  - 在 `think_end_then_regex + --corrupt-after-first-think` 下，如果还没看到 `Step` 行，会继续生成前缀最多 `N` token 再尝试改错/插入。
- `--no-step-fallback-offset-tokens N`
  - 如果依然没有 `Step` 行，会在首个 `</think>` 后约 `N` token 先定位，再对齐到附近句末标点（如 `.` `。` `;`）作为注入点，并在附近做数字改动。
  - 若首个 `</think>` 本身都没出现，会先在该锚点强制闭合第一段 think，再按原流程继续（改错与注入）。
  - 默认 `300`；设为 `0` 或负数可关闭这个 fallback。
- `--enable-think-word-limit`
  - 开启后才会启用 `--think-word-limit` 这条软约束。
  - 默认关闭（等价于“先注释掉 think-word-limit 逻辑”）。
- `--think-word-limit N`
  - 仅在开启 `--enable-think-word-limit` 时生效；用于 system prompt 软提示。
  - 这是软约束，不是硬截断；硬截断请用 `--first-think-max-words`。

## 结果文件（更易读）

除原有 `*.results.jsonl` / `*.summary.json` 外，改错脚本会自动额外生成：

- `*.branch_b_view.jsonl`
  - 每题保留 `task_id`、`corrupt_summary`、`branch_B.full_text`，便于程序筛选。
- `*.branch_b_view.md`
  - 人类可读版：每题先显示“改了什么”，再显示 Branch B 全文。

当开启 `--save-task-texts` 时，每题目录还会新增：

- `branch_B.view.md`
  - 单题可读版（改错摘要 + Branch B 全文）。

## 2 AIME + 2 Hard 数据集

已提供混合任务文件：

- `tasks_aime2_hard2.jsonl`
  - `aime2025_i_01`
  - `aime2025_i_02`
  - `hard_aya_walk`
  - `hard_hyperbola_rhombus`
- `tasks_aime2_hard2_signnum_step2.jsonl`
  - 在上面 4 题基础上，预写入：
    - `corrupt_plan=sign_and_number`
    - `corrupt_min_step=2`
    - `corrupt_after_first_think=true`

## 推荐命令（四题、符号+数字、Step2 以后改）

```bash
cd /path/to/close_think_experiement
MODEL=/scratch-ssd/guoeng/huggingface/models/Qwen3-32B

python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_aime2_hard2.jsonl \
  --output-dir suite_aime2_hard2_corrupt \
  --checkpoint-mode think_end_then_regex \
  --checkpoint-regex '(?i)step\s*3' \
  --enable-first-think-max-words \
  --first-think-max-words 120 \
  --step-wait-extra-tokens 2000 \
  --no-step-fallback-offset-tokens 300 \
  --corrupt-mode sign_and_number \
  --corrupt-min-step 2 \
  --corrupt-after-first-think \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --apply-match-cover \
  --save-task-texts
```

如需再打开软限制，可额外加：

```bash
--enable-think-word-limit --think-word-limit 60
```

只做“单分支插入评测（不改错）”建议命令：

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file tasks_aime2025.jsonl \
  --output-dir corrupt_exp_v1 \
  --branch-mode b \
  --locator-only \
  --corrupt-mode none \
  --corrupt-min-step 2 \
  --corrupt-after-first-think \
  --checkpoint-mode think_end_then_regex \
  --checkpoint-regex '(?i)step\s*3' \
  --no-step-fallback-offset-tokens 300 \
  --save-task-texts
```

评测命中逻辑：看 `branch_B.metrics.expected_hit`（是否命中题目 `expected_regex`）。
- 该值现在基于“先去掉所有 `<think>...</think>` 片段”后的文本计算。
- 原始未清洗文本命中可看 `branch_B.metrics.expected_hit_raw`。
