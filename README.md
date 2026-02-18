# close_think_experiement

用于验证一个核心问题：
在“同一次续写上下文”里，中途插入 `<think>`（加引导词）后，模型能否保持：
- `<think> ... </think>` 闭合
- 连贯续写（不复读、不重启）
- 正确性不下降

---

## 1. 你看到的“多个 step”到底是什么

这里有 3 种不同的 step，不要混淆：

1. 任务文本里的 `Step 1 / Step 2 / Step 3`
- 来自 `tasks_math_steps.jsonl` 的提示词格式（让模型按步骤输出）。
- 这个 step 是“题目输出格式”，不是代码执行步骤。

2. 实验流程的 Stage/Phase
- 在代码里是：
  - 先生成到 checkpoint（中间状态）
  - 破坏一部分前缀（造错）
  - 从同一状态分叉 A/B 继续生成
- 这是“算法流程 step”。

3. shell 脚本里的 `[Test 1]...[Test 6]`
- 来自 `run_test_matrix.sh`。
- 这是“批量实验编号 step”。

---

## 2. 代码文件作用

- `think_branch_common.py`
  - 公共核心逻辑：加载模型、KV prefill、checkpoint 截断、A/B 分叉、匹配覆盖。
- `test_close_suite.py`
  - 主实验脚本（批量跑任务、统计指标、保存结果）。
- `test_branch_manual.py`
  - 单例交互版（你手工改错后再看 A/B）。
- `tasks_math_steps.jsonl`
  - Step 格式任务集，适合 regex checkpoint。
- `run_test_matrix.sh`
  - 一键跑 baseline/enhanced/cover 与 8B/32B 对照。
- `export_full_outputs.py`
  - 从结果文件导出每题 A/B 完整文本，便于人工检查。

---

## 3. 基本逻辑（核心）

每条任务按下面流程执行：

1. 构造 prompt  
- `prompt_mode=baseline|enhanced`  
- enhanced 会追加关于 `<think>` 的行为约束（闭合、续写、不重复）。

2. 生成到 checkpoint（中间状态）  
- `checkpoint_mode=think_end`：遇到首次 `</think>` 后再走 N token 停下  
- `checkpoint_mode=regex`：匹配到如 `Step 3` 后再走 N token 停下

3. 造错（可选）  
- `number_shift`：全局找数字改动  
- `anchor_number_shift`：只在锚点附近改数字（更符合“中间步骤出错”）

4. 同状态分叉续写  
- 先 prefill 一次得到 `past_base/logits_base`  
- Branch A：直接续写  
- Branch B：先注入 `<think> + 引导词` 再续写  
- 两支都从同一前缀状态出发（只改变“是否注入 think”）。

5. 匹配覆盖（可选）  
- `--apply-match-cover` 时，对续写头部和前缀尾部做 overlap 去重：
  - exact overlap
  - fuzzy overlap（近似重复）

6. 计算指标并落盘  
- think 闭合率、重复率、overlap、expected 命中率等。

---

## 4. 超参数说明（最常用）

### 模型与采样
- `--model-paths`: 模型路径（可逗号分隔多个）
- `--dtype`: `bf16/fp16/fp32`
- `--temperature`: 采样温度，越高越发散
- `--top-p`: nucleus 采样阈值
- `--seed`: 随机种子

### Prompt 相关
- `--prompt-mode baseline|enhanced`
- `--think-word-limit`: enhanced 下建议 think 长度
- `--inject-text`: B 分支插入的文本（默认含 `<think>...</think>`）

### Checkpoint 相关
- `--checkpoint-mode think_end|regex`
- `--checkpoint-regex`: regex 模式下的锚点（例如 `(?i)step\\s*3`）
- `--checkpoint-delay`: 命中锚点后再走多少 token 停下
- `--max-prefix-tokens`: Stage1 生成上限
- `--max-new-after`: A/B 分叉后生成上限
- `--chunk-size`: prefill 分块长度（防 OOM）

### 造错相关
- `--corrupt-mode none|number_shift|anchor_number_shift`
- `--corrupt-anchor-regex`: 锚点造错时的定位 regex
- `--corrupt-max-changes`: 最多改几个数字
- `--corrupt-window-chars`: 锚点附近窗口大小

### 匹配覆盖相关
- `--apply-match-cover`
- `--cover-min-exact-overlap`
- `--cover-fuzzy-min-len`
- `--cover-fuzzy-max-len`
- `--cover-fuzzy-ratio`

### 输出相关
- `--print-full-output`: 终端打印每题 A/B 全文
- `--save-task-texts`: 保存每题 A/B 全文到 txt
- `--output-dir`: 输出目录

---

## 5. 如何看“完整输出”（你最关心）

### 方式 A：运行时直接打印
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --tasks-file tasks_math_steps.jsonl \
  --prompt-mode enhanced \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_32b_fullview \
  --print-full-output \
  --save-task-texts
```

### 方式 B：从历史结果导出全文
```bash
python export_full_outputs.py \
  --results-jsonl suite_32b_smoke/_scratch-ssd_guoeng_huggingface_models_Qwen3-32B.results.jsonl \
  --out-dir suite_32b_smoke_dump \
  --print-to-stdout
```

你重点看每题的：
- `branch_A.full.txt`
- `branch_B.full.txt`
- `meta.json`

---

## 6. 如何判断 `<think>` 是否按你期望工作

先看 summary：
- `think_balanced_rate` 是否接近 1.0（闭合）
- `branch_B.repetition_rate` 是否明显升高（说明中插 think 导致复读）

再看全文（强烈建议）：
1. 是否真出现了 `<think>`
2. 是否有 `</think>` 结尾
3. `</think>` 后是不是接着原句/原 step 继续，而不是重启答案
4. 是否重复了前缀末尾大段文本

---

## 7. 常见问题

### Q1: 为什么 public 仓库用 `git@github.com` clone 也会失败？
因为 SSH 协议仍需有效公钥认证。  
public 仓库可匿名 clone 的是 HTTPS：
```bash
git clone https://github.com/1shuimo/close_think_experiement.git
```

### Q2: 为什么 match-cover 开了却没效果？
看 `match_cover_mode_counts`。如果全是 `none`，通常是阈值太严。  
先尝试降低：
- `--cover-min-exact-overlap`
- `--cover-fuzzy-ratio`

### Q3: 32B 显存压力大怎么办？
先降：
- `--max-prefix-tokens`
- `--max-new-after`
- `--chunk-size`

---

## 8. LongProc 题目怎么跑

你现在可以不走 `tasks_math*.jsonl`，直接用 LongProc 官方加载器：

```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../bench/LongProc/data \
  --longproc-code-path ../bench/LongProc \
  --n-samples 6 \
  --prompt-mode enhanced \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_longproc_32b \
  --save-task-texts \
  --print-full-output
```

LongProc 模式下 summary 会多出：
- `branch_A.longproc_avg_metrics`
- `branch_B.longproc_avg_metrics`

这些就是 LongProc evaluator 的平均分（如 `accuracy`、`partial_accuracy`、`extraction_rate`）。
