# AIME / LCB Core Runner Guide

本文档只保留当前这条核心实验线：

- `run_aime_plain.py`
- `run_aime.py`
- `run_aime_corrupt.py`
- `run_lcb_insert.py`

除此之外的脚本、历史实验和其它 bench，全部不在本文档范围内。

## 1. 这条实验线到底在做什么

目标很明确：

1. 先让模型正常生成一段前缀；
2. 在你指定的生成位置，把一段新的 `<think>` 文本插进去；
3. 让模型从这个被修改过的前缀继续生成；
4. 观察模型是否会重新进入原生的 `<think> ... </think>` 思考分布，并在 `</think>` 之后从目标位置继续往下写。

这里有一个重要边界要说清楚：

- 当前实现不是 hidden state 编辑；
- 也不是直接改 KV cache 内部状态；
- 这里的“token level”意思是：插入点是根据生成出来的 token 流和 token 数量来决定的；
- 真正的干预形式是：先生成到某个 token 边界，再把注入文本 splice 进 prefix，然后从这个新 prefix 继续生成。

## 2. 三个核心 runner 的共同机制

这三个入口脚本背后都遵循同一条主流程。

### 2.1 先生成 prefix

先按正常 chat prompt 喂给模型，让模型生成一段前缀。

对这三个核心 runner 来说，默认都保留模型原生 thinking 能力，也就是底层 `apply_chat_template(..., enable_thinking=True)` 这条链路仍然打开。

### 2.2 在 checkpoint 截断

前缀不会一直生成到结尾，而是会在一个 checkpoint 停下来。这个 checkpoint 由 `--checkpoint-mode` 控制。

- `think_end`
  - 一看到第一个 `</think>` 就停
- `regex`
  - 一旦命中某个正则锚点就停
- `think_end_then_regex`
  - 先等第一个 `</think>`，然后只在 `</think>` 后面的正文里找 regex
- `think_end_mid`
  - 先等第一个 `</think>`，然后再往后走一小段正文 token，在正文中段停下来

### 2.3 选择插入位置

前缀停下后，代码会决定把新的 `<think>` 插到哪里。

对 AIME：

- 默认优先在第一个 `</think>` 之后的正文里找位置；
- 能找到 step/body 锚点就优先用锚点；
- 找不到时再退回到 token offset，比如 `300`。

对 LiveCodeBench：

- 不走数学 step 逻辑；
- 主要依赖 `think_end_mid` 和 token fallback；
- 默认会比 AIME 多等一段正文，因为代码题需要更长的上下文再插入。

### 2.4 注入文本

注入文本通常以 `<think>` 开头，要求模型做一次局部复查。

AIME 默认注入文件：

- `prompts/inject_think_v2.txt`

LCB 默认注入文件：

- `prompts/inject_think_codegen_aime_like_v2.txt`

这些注入文本的目的不是替模型写完整 reasoning，而是把模型重新推回原生 think 分布，让它自己完成 `</think>`，然后自然地接着往下写。

### 2.5 继续生成与打分

插入之后，模型继续生成，得到 Branch B。

不同 runner 是否还有 Branch A，取决于具体脚本。

主输出指标包括：

- `think_balanced`
- `repetition_flag`
- `overlap_prefix_to_continuation`
- `expected_hit`

如果担心插入后复读，可以打开：

- `--apply-match-cover`
- `--apply-cross-think-cover`

## 3. 三个 runner 分别负责什么

### 3.1 `run_aime_plain.py`

定位：

- AIME 原始能力基线脚本
- 不做插入
- 不做改错
- 直接按题生成答案

底层调用：

- 直接在脚本内部调用 `think_branch_common.py`

默认配置：

- 任务文件：`data/tasks_aime2025.jsonl`
- 输出目录：`outputs/aime/suite_aime2025_plain`
- 默认不额外加 system prompt

适用场景：

- 你想先看模型本身的 AIME 能力
- 你需要一个没有任何中插干预的 baseline

### 3.2 `run_aime.py`

定位：

- AIME 插入版；
- 只跑 Branch B；
- 不做改错，不做前缀扰动。

底层调用：

- `test_close_suite.py`

默认配置：

- 任务文件：`data/tasks_aime2025.jsonl`
- 输出目录：`outputs/aime/suite_aime2025_insert`
- system prompt：`prompts/system_enhanced_v1.txt`
- inject text：`prompts/inject_think_v2.txt`
- `branch-mode` 固定为 `b`
- 默认 checkpoint 模式：`think_end_mid`

适用场景：

- 你只想看“中途插入 `<think>` 本身”有没有效果；
- 你不想先人为破坏 prefix。

### 3.3 `run_aime_corrupt.py`

定位：

- AIME 改错版；
- 支持插入 + 改错 / 扰动；
- 支持 Branch A / Branch B 对照。

底层调用：

- `test_close_suite_corrupt.py`

默认配置：

- 任务文件：`data/tasks_aime2025.jsonl`
- 输出目录：`outputs/aime/suite_aime2025_corrupt`
- system prompt：`prompts/system_enhanced_v1.txt`
- inject text：`prompts/inject_think_v2.txt`
- 默认 `branch-mode = ab`
- 默认 `corrupt-mode = anchor_number_shift`
- 默认锚点：`(?i)step\\s*\\d+`
- 默认 `corrupt-min-step = 2`

适用场景：

- 你想看插入后的原生 think 能不能对局部错误产生修复；
- 你想直接比较同一份被扰动 prefix 上，Branch A 和 Branch B 的差异。

### 3.4 `run_lcb_insert.py`

定位：

- LiveCodeBench 插入版；
- 只跑 Branch B；
- 不做改错。

底层调用：

- `test_close_suite.py`

默认配置：

- 任务文件需要先单独准备；
- 输出目录：`outputs/lcb/suite_lcb_insert`
- system prompt：`prompts/system_lcb_r1_insert_v2.txt`
- inject text：`prompts/inject_think_codegen_aime_like_v2.txt`
- 默认 `prompt-mode = baseline`
- checkpoint 固定为 `think_end_mid`
- `checkpoint-mid-min-tokens = 500`
- `checkpoint-mid-max-tokens = 500`
- `no-step-fallback-offset-tokens = 500`
- `step-wait-extra-tokens = 0`
- `branch-mode` 固定为 `b`

适用场景：

- 你想在代码生成中段强行 reopen 一个 think block；
- 你希望模型在 `</think>` 后继续写最终可运行的 Python 代码。

## 4. 真正重要的参数

如果你只关心“插入位置”和“插入后效果”，下面这些参数最重要。

### 4.1 prefix 截在哪里

- `--checkpoint-mode`
- `--checkpoint-regex`
- `--checkpoint-mid-min-tokens`
- `--checkpoint-mid-max-tokens`
- `--checkpoint-delay`

理解方式：

- 插得太早，就把 `checkpoint-mid-*` 调大；
- 插得太晚，就把 `checkpoint-mid-*` 调小，或者改用 regex anchor。

### 4.2 插在哪

- `--corrupt-after-first-think`
- `--no-step-fallback-offset-tokens`
- `--force-inject-at-corrupt`
- `--force-inject-at-sentence-end`

理解方式：

- `--corrupt-after-first-think` 表示定位插入点时，优先只看第一个 `</think>` 之后的正文；
- `--no-step-fallback-offset-tokens` 是找不到更好锚点时的 token 级兜底位置；
- `--force-inject-at-corrupt` 表示 Branch B 不再简单 append 到尾部，而是强制插在定位到的位置上。

### 4.3 第一个原生 think 保留多少

- `--enable-think-word-limit`
- `--think-word-limit`
- `--enable-first-think-max-words`
- `--first-think-max-words`
- `--enable-first-think-smooth-close`

理解方式：

- 如果模型第一个原生 think 太长，把真正的插入点推得太靠后，这组参数就很重要；
- 它们控制的是 prefix 里第一个原生 think，而不是你后插进去的 think。

### 4.4 复读抑制

- `--apply-match-cover`
- `--apply-cross-think-cover`

理解方式：

- 如果模型在 `</think>` 后总是重复左边刚写过的内容，就打开这两个开关试。

## 5. 最小运行命令

假设：

```bash
cd /path/to/close
MODEL=/scratch-ssd/guoeng/huggingface/models/Qwen3-32B
```

### 5.1 AIME 原始能力基线

最小命令：

```bash
python run_aime_plain.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_plain \
  --save-task-texts
```

如果你想测试“完全不开原生 thinking”的版本：

```bash
python run_aime_plain.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_plain_no_think \
  --disable-model-thinking
```

### 5.2 AIME 插入版

最小命令：

```bash
python run_aime.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_insert \
  --save-task-texts
```

推荐的“先等原生 think 结束，再在正文中段插入”配置：

```bash
python run_aime.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_insert \
  --checkpoint-mode think_end_mid \
  --checkpoint-mid-min-tokens 20 \
  --checkpoint-mid-max-tokens 30 \
  --no-step-fallback-offset-tokens 300 \
  --corrupt-after-first-think \
  --save-task-texts
```

如果你想按 Step 锚点插：

```bash
python run_aime.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_insert_step3 \
  --checkpoint-mode think_end_then_regex \
  --checkpoint-regex '(?i)step\\s*3' \
  --no-step-fallback-offset-tokens 300 \
  --corrupt-after-first-think \
  --force-inject-at-corrupt \
  --save-task-texts
```

### 5.3 AIME 改错版

标准改错配置：

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_corrupt \
  --corrupt-mode anchor_number_shift \
  --corrupt-after-first-think \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --save-task-texts
```

如果你想做“优先翻符号，找不到再改数字”的修复测试：

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_math_mix5_corrupt.jsonl \
  --output-dir outputs/math/suite_math_mix5_corrupt \
  --corrupt-mode anchor_number_shift \
  --corrupt-prefer-sign-flip \
  --corrupt-min-step 2 \
  --corrupt-after-first-think \
  --force-inject-at-corrupt \
  --force-inject-at-sentence-end \
  --save-task-texts \
  --print-full-output
```

如果你只想定位插入点，不真正改 prefix：

```bash
python run_aime_corrupt.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_aime2025.jsonl \
  --output-dir outputs/aime/suite_aime2025_locator_only \
  --locator-only \
  --corrupt-mode none \
  --corrupt-after-first-think \
  --checkpoint-mode think_end_then_regex \
  --checkpoint-regex '(?i)step\\s*3' \
  --no-step-fallback-offset-tokens 300 \
  --save-task-texts
```

### 5.4 LiveCodeBench 插入版

先把 LCB 题目导出成 close runner 需要的任务格式：

```bash
python prepare_lcb_codegen_tasks.py \
  --question-id <LCB_QUESTION_ID> \
  --limit 1 \
  --output-jsonl data/tasks_lcb_1q.jsonl
```

然后跑插入版：

```bash
python run_lcb_insert.py \
  --model-paths "$MODEL" \
  --tasks-file data/tasks_lcb_1q.jsonl \
  --output-dir outputs/lcb/suite_lcb_1q_insert \
  --save-task-texts
```

LCB 这条线默认已经假设：

- 不加数学 step 格式引导；
- 先等第一个原生 think 结束，再往正文走 500 token 左右；
- 然后强制在定位点插入新的 `<think>`。

如果你想直接看 system prompt 和全文输出，加：

```bash
--print-full-output
```

## 6. LiveCodeBench 评测怎么接回去

`run_lcb_insert.py` 只负责生成 close runner 的输出，不负责官方 codegen 评分。

要把 Branch B 的结果接回 LCB 评测，用：

```bash
python evaluate_lcb_from_suite.py \
  --suite-results-jsonl outputs/lcb/suite_lcb_1q_insert/Qwen3-32B.results.jsonl \
  --branch b \
  --strip-think \
  --problems-json data/tasks_lcb_1q.jsonl.problems.json \
  --output-dir outputs/lcb/suite_lcb_1q_insert/lcb_eval
```

这一步做的事情是：

1. 读取 Branch B 全文；
2. 去掉 `<think> ... </think>`；
3. 提取最终代码块；
4. 按 LiveCodeBench 的 codegen 指标去评测。

## 7. 输出文件怎么看

三个 runner 通用的主输出：

- `<output_dir>/<model_label>.results.jsonl`
- `<output_dir>/<model_label>.summary.json`
- `<output_dir>/summary_all_models.json`

其中 `model_label` 默认取模型目录 basename。
例如 `/scratch-ssd/guoeng/huggingface/models/Qwen3-32B` 会写成 `Qwen3-32B.results.jsonl`。

如果打开 `--save-task-texts`，还会有每题单独目录，里面通常包括：

- `branch_A.full.txt`（如果该 runner 有 Branch A）
- `branch_B.full.txt`
- `meta.json`

对 `run_aime_corrupt.py`，还会多出：

- `<safe_model_name>.branch_b_view.jsonl`
- `<safe_model_name>.branch_b_view.md`

这两个文件适合快速人工看被扰动后的 Branch B。

## 8. 任务文件最小格式

### 8.1 AIME 任务格式

典型字段：

```json
{
  "id": "aime2025_i_01",
  "user_prompt": "Problem text...",
  "expected_regex": "(?m)^\\s*167167\\s*$"
}
```

常见可选字段：

- `expected_answer`
- `reference_output`
- `corrupt_plan`
- `corrupt_note`

### 8.2 LiveCodeBench 任务格式

通常由 `prepare_lcb_codegen_tasks.py` 生成。

典型字段：

```json
{
  "id": "3674",
  "question_id": "3674",
  "user_prompt": "LCB codegen prompt...",
  "expected_regex": null,
  "reference_output": null,
  "lcb_meta": {
    "question_title": "...",
    "platform": "...",
    "contest_date": "...",
    "difficulty": "..."
  }
}
```

## 9. 当前应该维护哪些文件

如果你现在的研究目标就是：

- 在生成过程中插入新的 `<think>`，
- 重启模型原生 think 分布，
- 让模型在指定位置局部重思考，
- bench 只看 AIME 和 LiveCodeBench，

那你现在真正需要维护的文件只有：

- `run_aime.py`
- `run_aime_plain.py`
- `run_aime_corrupt.py`
- `run_lcb_insert.py`
- `prepare_lcb_codegen_tasks.py`
- `evaluate_lcb_from_suite.py`
- `test_close_suite.py`
- `test_close_suite_corrupt.py`
- `think_branch_common.py`

其余脚本都可以视为历史遗留，或者至少不属于你这条核心实验线。
