# close_think_experiement

这个仓库当前只关注一条核心实验线：

- 在模型已经开始生成之后，
- 在中途选一个位置插入新的 `<think>`，
- 观察模型是否会重新进入原生 `<think> ... </think>` 思考分布，
- 并在 `</think>` 之后从目标位置继续往下写。

当前只保留两个 benchmark：

- AIME
- LiveCodeBench

详细命令见 [RUN_COMMANDS.md](/Users/shuimo/Desktop/interleaved/close/RUN_COMMANDS.md)。

默认输出现在统一放在 `outputs/` 下，并且单模型文件名只使用模型目录 basename。
例如：

- `/scratch-ssd/guoeng/huggingface/models/Qwen3-32B`
- 会写成 `Qwen3-32B.results.jsonl`
- 不再写成整条 `_scratch-ssd_...` 路径

## 背景

这条实验线要回答的问题是：

1. 如果模型在一次生成过程中已经写到一半，能不能通过中途插入一个新的 `<think>`，把它重新拉回原生的思考轨迹？
2. 这种插入，能不能让模型在局部不确定点重新检查最近的推理步骤，而不是直接复读前文或重新从头开始？
3. 如果前缀里故意放入一个局部错误，插入后的原生 think 能不能帮助模型修正后续续写？

所以这里的重点不是 prompt engineering 本身，而是：

- 中途介入生成轨迹；
- 在一个局部位置重新触发 think；
- 看这个 think 对后续 continuation 的影响。

## 方法

### 方法概述

三个核心入口是：

- [run_aime_plain.py](/Users/shuimo/Desktop/interleaved/close/run_aime_plain.py)
- [run_aime.py](/Users/shuimo/Desktop/interleaved/close/run_aime.py)
- [run_aime_corrupt.py](/Users/shuimo/Desktop/interleaved/close/run_aime_corrupt.py)
- [run_lcb_insert.py](/Users/shuimo/Desktop/interleaved/close/run_lcb_insert.py)

它们背后分别调用：

- [test_close_suite.py](/Users/shuimo/Desktop/interleaved/close/test_close_suite.py)
- [test_close_suite_corrupt.py](/Users/shuimo/Desktop/interleaved/close/test_close_suite_corrupt.py)

公共逻辑在：

- [think_branch_common.py](/Users/shuimo/Desktop/interleaved/close/think_branch_common.py)

### 插入是怎么做的

当前实现不是直接改 hidden state，也不是在已有 KV cache 中间硬插 token。

实际流程是：

1. 先正常构造 prompt，并保留模型原生 thinking。
2. 让模型先生成一段 prefix，直到某个 checkpoint 停下来。
3. 在 prefix 中定位一个插入点。
4. 取插入点左边的文本作为新的 Branch B 前缀。
5. 把注入文本 `inject_text` 接到这个前缀后面。
6. 对新的前缀重新 tokenize / prefill，然后继续生成。

因此这套方法更准确地说是：

- 用 token 流决定“停在哪里”和“候选插入点在哪里”；
- 但真正执行插入时，用的是文本切片 + 重新 prefill。

### 插入点怎么定位

插入点有两种主要来源：

1. 文本锚点  
   对 AIME，优先在 `Step N:` 这类结构化正文里定位。

2. token offset fallback  
   如果没有合适的 step 锚点，就用 tokenizer 把一个 token offset 近似映射成字符位置，再吸附到最近句末或行尾。

也就是说：

- tokenizer 会参与“token offset -> 字符位置”的映射；
- 但最终 `inject_pos` 是一个字符位置；
- 真正的插入是把文本切成左右两半，再把 `<think>` 放进中间。

### checkpoint 机制

prefix 会先生成到 checkpoint 再停。

当前常用模式有：

- `think_end`
- `regex`
- `think_end_then_regex`
- `think_end_mid`

实际最常用的是：

- AIME：先等第一个原生 `</think>`，再在正文里插
- LCB：先等第一个原生 `</think>`，再向后走更长一段正文 token，再插

### 注入文本

默认注入文本是显式以 `<think>` 开头，但不提前给出 `</think>`。

目的不是替模型写 reasoning，而是把模型重新推回原生的 think 分布，让它自己完成：

- `<think> ... </think>`
- 然后继续写最终答案或最终代码

相关 prompt 文件：

- [prompts/inject_think_v2.txt](/Users/shuimo/Desktop/interleaved/close/prompts/inject_think_v2.txt)
- [prompts/inject_think_codegen_aime_like_v2.txt](/Users/shuimo/Desktop/interleaved/close/prompts/inject_think_codegen_aime_like_v2.txt)

## 核心脚本

### `run_aime_plain.py`

用途：

- AIME 原始能力基线
- 不做插入
- 不做改错

它回答的问题是：

- 模型在没有任何中插干预时，原始 AIME 能力有多强？

### `run_aime.py`

用途：

- AIME 插入版
- 只做 Branch B
- 不做改错

它回答的问题是：

- 单纯中途插入 `<think>`，是否能稳定拉起新的原生 think，并继续解题？

### `run_aime_corrupt.py`

用途：

- AIME 改错版
- 支持 Branch A / Branch B 对照
- 在前缀里先做局部扰动，再看中插 `<think>` 的效果

它回答的问题是：

- 如果前缀推理局部出错，Branch B 的插入式 think 是否比直接续写更有机会纠正后续推理？

### `run_lcb_insert.py`

用途：

- LiveCodeBench 插入版
- 只做 Branch B
- 不做改错

它回答的问题是：

- 在代码生成已经进行到一半时，能否重新插入 think，并让模型继续输出更稳定的最终代码？

## 数据集

### AIME

当前主要用到这些任务文件：

- [data/tasks_aime2025.jsonl](/Users/shuimo/Desktop/interleaved/close/data/tasks_aime2025.jsonl)
  - AIME2025 全量题目
- [data/tasks_math_mix5_corrupt.jsonl](/Users/shuimo/Desktop/interleaved/close/data/tasks_math_mix5_corrupt.jsonl)
  - 小规模改错对比集
  - 每题自带 `corrupt_plan`、`corrupt_anchor_regex`、`corrupt_note`
- [data/tasks_math_hard_steps.jsonl](/Users/shuimo/Desktop/interleaved/close/data/tasks_math_hard_steps.jsonl)
  - 更难的 step-structured 数学题
- [data/tasks_aime2_hard2.jsonl](/Users/shuimo/Desktop/interleaved/close/data/tasks_aime2_hard2.jsonl)
  - AIME + hard 数学混合集

AIME 题目一般有：

- `user_prompt`
- `expected_regex`

有些改错任务还会额外指定：

- `corrupt_plan`
- `corrupt_note`

### LiveCodeBench

LCB 这条线使用：

- `livecodebench/code_generation_lite`
- 默认 release：`release_v6`

题目不是直接手写，而是通过：

- [prepare_lcb_codegen_tasks.py](/Users/shuimo/Desktop/interleaved/close/prepare_lcb_codegen_tasks.py)

导出为本仓库自己的 `tasks.jsonl` 格式，例如：

- [data/tasks_lcb_1q.jsonl](/Users/shuimo/Desktop/interleaved/close/data/tasks_lcb_1q.jsonl)

## 怎么验证

### 1. 插入版验证

对 insertion-only 任务，核心看三件事：

1. 能不能重新形成完整的 `<think> ... </think>`
2. `</think>` 之后会不会复读刚刚的 prefix
3. 最终答案 / 最终代码是否仍然有效

对应的主要指标是：

- `think_balanced_rate`
- `repetition_rate`
- `avg_overlap_prefix_to_continuation`
- `expected_hit_rate`

这些汇总会写到：

- `*.summary.json`
- `summary_all_models.json`

### 2. AIME 改错对比

改错对比是当前最重要的验证方式。

对同一个被扰动过的 prefix，会生成两个分支：

- Branch A：直接续写
- Branch B：在定位点插入新的 `<think>`，再续写

你真正关心的是：

1. 在同一份错误前缀上，Branch B 是否比 Branch A 更容易恢复正确答案
2. Branch B 是否比 Branch A 更少复读
3. Branch B 的 think 是否更容易完整闭合

在 AIME 这条线上，最直接的判据就是：

- `expected_hit`
- `expected_hit_rate`

如果 Branch B 的 `expected_hit_rate` 高于 Branch A，而且复读不更严重，那么就说明插入式 think 在“改错/补救”这个任务上有价值。

### 3. LiveCodeBench 验证

LCB 不走 `expected_regex` 这条线，而是把 Branch B 最终输出接回官方 code generation 评测。

流程是：

1. 跑 [run_lcb_insert.py](/Users/shuimo/Desktop/interleaved/close/run_lcb_insert.py)
2. 用 [evaluate_lcb_from_suite.py](/Users/shuimo/Desktop/interleaved/close/evaluate_lcb_from_suite.py) 读取 Branch B
3. 去掉 `<think> ... </think>`
4. 提取最终代码块
5. 计算 LCB codegen metrics，例如 `pass@1`

所以 LCB 这条线验证的是：

- 插入式 think 是否有助于最终可执行代码质量

## 输出文件

每次运行的主输出通常包括：

- `*.results.jsonl`
- `*.summary.json`
- `summary_all_models.json`

如果开了 `--save-task-texts`，还会有逐题导出：

- `branch_A.full.txt`
- `branch_B.full.txt`
- `meta.json`

对于改错版，还会额外生成：

- `*.branch_b_view.jsonl`
- `*.branch_b_view.md`

这些文件适合快速人工检查：

- 改错发生在哪里
- 插入发生在哪里
- Branch B 最终全文长什么样

## 最小运行入口

常用入口只有这三个：

```bash
python run_aime_plain.py ...
python run_aime.py ...
python run_aime_corrupt.py ...
python run_lcb_insert.py ...
```

更完整的命令见：

- [RUN_COMMANDS.md](/Users/shuimo/Desktop/interleaved/close/RUN_COMMANDS.md)

## 当前维护范围

如果只保留这条核心实验线，真正需要维护的文件是：

- [run_aime_plain.py](/Users/shuimo/Desktop/interleaved/close/run_aime_plain.py)
- [run_aime.py](/Users/shuimo/Desktop/interleaved/close/run_aime.py)
- [run_aime_corrupt.py](/Users/shuimo/Desktop/interleaved/close/run_aime_corrupt.py)
- [run_lcb_insert.py](/Users/shuimo/Desktop/interleaved/close/run_lcb_insert.py)
- [test_close_suite.py](/Users/shuimo/Desktop/interleaved/close/test_close_suite.py)
- [test_close_suite_corrupt.py](/Users/shuimo/Desktop/interleaved/close/test_close_suite_corrupt.py)
- [think_branch_common.py](/Users/shuimo/Desktop/interleaved/close/think_branch_common.py)
- [prepare_lcb_codegen_tasks.py](/Users/shuimo/Desktop/interleaved/close/prepare_lcb_codegen_tasks.py)
- [evaluate_lcb_from_suite.py](/Users/shuimo/Desktop/interleaved/close/evaluate_lcb_from_suite.py)

其它脚本可以视为历史遗留，或者至少不属于这条主线。
