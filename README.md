# close_think_experiement

这个仓库用于验证一个问题：
在同一条生成轨迹中，中途插入 `<think>`（和引导词）后，模型是否还能保持：
1. `<think>...</think>` 正确闭合。
2. 中断点连贯续写（不重启、不大段复读）。
3. 任务格式与正确性不明显下降。

## 1. 核心流程（你现在跑的就是这个）

每个任务按同一流程执行：
1. `prefill`：把 `system + user` 输入喂给模型，拿到 KV 缓存和最后一个位置的 logits。
2. `checkpoint 截断`：继续生成到中间锚点（如 `Step 3` 或 `<Route>`）后，再走 N token 停下。
3. `前缀扰动`：在锚点附近改几个数字（模拟“中间推理出错”）。
4. `同状态分叉续写`：
   - Branch A：直接续写。
   - Branch B：先注入 `<think> + 引导词` 再续写。
5. `评估`：统计闭合、重复、格式命中、LongProc 指标。

对应代码：
- `think_branch_common.py`
- `test_close_suite.py`

## 2. KV prefill / checkpoint / 分叉到底怎么做

### 2.1 KV prefill
- 入口：`think_branch_common.py` 的 `prefill_kv(...)`。
- 做法：把长输入按 `chunk_size` 分块前向，持续累积 `past_key_values`。
- 目的：避免一次性 prefill OOM，同时拿到最后位置 logits 用于后续采样。

### 2.2 checkpoint 截断
- 入口：`generate_until_checkpoint(...)`。
- 四种模式：
  - `think_end`：首次匹配到 `</think>` 后，再生成 `checkpoint_delay` 个 token 停止。
  - `regex`：文本命中 `checkpoint_regex` 后，再生成 `checkpoint_delay` 个 token 停止。
  - `think_end_then_regex`：先等首次 `</think>`，再等 `checkpoint_regex`，最后再生成 `checkpoint_delay` 个 token 停止。
  - `think_end_mid`：先等首次 `</think>`，再在“首次 think 后正文”里按 token 区间选中段截断（可设置范围，且可在命中 `Final` 时提前截断）。
- 推荐：LongProc 一般用 `regex`，并用 `__auto__` 让脚本按任务自动选锚点。

### 2.3 分叉续写
- 入口：`test_close_suite.py` 的 `run_task_ab(...)`。
- 做法：
  1. 对“prompt + 被扰动前缀”做一次 `prefill_kv` 得到 `past_base/logits_base`。
  2. 复制 KV（`clone_past_key_values`）给 A/B，保证两分支同起点。
  3. A 直接采样；B 先注入 `inject_text` 再采样。
- 这就是“同一次生成上下文内多路径采样”的近似实现。

## 3. 为什么会看到很多“step”

有三类 step：
1. 题目内 Step（如 `- Step 1:`）
- 这是任务输出格式的一部分。
2. 实验流程 step（prefill/checkpoint/扰动/分叉）
- 这是算法执行阶段。
3. 跑批日志 step（`[1/6]`）
- 这是第几条样本，不是题目 Step。

## 4. LongProc 任务格式要求（已内置）

`test_close_suite.py` 会根据 `--longproc-task` 自动推断任务族（如 `tom_tracking_0.5k -> tom_tracking`），并追加格式约束到 system prompt，同时自动计算 `format_hit_rate`。

当前内置映射：
- `tom_tracking`：需要 `- Step X:` 轨迹行。
- `path_traversal`：需要 `<Route>...</Route>`。
- `pseudo_to_code`：需要 ```cpp ... ```。
- `html_to_tsv`：需要 ```tsv ... ```。
- `countdown`：需要 `<Solution>...</Solution>`。
- `travel_planning`：需要 `<Solving Procedure>...</Solving Procedure>` 和 `<Plan>...</Plan>`。

## 5. Prompt 迭代方式（建议固定文件）

你每次迭代 prompt，不要直接改命令行长字符串，而是改文件：
- `prompts/system_base_v1.txt`
- `prompts/system_enhanced_v1.txt`
- `prompts/inject_think_v1.txt`

其中 `inject_think_v1.txt` 现在是“只开 `<think>` 不手动闭合”，并带有通用自检上下文（说明这是中途注入、要求做局部复核并从中断点续写）；闭合由模型自行输出 `</think>`。

运行时通过参数加载：
- `--system-prompt-file`
- `--inject-text "$(cat prompts/inject_think_v1.txt)"`

这样可以复现实验并与结果目录一一对应。

## 6. 超参数速查（最关键）

- 采样：`temperature`, `top_p`, `seed`
- 截断：`checkpoint_mode`, `checkpoint_regex`, `checkpoint_delay`, `max_prefix_tokens`
- 续写：`max_new_after`
- OOM 控制：`chunk_size`
- 扰动：`corrupt_mode`, `corrupt_anchor_regex`, `corrupt_max_changes`, `corrupt_window_chars`, `corrupt_after_first_think`, `corrupt_prefer_sign_flip`
- 去重：`apply_match_cover`, `cover_*`

## 7. 结果指标怎么解释

summary 里重点看：
- `think_balanced_rate`：`<think>` 与 `</think>` 是否平衡。
- `repetition_rate`：是否出现大段前缀复读。
- `avg_overlap_prefix_to_continuation`：复读重叠长度（越低越好）。
- `format_hit_rate`：是否命中任务规定格式。
- `expected_hit_rate`：命中 `expected_regex`（仅轻量正确性）。
- `longproc_avg_metrics`：LongProc 官方 evaluator 的均值指标。

## 8. 快速运行（32B + LongProc）

### 8.1 一键三组（baseline / enhanced / enhanced+cover）
```bash
cd /auto/users/guoeng/guolei/close_think_experiement
MODEL_32B="/scratch-ssd/guoeng/huggingface/models/Qwen3-32B" \
TASK="tom_tracking_0.5k" \
N_SAMPLES=20 \
bash run_longproc_32b.sh \
  --max-prefix-tokens 1600 \
  --max-new-after 600
```

`run_longproc_32b.sh` 支持命令行覆盖参数：
- `--max-prefix-tokens`
- `--max-new-after`
- `--checkpoint-delay`
- `--checkpoint-mode think_end|regex|think_end_then_regex|think_end_mid`
- `--n-samples`
- `--task`
- `--out-root`
- `--out-suffix`（防止和旧目录重名）
- `--model-path`
- `--branch-mode ab|b`
- `--checkpoint-regex` / `--corrupt-anchor-regex`
- `--checkpoint-mid-min-tokens` / `--checkpoint-mid-max-tokens`
- `--checkpoint-mid-avoid-final-regex`
- `--corrupt-mode number_shift|anchor_number_shift|none`
- `--corrupt-after-first-think`
- `--corrupt-prefer-sign-flip`
- `--min-b-tokens-before-eos`
- `--b-retry-times`
- `--auto-close-unclosed-think`
- `--apply-cross-think-cover`
- `--print-full-output`

说明：默认不会把每题全文打印到终端，只会保存到文件。要终端直接看全文和改错信息，请加 `--print-full-output`。
说明：`tom_tracking` 默认锚点是 `- Step 3:`（带短横线），因此注入点通常在后半段结构化列表中；如果你想更早插入，可显式传 `--checkpoint-regex '(?i)step\\s*3:'`。
说明：默认不会自动补齐未闭合 `<think>`，用于保留真实行为观测；如需补齐可加 `--auto-close-unclosed-think`。
说明：如果你希望“先等第一次 think 闭合，再在正文中段截断”，用 `--checkpoint-mode think_end_mid --checkpoint-mid-min-tokens 80 --checkpoint-mid-max-tokens 220`。
说明：`--checkpoint-mid-avoid-final-regex` 用于避免注入点跑到 `Final` 之后；默认开启。
说明：`test_close_suite.py` 在 `think_end_then_regex` 下现在会正确使用 `--checkpoint-regex`。
说明：如果你希望“只在第一次 think 闭合后正文改错”，加 `--corrupt-after-first-think`；如果还希望优先改 `+/-` 符号，加 `--corrupt-prefer-sign-flip`。
说明：`--apply-cross-think-cover` 会匹配“第一次 think 后正文”和“第二次 think 后正文头”的重叠（支持 `exact/fuzzy/anchor_exact`），命中后裁掉重复段。
说明：LongProc evaluator 默认会先去掉所有 `<think>...</think>` 再评分；如需关闭，用 `--no-eval-strip-think`。

只跑 1 题、只跑 Branch B、三种模式都跑，并且终端打印：
```bash
bash run_longproc_32b.sh \
  --task tom_tracking_0.5k \
  --n-samples 1 \
  --branch-mode b \
  --checkpoint-regex '(?i)step\\s*3:' \
  --corrupt-anchor-regex '(?i)step\\s*3:' \
  --max-prefix-tokens 1600 \
  --max-new-after 600 \
  --min-b-tokens-before-eos 64 \
  --b-retry-times 2 \
  --out-suffix one_b_stdout \
  --print-full-output
```

### 8.2 单独跑一组（方便迭代 prompt）
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 20 \
  --prompt-mode enhanced \
  --system-prompt-file prompts/system_enhanced_v1.txt \
  --inject-text "$(cat prompts/inject_think_v1.txt)" \
  --checkpoint-mode regex \
  --checkpoint-regex __auto__ \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex __auto__ \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_longproc_32b/enhanced_custom \
  --save-task-texts \
  --print-full-output
```

### 8.3 三数据集完整跑（各 20 题，baseline+enhanced+enhanced_cover，A/B 都跑）
```bash
bash run_longproc_32b_3tasks.sh
```
输出目录示例：
`suite_longproc_32b_3tasks_20260218_230501/{tom_tracking_0.5k|pseudo_to_code_0.5k|path_traversal_0.5k}/{baseline|enhanced|enhanced_cover}`

### 8.4 数学题“先闭合 think，再正文中段截断后分叉”示例
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --tasks-file tasks_math_steps.jsonl \
  --prompt-mode enhanced \
  --checkpoint-mode think_end_mid \
  --checkpoint-mid-min-tokens 80 \
  --checkpoint-mid-max-tokens 220 \
  --checkpoint-mid-avoid-final-regex '(?i)\\bfinal\\s*:|\\bfinal answer\\b' \
  --checkpoint-delay 0 \
  --max-prefix-tokens 3500 \
  --max-new-after 1200 \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\\s*\\d+' \
  --corrupt-after-first-think \
  --corrupt-prefer-sign-flip \
  --branch-mode ab \
  --save-task-texts \
  --print-full-output \
  --output-dir suite_math_step5_branch_compare
```

说明：
- 这个配置对应“先生成到第一次 `</think>` 后，再在正文中段随机截断，然后改错，再做 A/B 分路续写”。
- 改错限定在第一次 `</think>` 之后；会优先尝试翻转 `+/-`，找不到再做数字扰动。

### 8.5 更难数学题（4题：Aya + Hyperbola + Complex + Token Game）
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --tasks-file tasks_math_hard_steps.jsonl \
  --prompt-mode enhanced \
  --system-prompt-file prompts/system_enhanced_v1.txt \
  --inject-text "$(cat prompts/inject_think_v1.txt)" \
  --checkpoint-mode think_end_mid \
  --checkpoint-mid-min-tokens 60 \
  --checkpoint-mid-max-tokens 180 \
  --checkpoint-mid-avoid-final-regex '(?i)\\bfinal\\s*:|\\bfinal answer\\b' \
  --checkpoint-delay 0 \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\\s*\\d+' \
  --corrupt-after-first-think \
  --corrupt-prefer-sign-flip \
  --max-prefix-tokens 5000 \
  --max-new-after 1600 \
  --branch-mode ab \
  --save-task-texts \
  --print-full-output \
  --output-dir suite_math_hard_mid_32b
```

跑完后统计这 4 题的 `expected_hit` 正确率：
```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("suite_math_hard_mid_32b/_scratch-ssd_guoeng_huggingface_models_Qwen3-32B.results.jsonl")
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
hits = []
for r in rows:
    h = r.get("branch_B", {}).get("metrics", {}).get("expected_hit")
    if h is not None:
        hits.append(bool(h))
    print(r.get("task_id"), "expected_hit=", h)
print("branch_B_expected_hit_rate =", (sum(hits) / len(hits)) if hits else None)
PY
```

## 9. 输出文件说明

每次运行会生成：
- `*.results.jsonl`：逐题详细记录（A/B全文、metrics、扰动信息）。
- `*.summary.json`：单模型汇总。
- `summary_all_models.json`：多模型总汇总。
- `*.task_texts/<task_id>/branch_A.full.txt`、`branch_B.full.txt`、`meta.json`（若开 `--save-task-texts`）。

## 10. 配套计划文档

- `todo.md`：你下一步实验清单（要跑什么命令、看什么指标、什么结果对应什么结论）。
