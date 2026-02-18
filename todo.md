# TODO: close-think 实验执行清单

## 0. 实验目标

验证三件事：
1. Prompt 增强是否提升 `<think>` 闭合和续写连贯。
2. 中途注入 `<think>` 是否比直接续写更稳。
3. 若出现复读，`match-cover` 是否能降重复且不伤正确性。

## 1. 环境准备

```bash
cd /auto/users/guoeng/guolei/close_think_experiement
conda activate interleaved
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

判定：
- 能正常打印版本即可进入下一步。

## 2. 小样本冒烟（先确保流程通）

也可以直接用脚本并在命令行传长度参数：
```bash
bash run_longproc_32b.sh \
  --task tom_tracking_0.5k \
  --n-samples 6 \
  --max-prefix-tokens 1200 \
  --max-new-after 400
```

说明：`prompts/inject_think_v1.txt` 默认只注入开标签 `<think>` 和引导句，不带 `</think>`，闭合由模型自己完成。
默认不打印每题全文到终端，只保存到文件。若要终端直接看输出，加 `--print-full-output`。
默认不自动补齐未闭合 `<think>`（保留真实行为）；若要补齐可加 `--auto-close-unclosed-think`，并在 `branch_B_retry.forced_close_think` 里查看触发次数。
若你要“第一次 think 必须先闭合，再注入”，请使用 `--checkpoint-mode think_end_then_regex`。
若你要“第一次 think 后正文尾 vs 第二次 think 后正文头”的去重，请开启 `--apply-cross-think-cover`，并查看 `branch_B_cross_think_cover` 字段。

你当前要的模式（1 题、只跑 Branch B、三种模式、终端打印、带后缀防覆盖）：
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

等价的单组 Python 命令如下（用于精细控制）：
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 6 \
  --prompt-mode baseline \
  --system-prompt-file prompts/system_base_v1.txt \
  --checkpoint-mode regex \
  --checkpoint-regex __auto__ \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex __auto__ \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_smoke_32b_baseline \
  --save-task-texts
```

看：`suite_smoke_32b_baseline/summary_all_models.json`

判定：
- 有 summary 文件，且 `n_tasks=6`，流程正常。

## 3. 主实验矩阵（同一任务先比三组）

### 3.1 baseline
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 50 \
  --prompt-mode baseline \
  --system-prompt-file prompts/system_base_v1.txt \
  --checkpoint-mode regex \
  --checkpoint-regex __auto__ \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex __auto__ \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_32b_tom_baseline \
  --save-task-texts
```

### 3.2 enhanced prompt
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 50 \
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
  --output-dir suite_32b_tom_enhanced \
  --save-task-texts
```

### 3.3 enhanced + match-cover
```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 50 \
  --prompt-mode enhanced \
  --system-prompt-file prompts/system_enhanced_v1.txt \
  --inject-text "$(cat prompts/inject_think_v1.txt)" \
  --apply-match-cover \
  --checkpoint-mode regex \
  --checkpoint-regex __auto__ \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex __auto__ \
  --checkpoint-delay 120 \
  --max-prefix-tokens 1200 \
  --max-new-after 400 \
  --output-dir suite_32b_tom_enhanced_cover \
  --save-task-texts
```

## 4. 指标到结论（必须按这个规则读）

从每个目录的 `summary_all_models.json` 读取：
- `branch_A.think_balanced_rate` / `branch_B.think_balanced_rate`
- `branch_A.repetition_rate` / `branch_B.repetition_rate`
- `branch_A.avg_overlap_prefix_to_continuation` / `branch_B.avg_overlap_prefix_to_continuation`
- `branch_A.format_hit_rate` / `branch_B.format_hit_rate`
- `branch_A.longproc_avg_metrics` / `branch_B.longproc_avg_metrics`

结论规则：
1. Prompt 增强有效
- enhanced 相对 baseline：`think_balanced_rate` 上升，且 `longproc_avg_metrics` 不下降（最好上升）。

2. 注入 think 有价值
- 同一组内 Branch B 相对 Branch A：
  - `think_balanced_rate` 不低于 A，
  - `repetition_rate` 不高于 A，
  - `longproc_avg_metrics` 不低于 A。

3. match-cover 有效
- enhanced+cover 相对 enhanced：
  - `repetition_rate` 或 `avg_overlap` 下降，
  - `longproc_avg_metrics` 不明显下降（建议下降 < 1 个百分点）。

## 5. 全文抽查（不要只看分数）

```bash
python export_full_outputs.py \
  --results-jsonl suite_32b_tom_enhanced/_scratch-ssd_guoeng_huggingface_models_Qwen3-32B.results.jsonl \
  --out-dir suite_32b_tom_enhanced_dump \
  --print-to-stdout
```

抽查要点：
1. Branch B 是否真的插入了 `<think>`，并且后续由模型自行闭合 `</think>`。
2. `</think>` 后是否从中断点继续，而不是重启回答。
3. 是否出现大段复读。
4. 任务格式是否满足（如 `- Step X:`、`<Route>`、代码块标签等）。

## 6. Prompt 迭代规范（每次改动都留痕）

每次修改都按下面做：
1. 新建版本文件：
- `prompts/system_enhanced_v2.txt`
- `prompts/inject_think_v2.txt`

2. 运行一组固定对照（建议 `n_samples=50`）并写目录名：
- `suite_32b_tom_enhanced_v2`

3. 在 `README.md` 追加：
- 本次改动点（改了哪几条约束）。
- 对比上一版的关键指标变化。
- 是否保留本版。

## 7. 可选扩展（跨任务泛化）

把 `--longproc-task` 换成：
- `path_traversal_0.5k`
- `pseudo_to_code_0.5k`
- `html_to_tsv_0.5k`
- `countdown_0.5k`
- `travel_planning_0.5k`

命令不变，`checkpoint_regex` 和 `corrupt_anchor_regex` 保持 `__auto__` 即可。
