# Test Plan (Close Think)

## A. 脚本职责（先搞清楚每个脚本做什么）

1. `test_close_suite.py`
- 主实验脚本。
- 输入任务 -> checkpoint 截断 -> 前缀扰动 -> A/B 分叉续写 -> 打分汇总。
- 适合做对比实验和统计。

2. `run_longproc_32b.sh`
- 32B 一键矩阵：`baseline / enhanced / enhanced+cover`。
- 适合快速跑一轮稳定对照。

3. `test_branch_manual.py`
- 单样本人工观察版。
- 适合看具体失败案例和输出细节。

4. `export_full_outputs.py`
- 从 `results.jsonl` 导出每题 A/B 全文。
- 适合人工审阅“是否闭合/是否复读/是否续写连贯”。

## B. 最小可复现（Smoke）

```bash
python test_close_suite.py \
  --model-paths /scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --longproc-task tom_tracking_0.5k \
  --longproc-data-path ../LongProc/data \
  --longproc-code-path ../LongProc \
  --n-samples 6 \
  --prompt-mode enhanced \
  --system-prompt-file prompts/system_enhanced_v1.txt \
  --inject-text "$(cat prompts/inject_think_v1.txt)" \
  --checkpoint-mode regex \
  --checkpoint-regex __auto__ \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex __auto__ \
  --output-dir suite_smoke
```

通过标准：
- 生成 `suite_smoke/summary_all_models.json`。
- `n_tasks == 6`。

## C. 主对照（Prompt/覆盖策略）

按顺序跑：
1. baseline（`system_base_v1.txt`）
2. enhanced（`system_enhanced_v1.txt` + `inject_think_v1.txt`）
3. enhanced + match-cover

可直接执行：
```bash
bash run_longproc_32b.sh
```

## D. 指标解释

- `think_balanced_rate`：闭合率。
- `repetition_rate`：复读率。
- `avg_overlap_prefix_to_continuation`：复读强度。
- `format_hit_rate`：任务格式命中率。
- `longproc_avg_metrics`：LongProc evaluator 平均分。

## E. 结论规则

1. Prompt 增强有效：
- enhanced 比 baseline 的闭合率更高，且 LongProc 分数不降。

2. 插入 think 值得保留：
- 同一设置下 Branch B 的闭合不低于 A，复读不高于 A，LongProc 分数不低于 A。

3. 覆盖策略有效：
- enhanced+cover 比 enhanced 的复读指标变好，且 LongProc 分数基本不降。

## F. 抽检样例（必须做）

抽 5 个任务看全文：
- 是否出现 `<think>...</think>` 完整闭合。
- `</think>` 后是否从中断点继续。
- 格式是否符合任务要求（如 `- Step X:` / `<Route>` / code fence）。
