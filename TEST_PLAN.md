# Close-Think 实验方案

## 目标
- 比较 `direct continuation` 与 `inject <think>` 对闭合、连贯、重复的影响。
- 比较 `Qwen3-8B` 与 `Qwen3-32B` 在同一实验协议下的稳定性差异。
- 观察“前缀被故意改错后”模型是否更容易延续错误，或被 `<think>` 触发后修正。
- 对应 `本次todo.md`：验证 `prompt 增强` 是否有效，并验证 `匹配覆盖方案` 是否能降低重复导致的格式错乱。

## 已提供脚本
- `close/test_branch_manual.py`
  - 交互式流程：Stage1 生成中间前缀 -> 手工改错 -> Branch A/B 分叉续写。
- `close/test_close_suite.py`
  - 批量自动流程：自动 checkpoint、自动改错、自动跑 Branch A/B、自动打分。
- `close/run_test_matrix.sh`
  - 一键跑 6 组测试（1 组手工 + 5 组自动）。
- `close/tasks_math.jsonl` / `close/tasks_math_steps.jsonl`
  - 可自动判分的小样本任务集（`expected_regex`），后者更适合 Step 锚点实验。

## 每个脚本在做什么（带例子）
### 1) `test_branch_manual.py`：看“单条样本”的细节行为
- 你给一个题，例如：`Solve for x: (x-1)(x+1)=35.`
- Stage1 会先生成一段中间前缀并保存：
  - `generated_prefix.ORIG.txt`（原始）
  - `generated_prefix.EDIT_ME.txt`（你手改）
- 你在 `EDIT_ME` 里故意改错一个数字（例如把 `x=6` 改成 `x=7`）。
- Stage2 跑两条分支：
  - Branch A：直接续写
  - Branch B：先注入 `<think>` 再续写
- 结果文件：
  - `branch_A.txt`
  - `branch_B.txt`
  - `branch_summary.json`（基本统计）

这个脚本的价值：
- 适合看“模型是怎么接着写的”，是否出现断裂、重复、忘记闭合等细节问题。

### 2) `test_close_suite.py`：看“多条样本”的统计结果
- 输入是 `tasks_math.jsonl`（多条任务）。
- 每条任务都会跑 A/B 两分支，并自动算指标。
- 支持两类关键开关：
  - `--prompt-mode baseline|enhanced`：对应你的 prompt 增强实验。
  - `--apply-match-cover`：对应“重复匹配后覆盖”的实验。
- 支持你说的“Step 锚点 + 造错 + 分叉”流程：
  - `--checkpoint-mode regex --checkpoint-regex '(?i)step\\s*3'`
  - `--corrupt-mode anchor_number_shift --corrupt-anchor-regex '(?i)step\\s*3'`
  - 在同一前缀状态上分叉 A/B（A 直接续写，B 注入 think+引导词后续写）。
- 输出：
  - `*.results.jsonl`：逐题详细结果（可回看每题文本）
  - `*.summary.json`：每个模型的汇总指标
  - `summary_all_models.json`：多模型总对比（8B vs 32B）

这个脚本的价值：
- 适合做论文/汇报里的“量化对比”，不只看单个 case。

### 3) `run_test_matrix.sh`：按固定实验矩阵批量跑
- 预置 6 组测试：
  - 手工单例
  - 8B baseline
  - 8B enhanced
  - 8B enhanced + match-cover
  - 32B enhanced
  - 32B enhanced + match-cover
- 一次执行后，你能直接看到“模型大小 + prompt 增强 + match-cover”三个维度的差异。
- 默认使用 `tasks_math_steps.jsonl`，更容易匹配到 `Step 3` 锚点。

这个脚本的价值：
- 适合第一轮摸底，快速知道该把精力放在哪组参数上。

## 推荐先跑的 6 组测试
1. 手工单例验证（看细粒度行为）
2. 8B baseline prompt
3. 8B enhanced prompt
4. 8B enhanced prompt + match-cover
5. 32B enhanced prompt
6. 32B enhanced prompt + match-cover

## 输出指标
- `think_balanced_rate`：`<think>` 与 `</think>` 是否闭合。
- `repetition_rate`：续写开头是否大段重复前缀末尾。
- `avg_overlap_prefix_to_continuation`：前缀尾部与续写头部的重叠字符均值。
- `expected_hit_rate`：是否命中任务的 `expected_regex`（可当轻量正确率）。
- `avg_trimmed_chars_by_match_cover`：匹配覆盖平均裁掉多少重复字符（越高说明覆盖起作用，但需结合正确率看副作用）。

## 你提到的流程如何对应到脚本
1. 先生成到锚点：`Step 3`（regex checkpoint）。
2. 在锚点附近改几个数字：`anchor_number_shift`（破坏正确性）。
3. 从同一个前缀状态分叉：
   - A: 直接下一 token 继续采样
   - B: 先插入 `<think> + 引导词` 再继续采样
4. 若连贯有复读问题，再开启 `match-cover` 做后处理。
5. 用 `expected_hit_rate` 和重复率看真实 bench 效果。

## 结果怎么解读（例子）
假设你看到：
- 8B:
  - Branch A `think_balanced_rate=0.62`
  - Branch B `think_balanced_rate=0.84`
  - Branch B `repetition_rate` 更低
- 32B:
  - Branch A/B 都比 8B 更高，且 Branch B 最好

可以得出：
- `<think>` 注入策略有效（至少在当前任务和提示词下有效）。
- 32B 对“中途 think + 续写”更稳，说明模型能力确实是关键变量。

如果出现反例：
- Branch B 闭合更好，但 `expected_hit_rate` 更低  
说明 think 触发帮助了格式/闭合，但可能扰动了解题正确性，需要调：
- 注入文本更短
- checkpoint_delay 更后
- 温度更低

对于匹配覆盖方案：
- 若 `repetition_rate` 降低，且 `expected_hit_rate` 不下降，说明覆盖方案有效。
- 若 `repetition_rate` 降低但正确率下降，说明覆盖过强，需要调高：
  - `--cover-min-exact-overlap`
  - `--cover-fuzzy-ratio`

## 运行示例
```bash
python close/test_branch_manual.py \
  --model-path autodl-tmp/Qwen/Qwen3-8B \
  --output-dir close/manual_outputs
```

```bash
python close/test_close_suite.py \
  --model-paths autodl-tmp/Qwen/Qwen3-8B,/scratch-ssd/guoeng/huggingface/models/Qwen3-32B \
  --tasks-file close/tasks_math_steps.jsonl \
  --prompt-mode enhanced \
  --checkpoint-mode regex \
  --checkpoint-regex '(?i)step\s*3' \
  --corrupt-mode anchor_number_shift \
  --corrupt-anchor-regex '(?i)step\s*3' \
  --apply-match-cover \
  --checkpoint-delay 200 \
  --output-dir close/suite_outputs
```

```bash
MODEL_8B="autodl-tmp/Qwen/Qwen3-8B" \
MODEL_32B="/scratch-ssd/guoeng/huggingface/models/Qwen3-32B" \
bash close/run_test_matrix.sh
```
