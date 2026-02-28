## Summary
本变更定义一个统一的 pass@k 测试流程：先扩充 MultiPL-E 数据，再把推理结果送入 SandboxFusion 重测，最后进行分片聚合并生成 `final_metrics.json`。默认 `k=4`，当前语言范围限定为 C++、Java、Python。

## Architecture
### 1) 数据扩充层（MultiPL-E HumanEval / MBPP）
- 参考脚本：
  - `scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh`
- 参考实现：
  - `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`
- 目标：形成可用于 pass@k 重测的扩充数据输入，保留题目唯一标识与候选代码集合，且仅生成 C++、Java、Python 语言样本。

### 2) 沙箱重测 + 通过统计层
- CodeContests:
  - 入口：`scripts/evaluation/FABE/run_calculation.sh`
  - 计算：`src/evaluation/FABE/Calculate_passk.py`
- MultiPL-E HumanEval / MBPP:
  - 入口：`scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - 入口：`scripts/evaluation/FABE/run_calculation_mbpp.sh`
  - 计算：`src/evaluation/FABE/Calculate_passk_multiple.py`

### 3) 聚合层
- 聚合脚本：`src/evaluation/FABE/aggregate_results.py`
- 目标：汇总分片结果并输出统一指标文件。

## Data Contract
### 1) MultiPL-E 扩充数据（HumanEval / MBPP）的输入契约（JSONL）
扩充数据（供 `src/evaluation/FABE/Calculate_passk_multiple.py` 消费）至少应满足：

- 格式：JSONL（UTF-8），每行一个 JSON 对象（对应一个题目/样本）。
- 最小字段（MUST）：
  - `candidates`: `string[]`，待 SandboxFusion 重测的候选代码列表。
  - `variant_type`: `string`，扩充/扰动类型标签（如 `clean`、`dead`、`suffix`、`style_*`、`random`、`unknown`）。
  - `name` / `task_id` / `problem_id`：至少包含一个题目标识键，用于与 HuggingFace `nuprl/MultiPL-E` 子集（如 `humaneval-cpp`）对齐。
- ID 对齐规则（与实现保持一致）：
  - 优先使用 `name`，否则使用 `task_id`，否则使用 `problem_id`；
  - 若三者均缺失或无法与题库对齐，则该条记录的候选计为 failed（并产生日志告警），但仍计入总候选数。
- 语言范围（本变更固定）：
  - 扩充与评估仅覆盖 `cpp`、`java`、`py`（分别对应 C++/Java/Python；与 `nuprl/MultiPL-E` 子集命名一致）。

示例（单行 JSON）：
```json
{"name":"HumanEval/0","variant_type":"clean","candidates":["<code_candidate_1>","<code_candidate_2>"]}
```

分片结果至少应满足：
- `passed_count`
- `failed_count`
- `total_candidates`
- `variant_type`

最终结果文件：
- `results/evaluation/FABE/.../pass_at_k/final_metrics.json`
- `final_metrics.json` 最小字段：
  - `benchmark`
  - `pass@1`
  - `pass@4`
  - `total_candidates`
  - `evaluated_candidates`

## Execution Flow
1. 生成扩充后的 MultiPL-E HumanEval/MBPP 数据（当前仅 C++、Java、Python；扩充倍数参数化）。
2. 将扩充数据对应的推理结果输入 pass@k 计算脚本。
3. `Calculate_passk*` 将每个候选重新提交到 SandboxFusion 执行单元测试并记录通过/失败情况（默认计算 `pass@4`）。
4. `aggregate_results.py` 聚合分片，输出 `final_metrics.json`。

## Failure Handling
- 单条样本 ID 对齐失败：该候选计为 failed，并在日志中记录（保留在总数中）。
- SandboxFusion 请求失败：按既有重试策略执行，重试后仍失败则计为 failed，并在日志中记录。
- 分片缺失：聚合阶段给出告警并跳过缺失分片（仅基于有效分片计算）。
