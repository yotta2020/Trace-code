## Summary
该变更引入“4N 模拟预测输出”模式，复用现有 pass@k 执行链路完成脚本正确性验证。验证分两阶段：
1) 正确答案输入，验证 `pass@1 > 90%`；
2) 4N 模拟候选输入，验证 `pass@4 > 50%`。
门禁规则为 benchmark 级别 `AND`：CodeContests、HumanEval、MBPP 三者均达标才算通过。

## Architecture
- 数据构造层：
  - 目标文件：`src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`
  - 新增 4N 模式输出（每题 4 candidates）
- 校验控制层：
  - 新增独立脚本（建议路径：`scripts/evaluation/FABE/run_passk_gate_validation.sh`）
  - 参数：`--n`、`--augment_types`、`--mode`、`--stop_on_pass1_fail`
- 执行层：
  - CodeContests: `scripts/evaluation/FABE/run_calculation.sh` -> `Calculate_passk.py`
  - HumanEval/MBPP: `run_calculation_humaneval.sh` / `run_calculation_mbpp.sh` -> `Calculate_passk_multiple.py`
- 聚合层：
  - `src/evaluation/FABE/aggregate_results.py`

## Data Model
每条评测记录至少包含：
- `candidates`: 长度为 4（4N 模式）
- `variant_type`
- 问题标识：`name` 或 `task_id` 或 `problem_id`

聚合前新增字段：
- `passed_count`
- `total_candidates`

最终指标文件：
- `results/evaluation/FABE/.../pass_at_k/final_metrics.json`

失败诊断文件（新增）：
- `results/evaluation/FABE/<benchmark>/pass_at_k/gate_debug.json`
  - 字段至少包含：`benchmark`、`threshold`、`actual`、`sample_records`、`sandbox_errors`

## Interfaces
- 不新增新入口脚本，继续复用：
  - `run_calculation.sh`
  - `run_calculation_humaneval.sh`
  - `run_calculation_mbpp.sh`
- 通过输入数据内容区分“正确答案场景”与“4N 模拟场景”。
- 新增总控脚本负责阶段编排：
  - Phase 1: 正确答案基线（只填一个正确候选）并检查 `pass@1 > 90%`
  - Phase 2: 仅当 Phase 1 全通过时才执行 4N 并检查 `pass@4 > 50%`

MultiPL-E 对齐参考：
- `https://bytedance.github.io/SandboxFusion/docs/docs/how-to/use-dataset/multiple-humaneval`

## Key Flows
1. 生成正确答案输入集，执行 pass@k，读取 `final_metrics.json`，验证 `pass@1 > 90%`。
2. 生成 4N 模拟候选输入集，执行 pass@k，读取 `final_metrics.json`，验证 `pass@4 > 50%`。
3. 任一阈值不满足则判定脚本校验失败。
4. `pass@1` 任一 benchmark 不达标，直接结束流程并输出排障日志，不执行 `pass@4`。

## Failure Handling
- 样本无法映射到 benchmark 键时，记录告警并纳入失败统计。
- 某分片执行失败时，保留 shard 日志并中止阈值判定。
- 阈值失败时必须输出可复现排障信息：
  - 数据集样例（脱敏后节选）
  - SandboxFusion 返回体（错误码/报错片段）
  - 初步归因标签：`dataset_issue` / `script_issue` / `sandbox_issue`

## Observability
- 分片日志：`.../pass_at_k/shards/shard_*.log`
- 分片结果：`.../pass_at_k/shards/shard_*.jsonl`
- 汇总指标：`.../pass_at_k/final_metrics.json`
- Gate 调试：`.../pass_at_k/gate_debug.json`

## Migration / Compatibility
- 默认不影响当前 11N 数据流程。
- 4N 模式以显式参数开关启用。

## Alternatives Considered
- 单独新建 4N 生成脚本：
  - 未选用，优先在 `generate_11n_dataset.py` 内扩展，减少入口分散。

## Decision Log
- 决策：脚本正确性以阈值方式判定（`pass@1 > 90%`, `pass@4 > 50%`）。
- 决策：先复用现有 FABE 评测脚本，不新增并发执行框架。
- 决策：阈值按三个 benchmark 的 AND 条件判定。
- 决策：`pass@1` 阶段失败即短路，不进入 `pass@4`。
- 决策：4N 变换优先组合为 2 种变量名变换 + 2 种死代码插入。

## Open Questions
- Open Question: 4N 默认变换集合是否完全固定，还是允许通过参数替换为其他触发器集合。
