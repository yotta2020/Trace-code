## Why
当前 pass@k 评估流程已覆盖 CodeContests 与 MultiPL-E，但缺少一个明确、可复现的 OpenSpec 提案，来约束：
- MultiPL-E HumanEval / MBPP 扩充数据集的生成方式
- 扩充数据进入 SandboxFusion 单元测试后的 pass@k 计算与聚合流程
- 最终输出物路径与结构

## What Changes
- 新增一套面向 MultiPL-E HumanEval / MBPP 的扩充数据契约定义，明确字段和 11N 参考实现路径。
- 新增一套“扩充数据 -> SandboxFusion 重测 -> pass@k 聚合”的标准流程说明，绑定现有评估入口脚本。
- 新增统一输出契约：三个基准最终都产出 `results/evaluation/FABE/.../pass_at_k/final_metrics.json`，且默认 `k=4`。
- 明确失败语义：ID 缺失或沙箱失败计为 failed（不从分母排除）；分片缺失在聚合时告警并跳过。
- 当前语言范围限定为 C++、Java、Python。

## MultiPL-E 扩充数据集规范（HumanEval/MBPP）
本提案将 MultiPL-E HumanEval/MBPP 的“扩充数据”定义为一份可直接用于 SandboxFusion pass@k 重测的 JSONL 文件（UTF-8，每行一个 JSON 对象）。

### 最小字段（MUST）
- `candidates`: `string[]`，候选代码列表（每个元素为目标语言的完整代码字符串）。
- `variant_type`: `string`，扩充/扰动类型标签（如 `clean`、`dead`、`suffix`、`style_*`、`random`、`unknown`）。
- `name` / `task_id` / `problem_id`：至少包含一个题目标识键，用于与 `nuprl/MultiPL-E` 的题库对齐。

### 语言范围（本变更固定）
- 当前评估与扩充仅覆盖 `cpp`、`java`、`py`（分别对应 C++/Java/Python；与 `nuprl/MultiPL-E` 子集命名一致）。

### 参考实现路径（11N 模式）
为确保扩充流程以“参数化生成（可配置变体数与语言）”方式落地，本提案将以下现有实现作为参考路径（用于对齐脚本结构与输出模式）：
- `scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh`
- `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`

## Goals
- 明确 MultiPL-E HumanEval、MultiPL-E MBPP 扩充数据集流程，参考：
  - `scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh`
  - `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`
- 明确扩充数据输入 SandboxFusion 后的 pass@k 计算流程与入口脚本：
  - `scripts/evaluation/FABE/run_calculation.sh`
  - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
- 明确核心实现代码：
  - `src/evaluation/FABE/Calculate_passk.py`
  - `src/evaluation/FABE/Calculate_passk_multiple.py`
  - `src/evaluation/FABE/aggregate_results.py`
- 标准化最终输出：
  - `results/evaluation/FABE/.../pass_at_k/final_metrics.json`
  - `final_metrics.json` 最小字段：`benchmark`、`pass@1`、`pass@4`、`total_candidates`、`evaluated_candidates`

## Non-Goals
- 不修改 pass@k 公式本身。
- 不新增模型训练或推理策略。
- 不改造 SandboxFusion 服务端实现。

## Scope
In scope:
- 为 MultiPL-E HumanEval/MBPP 定义“扩充数据 -> SandboxFusion 重测 -> pass@k 聚合”的规范流程。
- 规范三个评估脚本与两个 pass@k 计算模块、一个聚合模块之间的职责边界。
- 规范最终指标文件位置与最小字段要求。
- 语言先限定为 C++、Java、Python（lang 编码为 `cpp`、`java`、`py`）。

Out of scope:
- 扩展到除 CodeContests、MultiPL-E HumanEval、MultiPL-E MBPP 之外的基准。
- 引入新的评估指标（如 pass@10 之外的自定义指标体系）。
- 在本提案中扩展到 C++、Java、Python 之外的语言（lang 编码超出 `cpp`/`java`/`py`）。

## Success Metrics
- 可基于扩充后的 HumanEval/MBPP 数据，执行到 `final_metrics.json` 产出。
- 三条入口脚本均可映射到明确的 pass@k 计算代码路径。
- 指标聚合输出在三个基准上保持统一路径契约（`.../pass_at_k/final_metrics.json`）。
- `final_metrics.json` 包含最小字段：`benchmark`、`pass@1`、`pass@4`、`total_candidates`、`evaluated_candidates`。

## Risks
- 扩充样本的题目标识（`name`/`task_id`/`problem_id`）与 MultiPL-E 原始题库映射失败，导致误判为未通过。
- 多分片并发下 SandboxFusion 请求失败或超时，影响通过率稳定性。
- HumanEval 与 MBPP 数据字段不一致导致聚合器处理分歧。

## Rollout
1. 完成提案与规格定义。
2. 按规格实现/对齐扩充脚本与评估脚本参数契约。
3. 基于扩充数据触发 SandboxFusion 重测并产出聚合指标。

## Rollback
- 回退到原有未扩充数据或仅单基准评估流程。
- 保留现有 CodeContests 路径不受影响。

## Open Questions
*(已解决)* 默认 `k=4`。
*(已解决)* ID 缺失与沙箱失败计为 failed，不从分母排除。
*(已解决)* 分片缺失在聚合阶段告警并跳过。
*(已解决)* 当前语言范围限定为 C++、Java、Python。
*(已解决)* 扩充倍数与采样策略采用参数化 (Parameterized)，允许通过脚本参数（如 `--num_variants`）灵活配置，不固定为 11N 模式。
