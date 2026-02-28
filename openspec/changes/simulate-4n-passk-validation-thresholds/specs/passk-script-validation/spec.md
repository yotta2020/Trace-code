## ADDED Requirements

### Requirement: 4N Candidate Generation For Pass@k Validation
系统 MUST 支持从数据生成脚本产出用于 pass@k 校验的 4N 候选输入。

#### Scenario: Generate 4 candidates per problem
- **GIVEN** 启用 4N 模式的数据生成任务
- **WHEN** `generate_11n_dataset.py` 处理单个问题
- **THEN** 输出记录包含长度为 4 的 `candidates`
- **AND** 输出记录包含 `variant_type`
- **AND** 输出记录包含可用于 benchmark 匹配的问题标识（`name`/`task_id`/`problem_id` 之一）

#### Scenario: Default 4N augmentation set
- **GIVEN** 未显式传入扩充类型参数
- **WHEN** 生成 4N 校验数据
- **THEN** 默认使用 4 种可执行变换
- **AND** 优先组合为 2 种变量名更改 + 2 种死代码插入
- **AND** 若默认变换不可用，允许回退为全局可搜索到的 4 种投毒/触发器变换

### Requirement: Correct-Answer Baseline Must Validate Pass@1
系统 MUST 通过“输入正确答案”场景验证 pass@1 高通过率，以确认脚本基础正确性。

#### Scenario: Pass@1 gate for correct-answer input
- **GIVEN** 正确答案输入数据集
- **WHEN** 运行以下脚本并生成 `final_metrics.json`
  - `scripts/evaluation/FABE/run_calculation.sh`
  - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
- **THEN** 对每个 benchmark，`pass@1 > 90%`
- **AND** 三个 benchmark 以 AND 条件同时满足才通过
- **AND** 每题仅填 1 个正确候选用于该基线测试

### Requirement: 4N Simulation Must Validate Pass@4
系统 MUST 通过 4N 模拟候选场景验证 pass@4，确认多候选计算链路正确。

#### Scenario: Pass@4 gate for 4N simulated predictions
- **GIVEN** 4N 模拟候选输入数据集
- **WHEN** 运行 pass@k 计算脚本并生成 `final_metrics.json`
- **THEN** 对每个 benchmark，`pass@4 > 50%`
- **AND** 三个 benchmark 以 AND 条件同时满足才通过

### Requirement: Validation Gate Must Be Enforced
系统 MUST 将阈值作为脚本正确性的硬门槛。

#### Scenario: Fail fast on threshold breach
- **GIVEN** 任意 benchmark 结果未达到阈值
- **WHEN** 校验流程读取 `final_metrics.json`
- **THEN** 标记该次脚本校验为失败
- **AND** 输出未达标指标值与结果文件路径

#### Scenario: Stop pass@4 when pass@1 baseline fails
- **GIVEN** `pass@1` 阶段存在任一 benchmark 未达标
- **WHEN** 总控流程执行 gate
- **THEN** 立即停止后续 `pass@4` 评测
- **AND** 输出失败原因与排障指引

### Requirement: Failure Debug Logs Must Be Actionable
系统 MUST 在阈值失败时输出可用于定位问题的数据与错误上下文。

#### Scenario: Emit dataset samples and sandbox error payloads
- **GIVEN** gate 校验失败
- **WHEN** 生成失败报告
- **THEN** 报告包含数据集样例节选（部分记录）
- **AND** 包含 SandboxFusion 返回错误内容
- **AND** 包含初步归因标签（`dataset_issue` / `script_issue` / `sandbox_issue`）

## Traceability

| Ref | Requirement | Design decision | Acceptance / Test |
|-----|-------------|-----------------|-------------------|
| R1 | 4N Candidate Generation For Pass@k Validation | 在 `generate_11n_dataset.py` 增加 4N 开关模式 | Task 1 |
| R2 | Correct-Answer Baseline Must Validate Pass@1 | 先 pass@1 再 pass@4 的阶段 gate | Task 3 |
| R3 | 4N Simulation Must Validate Pass@4 | 以 4N 场景校验多候选计算 | Task 4 |
| R4 | Validation Gate Must Be Enforced | 引入统一阈值 AND gate 逻辑 | Task 5 |
| R5 | Failure Debug Logs Must Be Actionable | 失败必须输出样例与错误返回体 | Task 5 |
