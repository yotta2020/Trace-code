## ADDED Requirements

### Requirement: MultiPL-E 扩充数据必须可用于 SandboxFusion pass@k 重测
The system MUST define an executable input contract for expanded MultiPL-E HumanEval and MBPP data (parameterized variants, scoped to C++/Java/Python in this change) to support SandboxFusion re-testing and downstream aggregation.

#### Scenario: 扩充记录最小字段约束
- **GIVEN** 一条用于 pass@k 的扩充记录
- **WHEN** 记录被 `src/evaluation/FABE/Calculate_passk_multiple.py` 消费
- **THEN** 记录包含 `candidates` 与 `variant_type`
- **AND** 记录至少包含一个题目标识键：`name`、`task_id`、`problem_id`
- **AND** 若三个标识均缺失，则该候选计为 failed，并生成日志警告

#### Scenario: 扩充数据 JSONL 契约（可执行输入）
- **GIVEN** 一份 MultiPL-E HumanEval/MBPP 的扩充数据文件
- **WHEN** 该文件作为 `--inference_results` 传入 `src/evaluation/FABE/Calculate_passk_multiple.py`
- **THEN** 文件格式为 UTF-8 编码的 JSONL（每行一个 JSON 对象）
- **AND** 每行对象满足以下最小 schema：
  - `candidates`: `string[]`（候选代码列表；每个元素为目标语言的完整代码字符串）
  - `variant_type`: `string`（扩充/扰动类型标签，如 `clean`、`dead`、`suffix`、`style_*`、`random`、`unknown`）
  - `name` / `task_id` / `problem_id`：至少存在一个（用于对齐 HuggingFace `nuprl/MultiPL-E` 子集题库）
- **AND** 推荐（SHOULD）每条记录的 `candidates` 至少包含 1 个候选；若为空列表，则 `total_candidates=0` 且该条记录不会产生通过候选

#### Scenario: 扩充语言范围受控
- **GIVEN** MultiPL-E 扩充任务
- **WHEN** 生成 HumanEval/MBPP 扩充数据
- **THEN** 仅处理 C++、Java、Python 三种语言
- **AND** 其他语言不在本次变更范围内
- **AND** 语言编码使用：`cpp`、`java`、`py`（分别对应 C++/Java/Python；与 `nuprl/MultiPL-E` 子集命名一致）

#### Scenario: 扩充流程采用参数化生成模式
- **GIVEN** 维护者实现 MultiPL-E 扩充流程
- **WHEN** 定义生成入口与逻辑
- **THEN** 支持通过参数（如 `--num_variants`）配置扩充数量及语言类型
- **AND** 兼容现有实现路径模式：
  - `scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh`
  - `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`

### Requirement: 扩充数据重测流程必须覆盖 HumanEval 与 MBPP
The system MUST route expanded-data inference results for HumanEval and MBPP through existing scripts into SandboxFusion re-testing and produce aggregatable shard outputs.

#### Scenario: HumanEval 重测与聚合
- **GIVEN** HumanEval 扩充数据对应的 `inference_results.jsonl`
- **WHEN** 执行 `scripts/evaluation/FABE/run_calculation_humaneval.sh`
- **THEN** 通过 `src/evaluation/FABE/Calculate_passk_multiple.py` 对候选代码进行沙箱重测
- **AND** 通过 `src/evaluation/FABE/aggregate_results.py` 聚合分片
- **AND** 默认输入路径（可通过脚本环境变量覆盖）为 `results/evaluation/FABE/humaneval_<lang>/pass_at_k/inference_results.jsonl`
- **AND** 默认分片输出目录（可通过脚本环境变量覆盖）为 `results/evaluation/FABE/humaneval_<lang>/pass_at_k/shards/`
  - 每个分片产出：`shards/shard_<shard_id>.jsonl`（`Calculate_passk_multiple.py` 输出；每条记录包含 `passed_count` / `total_candidates` / `variant_type` 以供聚合）
  - 每个分片日志：`shards/shard_<shard_id>.log`（脚本重定向 stdout/stderr）
- **AND** 聚合后输出 `results/evaluation/FABE/humaneval_<lang>/pass_at_k/final_metrics.json`
- **AND** 默认 pass@k 使用 `k=4`（即默认聚合 `pass@4`；后续可通过参数化扩展支持更多 k）

#### Scenario: MBPP 重测与聚合
- **GIVEN** MBPP 扩充数据对应的 `inference_results.jsonl`
- **WHEN** 执行 `scripts/evaluation/FABE/run_calculation_mbpp.sh`
- **THEN** 通过 `src/evaluation/FABE/Calculate_passk_multiple.py` 对候选代码进行沙箱重测
- **AND** 通过 `src/evaluation/FABE/aggregate_results.py` 聚合分片
- **AND** 默认输入路径（可通过脚本环境变量覆盖）为 `results/evaluation/FABE/mbpp_<lang>/pass_at_k/inference_results.jsonl`
- **AND** 默认分片输出目录（可通过脚本环境变量覆盖）为 `results/evaluation/FABE/mbpp_<lang>/pass_at_k/shards/`
  - 每个分片产出：`shards/shard_<shard_id>.jsonl`（`Calculate_passk_multiple.py` 输出；每条记录包含 `passed_count` / `total_candidates` / `variant_type` 以供聚合）
  - 每个分片日志：`shards/shard_<shard_id>.log`（脚本重定向 stdout/stderr）
- **AND** 聚合后输出 `results/evaluation/FABE/mbpp_<lang>/pass_at_k/final_metrics.json`
- **AND** 默认 pass@k 使用 `k=4`（即默认聚合 `pass@4`；后续可通过参数化扩展支持更多 k）

### Requirement: CodeContests 与 MultiPL-E 的 pass@k 输出契约必须一致
The system MUST preserve the existing CodeContests flow and keep a consistent pass@k output contract across CodeContests, MultiPL-E HumanEval, and MultiPL-E MBPP.

#### Scenario: CodeContests 流程保持可用
- **GIVEN** CodeContests 推理结果
- **WHEN** 执行 `scripts/evaluation/FABE/run_calculation.sh`
- **THEN** 通过 `src/evaluation/FABE/Calculate_passk.py` 执行 pass@k 计算
- **AND** 通过 `src/evaluation/FABE/aggregate_results.py` 完成聚合
- **AND** 默认输入路径（与现有脚本路径模式一致）为 `results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl`
  - 其中 `<lang>` 为 CodeContests 流程使用的语言目录名（与 `Calculate_passk.py --lang` 保持一致；常见取值为 `cpp`、`java`、`py3`）
- **AND** 默认分片输出目录（与现有脚本路径模式一致）为 `results/evaluation/FABE/<lang>/pass_at_k/shards/`
  - 每个分片产出：`shards/shard_<shard_id>.jsonl`（`Calculate_passk.py` 输出）
  - 每个分片日志：`shards/shard_<shard_id>.log`（脚本重定向 stdout/stderr）
- **AND** 聚合后输出 `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

#### Scenario: 统一最终输出路径契约
- **GIVEN** 任意支持基准的分片结果
- **WHEN** 聚合完成
- **THEN** 必须产出 `results/evaluation/FABE/.../pass_at_k/final_metrics.json`

#### Scenario: 默认 k=4 且输出最小字段完整
- **GIVEN** 任意支持基准完成聚合
- **WHEN** 读取 `final_metrics.json`
- **THEN** 默认 pass@k 指标包含 `pass@4`
- **AND** 文件至少包含 `benchmark`、`pass@1`、`pass@4`、`total_candidates`、`evaluated_candidates`

#### Scenario: 沙箱失败计为 failed
- **GIVEN** 候选代码提交 SandboxFusion 执行
- **WHEN** 请求重试后仍失败
- **THEN** 该候选计为 failed
- **AND** 该候选仍计入总候选数（不从分母排除）

### Requirement: 实现阶段验证命令必须可执行且可校验字段
The system MUST provide executable CLI verification commands for three benchmark entry scripts and JSON output schema checks.

#### Scenario: 三个入口脚本可执行
- **GIVEN** 实现阶段验证流程
- **WHEN** 执行以下命令：
  - `bash scripts/evaluation/FABE/run_calculation.sh`
  - `bash scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - `bash scripts/evaluation/FABE/run_calculation_mbpp.sh`
- **THEN** 各流程可运行至结果产出阶段

#### Scenario: 最终结果文件与字段可验证
- **GIVEN** 任意支持基准评估完成
- **WHEN** 执行文件存在检查 `test -f results/evaluation/FABE/<benchmark>/pass_at_k/final_metrics.json`
- **THEN** 检查返回成功
- **AND** 可通过 CLI 检查 JSON 最小字段：`benchmark`、`pass@1`、`pass@4`、`total_candidates`、`evaluated_candidates`

## Traceability

| Ref | Requirement | Acceptance / Test |
|-----|-------------|-------------------|
| R1 | MultiPL-E 扩充数据必须可用于 SandboxFusion pass@k 重测 | Task 1 |
| R2 | 扩充数据重测流程必须覆盖 HumanEval 与 MBPP | Task 2 |
| R3 | CodeContests 与 MultiPL-E 的 pass@k 输出契约必须一致 | Task 3 |
| R4 | 提供实现阶段验证命令与结果检查 | Task 4 |
