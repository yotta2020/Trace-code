## Why
当前 `generate_11n_dataset.py` 主要面向 11N 扩展，尚未形成一个面向 pass@k 脚本校验的“可控预测输出数据”生成流程。需要新增一个模拟模型预测输出 4N 候选的能力，用于稳定验证 pass@k 计算链路是否正确。

## Goals
- 改写 `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`，支持生成“模拟模型预测输出 4N”的测试数据。
- 用该测试数据校验 pass@k 脚本行为，至少覆盖 CodeContests、MultiPL-E HumanEval、MultiPL-E MBPP 三条计算链路。
- 定义脚本正确性的量化门槛：
  - `pass@k`（本次重点 `pass@4`）必须 `> 50%`
  - 输入正确答案计算 `pass@1` 必须 `> 90%`
  - 三条 benchmark 采用 `AND` 判定：任一不达标即整体失败

## Non-Goals
- 不改动 pass@k 公式本身。
- 不改动 SandboxFusion 服务端实现。
- 不进行模型训练或推理模型替换。

## Scope
In scope:
- 数据生成：在 `generate_11n_dataset.py` 增加 4N 候选生成模式（用于评测输入构造）。
- 新脚本：新增独立校验脚本（不复用旧入口参数），支持可配置 `N` 与扩充类型。
- 评测验证：对以下脚本产物进行一致性验证：
  - `scripts/evaluation/FABE/run_calculation.sh`
  - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
- 输出检查：`results/evaluation/FABE/.../pass_at_k/final_metrics.json`。
- 失败诊断：阈值失败时必须自动收集日志与错误样例，包含部分数据集示例与 SandboxFusion 返回体。

Out of scope:
- 新增其他 benchmark。
- 变更现有结果目录结构。

## Success Metrics
- 能生成结构合法的 4N 候选测试数据，且可被三条 pass@k 计算链路消费。
- 正确答案输入场景下：`pass@1 > 90%`。
- 4N 模拟候选场景下：`pass@4 > 50%`。
- 仅当三条链路均满足阈值（AND）才判定脚本正确。
- `pass@1` 不达标时，流程立即停止，不进入 `pass@4` 阶段。

## Impact
- 影响 `CodeContestsPlus` 数据生成与 FABE pass@k 校验流程。
- 提高 pass@k 脚本回归测试可重复性。

## Rollout / Rollback
Rollout:
- 增加 4N 生成模式。
- 增加用于“正确答案/模拟预测”两类输入的验证流程与文档。

Rollback:
- 回退到现有 11N 逻辑，不启用 4N 生成模式。

## Risks
- 4N 候选构造若分布不合理，可能导致阈值判断失真。
- 不同 benchmark 的样本键映射不一致可能导致通过率被低估。
- SandboxFusion 数据集配置不匹配（例如 MultiPL-E 子集与 dataset 参数不一致）会导致假失败。

## Open Questions
- 已确认：4N 候选优先使用 4 种变换（2 种变量名更改 + 2 种死代码插入）；若不可用，可改为全局可搜索到的 4 种投毒/触发器变换。
- 已确认：阈值判定按 benchmark 级别执行 AND gate，不采用“仅总平均达标”。
- 已确认：正确答案基线仅填 1 个正确候选用于 `pass@1` 校验。
