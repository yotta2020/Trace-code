# 防御模型 Pass@k 评测方案（SandboxFusion 版）

## 1. 目标

基于 SandboxFusion 的执行评测

- 评测对象：代码后门防御模型（Baseline / TRACE / FABE）输出的清洗代码
- 核心指标：Pass@k（功能保持率）

- **核心内容**:
  - **背景与目标**: 衡量防御模型清洗投毒代码后的功能正确性 (Pass@k)。
  - **核心评测流程**:
    1. **环境准备**: 使用 Docker 启动 SandboxFusion 服务 `docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609`。
    2. **离线推理**: 将带 Trigger 的“投毒代码”输入防御模型，生成多个去毒后的代码候选 (Candidates)，保存为 JSON 格式。
    3. **格式适配与提交**: 将生成的候选代码封装为 SandboxFusion 要求的请求体：`{"dataset": "数据集名", "id": "题目ID", "completion": "生成的代码", "config": {}}`。
    4. **Pass@k 计算**: 通过 POST 请求 `/submit` 接口提交通信，解析响应中的 `"accepted": true` 状态，结合生成的 n 个候选计算 Pass@k。
  - **与 BigCode 的优势对比**:
    - 统一的 RESTful API，解耦评测框架与生成框架。
    - 开箱即用的多语言 Docker 执行沙箱，无需复杂的 Task 继承与指标封装。
  - **实施步骤**:
    - 开发推理脚本生成多候选 JSONL。
    - 开发 SandboxFusion 客户端脚本，并发提交代码候选。
    - 聚合统计最终的 Pass@1/2/4。


## 2. 评测总流程

```
构建评测集(含 test_cases) → 防御模型推理生成多候选 → SandboxFusion 执行测试 → 统计 Pass@k
```

输入输出约定：
- 输入样本：`poisoned_code` + `test_cases` + `variant_type` 等元数据
- 推理输出：每题 `n` 个候选代码（默认 `n=4`）
- 评测输出：
  - 单题：`pass@1/pass@2/pass@4`、成功候选数、错误类型
  - 聚合：按 `variant_type`、语言、模型分组的 Pass@k

## 3. 数据准备

### 3.1 评测集构建
优先复用现有 ccplus 管线，确保每条记录保留：
- `problem_id`
- `variant_id` / `variant_type`
- `language`
- `poisoned_code`
- `test_cases`（stdin/stdout）

建议基于已有数据脚本组织样本，不再转换为 BigCode assertion-only 格式，直接对接 SandboxFusion 的运行接口。

### 3.2 样本抽样策略
建议固定评测切片，保证可复现：
- 变体覆盖：至少包含 DeadCode / Suffix / Style / Mixed / Rename
- 每变体样本数固定
- 固定随机种子与清单文件

## 4. 防御推理设置

推理阶段输出多候选，用于 Pass@k 估计：
- 候选数：`n=4`（可扩到 8）
- 采样：`temperature=0.7`, `top_p=0.95`
- 输出字段：
  - `candidates`: `List[str]`
  - `metadata`: 模型名、checkpoint、时间戳、推理参数

落盘格式建议为 JSONL：每行一题，包含原样本字段 + `candidates`。

## 5. SandboxFusion 执行评测

### 5.1 执行原则
对每题每个候选代码执行全部 `test_cases`：
- 全部通过记为 `success=1`
- 任一失败记为 `success=0`
- 记录失败原因（编译错误 / 运行超时 / 输出不一致 / 运行异常）

### 5.2 并发与稳定性
- 使用多个 SandboxFusion 实例轮询（与 ccplus 既有方式一致）
- 设置统一超时与资源限制
- 对网络抖动或临时错误进行有限重试（例如 2 次）

### 5.3 Pass@k 计算
对每题统计：
- `n`：候选总数
- `c`：成功候选数

使用标准估计式：

- 当 `n - c < k` 时，`pass@k = 1`
- 否则：`pass@k = 1 - C(n-c, k) / C(n, k)`

最终对全体样本取均值，并按分组输出（模型/语言/variant_type）。

## 6. 推荐产物结构

- 原始推理结果：`results/.../inference_results.jsonl`
- 执行明细：`results/.../sandboxfusion_exec_details.jsonl`
- 指标汇总：`results/.../passk_summary.json`
- 分组报表：`results/.../passk_by_variant.json`

## 7. 最小可运行命令骨架（示例）

```bash
# 1) 生成防御候选
python src/evaluation/FABE/evaluation.py \
  --input_path data/.../eval_5n_with_tc.jsonl \
  --output_path results/.../inference_results.jsonl

# 2) 调 SandboxFusion 执行并产出每候选 success/fail
python scripts/evaluation/run_sandboxfusion_passk.py \
  --input results/.../inference_results.jsonl \
  --sandbox_url "http://127.0.0.1:8000/run_code" \
  --timeout 10 \
  --output results/.../sandboxfusion_exec_details.jsonl

# 3) 计算并聚合 Pass@k
python scripts/evaluation/aggregate_sandboxfusion_passk.py \
  --exec_details results/.../sandboxfusion_exec_details.jsonl \
  --k 1 2 4 \
  --output results/.../passk_summary.json
```

## 8. 风险与控制

- 语言运行环境差异导致结果波动：固定 SandboxFusion 镜像版本
- 并发执行导致偶发失败：重试并记录 `flaky` 标记
- 候选数不足影响对比：统一 `n` 并在报告注明

## 9. 验收标准

- 已完全移除 BigCode Harness 依赖路径
- 可基于同一输入数据稳定复现 Pass@k 结果
- 至少输出：总体 Pass@1/2/4 + 按 `variant_type` 分组结果




## 验证与测试
- 启动 SandboxFusion 容器确保 API `/submit` 正常响应。
- 构造单条数据的评测请求，确认返回的 `accepted` 字段准确反映代码测试用例的通过情况。
- 对比新脚本计算的 Pass@k 与之前 `Calculate_passk.py` 的计算结果是否一致。