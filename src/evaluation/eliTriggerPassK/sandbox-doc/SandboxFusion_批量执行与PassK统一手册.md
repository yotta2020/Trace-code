# SandboxFusion 批量执行与 Pass@k 统一手册

本文整合以下文档内容并统一为一份可执行手册：
- 一键启动 SandboxFusion
- CodeContestsPlus v1/v2 运行流程
- Docker 跑满与分片执行
- 防御模型 Pass@k 评测方案

适用目录：`/home/nfs/share-yjy/dachuang2025/users/d2022-yjy/eliTriggerPassK`

## 1. 目标与产出

目标：用 SandboxFusion 批量执行单元测试，支持数据处理与防御模型评测，最终得到 Pass@k。

核心产出：
- 分片执行结果（jsonl/log）
- 聚合评测结果（`final_metrics.json`）
- 按任务/变体统计的 Pass@1/2/4

## 2. 关键脚本定位

数据处理与并发执行：]
- `ccplus-muti-docker.sh`
- `CausalCode-Defender-codex/scripts/data_preprocessing/CodeContestsPlus/run_selection.sh`
- `CausalCode-Defender-codex/scripts/data_preprocessing/CodeContestsPlus/run_ist_clean.sh`
- `CausalCode-Defender-codex/src/data_preprocessing/CodeContestsPlus/ist_clean.py`

评测与 Pass@k：
- `CausalCode-Defender-codex/scripts/evaluation/FABE/run_calculation.sh`
- `CausalCode-Defender-codex/src/evaluation/FABE/Calculate_passk.py`
- `CausalCode-Defender-codex/src/evaluation/FABE/aggregate_results.py`

## 3. 环境准备

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
```

Sandbox 镜像（当前脚本使用）：
```bash
vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

官方文档：
- https://bytedance.github.io/SandboxFusion/docs/docs/get-started

## 4. 一键批量启动 Sandbox 容器

### 4.1 推荐：按你需求固定端口段

你之前要求端口统一到 `12408-12412`。如果要“启动 4 个”，建议用 `12408-12411`：

```bash
for p in $(seq 12408 12411); do
  docker rm -f sandbox-yjy-${p} >/dev/null 2>&1 || true
  docker run -d -p ${p}:8080 --name sandbox-yjy-${p} \
    vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
 done
```

校验：
```bash
docker ps --format '{{.Names}} {{.Ports}}' | grep '^sandbox-yjy-' | sort
docker ps --format '{{.Names}}' | grep '^sandbox-yjy-' | wc -l
```

清理：
```bash
docker ps -a --format '{{.Names}}' | grep '^sandbox-yjy-' | xargs -r docker rm -f
```

## 5. CodeContestsPlus 数据流程（v1 -> v2）

## 5.1 v1：每题一答构建与切分

小批量 smoke（建议先跑）和全量 nohup 的完整参数策略，沿用：
- `ccplus_1ans_v1_build_and_split.py`
- 关键参数：`--langs`、`--sandbox`、`--workers`、`--compile_timeout`、`--request_retries`

实践建议：
1. 先按语言分机跑（cpp/java/py3 各一台）
2. 先 smoke 再全量
3. 全量日志只看关键行：`PROGRESS|OK|ERROR|Traceback`

## 5.2 v2：IST 转换 + 语法检查 + 单测过滤

执行入口：
- `CausalCode-Defender-codex/scripts/data_preprocessing/CodeContestsPlus/run_ist_clean.sh`
- 核心逻辑：`CausalCode-Defender-codex/src/data_preprocessing/CodeContestsPlus/ist_clean.py`

说明：
- 脚本会并发起多个 Sandbox 容器并按分片执行
- `ist_clean.py` 会调用 SandboxFusion `/run_code` 验证 test cases
- 失败样本会记录失败原因（编译错误/超时/WA/请求错误）

## 6. 批量分片执行（旧流水线）

可直接调用：
```bash
eliTriggerPassK/ccplus-muti-docker.sh
```

该脚本包含：
- 批量起容器
- 自动探测容器端口
- 多分片并发执行 Python 任务
- 任务退出时自动清理后台进程和容器

## 7. 防御模型 Pass@k 评测

## 7.1 流程

```text
构建评测集(含 test_cases)
-> 防御模型生成多候选
-> SandboxFusion 执行测试
-> 统计 Pass@1/2/4
```

## 7.2 推荐执行入口

```bash
bash CausalCode-Defender-codex/scripts/evaluation/FABE/run_calculation.sh
```

该脚本会：
1. 启动并发沙箱容器
2. 按 shard 调用 `Calculate_passk.py`
3. 用 `aggregate_results.py` 聚合最终结果

## 7.3 Pass@k 公式

- `n`：候选总数
- `c`：通过候选数
- 当 `n-c < k`：`pass@k = 1`
- 否则：`pass@k = 1 - C(n-c, k)/C(n, k)`

## 8. 结果目录建议

- 推理候选：`results/.../inference_results.jsonl`
- 分片执行：`results/.../pass_at_k/shards/shard_*.jsonl`
- 最终指标：`results/.../pass_at_k/final_metrics.json`

## 9. 常见问题



Q2：如何判断任务正常推进？
- 查看 shard 日志中的 `PROGRESS` 和末尾聚合是否成功产出 `final_metrics.json`。

Q3：如何避免跑满后结果不稳定？
- 固定镜像版本、固定超时参数、保留重试并记录失败明细。

## 重点文件 
- 批量启动 SandboxFusion:
  - eliTriggerPassK/ccplus-muti-docker.sh
  - CausalCode-Defender-codex/scripts/data_preprocessing/CodeContestsPlus/run_ist_clean.sh
  - CausalCode-Defender-codex/scripts/evaluation/FABE/run_calculation.sh
- 执行 SandboxFusion（调用 /run_code 单测）:
  - CausalCode-Defender-codex/src/data_preprocessing/CodeContestsPlus/ist_clean.py
- 使用方法 md:
  - CausalCodeDefense2main/doc/00.private-data/下一步/整理CodeContests数据集开启训练/一键启动SandboxFusion脚本.md
  - CausalCodeDefense2main/doc/00.private-data/下一步/整理CodeContests数据集开启训练/ccplus_1ans_v1v2_运行手册.md
  - CausalCodeDefense2main/doc/02.实验相关文档/01.数据/单元测试docker跑满.md
  - Trace-doc/03_Task/待办收集箱/plan-defense-passk-sandboxfusion.md
- 单元测试 + pass@k:
  - CausalCode-Defender-codex/src/evaluation/FABE/Calculate_passk.py
  - CausalCode-Defender-codex/src/evaluation/FABE/aggregate_results.py



