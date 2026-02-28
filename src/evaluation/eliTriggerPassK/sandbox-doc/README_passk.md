# SandboxFusion Pass@k 自动化评测

## 1. 简介
本项目用于基于 SandboxFusion 的代码 Pass@k 自动化评测。

## 2. 环境配置
使用 conda 环境：
```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
```

确保 SandboxFusion 服务已启动，例如在 `http://127.0.0.1:8081`。可以参考手册启动 docker:
```bash
docker run -d -p 8081:8080 --name sandbox-yjy-8081 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

## 3. 基本使用

### 3.1 运行 HumanEval 基线测试
测试 SandboxFusion `/run_code` 接口是否能成功执行代码:
```bash
python src/evaluation/eliTriggerPassK/run_humaneval_sandbox.py --limit 5 --sandbox_url http://127.0.0.1:8081
```

### 3.2 构造带 Trigger (触发器) 的候选集
模拟生成带有投毒触发器或错误逻辑的代码：
```bash
python src/evaluation/eliTriggerPassK/generate_poisoned_dataset.py --limit 5 --output poisoned_humaneval.jsonl
```

### 3.3 计算 Pass@k
对每个候选使用 SandboxFusion 评测，统计 Pass@1/2/4 指标：
```bash
python src/evaluation/eliTriggerPassK/evaluate_passk.py --input poisoned_humaneval.jsonl --output passk_sandboxfusion_results.jsonl --sandbox_url http://127.0.0.1:8081
```
