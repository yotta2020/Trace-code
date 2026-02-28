# Validation Bundle (Worker)
change-id: propose-passk-test-on-expanded-multiple
run: 3
task-id: 3
ref-id: R3
scope: CLI

## What this bundle validates
本 bundle 用“机器可判定”的方式核对 CodeContests 的 pass@k 路径契约保持兼容，且在规格中被显式声明（避免被 MultiPL-E 扩充流程变更破坏）：

- 入口脚本不变：`scripts/evaluation/FABE/run_calculation.sh`
- 计算代码不变：`src/evaluation/FABE/Calculate_passk.py`
- 聚合代码：`src/evaluation/FABE/aggregate_results.py`
- 默认 I/O 路径模式（与现有脚本一致）：
  - `results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl`
  - `results/evaluation/FABE/<lang>/pass_at_k/shards/`
  - `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`
- 统一最终输出契约：`.../pass_at_k/final_metrics.json`

检查对象：
- `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`
- `scripts/evaluation/FABE/run_calculation.sh`

## How to run

### macOS/Linux
```bash
bash run.sh
```

### Windows (cmd.exe)
```bat
run.bat
```

## Outputs
- `logs/worker_startup.txt`: Worker 启动快照（包含 GIT_BASE、git log、观察摘要）
- `logs/run_stdout.txt`: 本次检查的 stdout/stderr 记录
- `logs/provenance.txt`: Python 解释器与版本信息（runner 生成）

## Pass/Fail criteria (machine-decidable)
PASS 当且仅当：
- `run.sh` / `run.bat` 退出码为 0
- 且 `logs/run_stdout.txt`（同时也是控制台输出）包含 `ALL CHECKS PASSED`

FAIL 当任一条件不满足（退出码非 0 或断言失败）。

