# Validation Bundle (Worker)
change-id: propose-passk-test-on-expanded-multiple
run: 2
task-id: 2
ref-id: R2
scope: CLI

## What this bundle validates
本 bundle 用“机器可判定”的方式核对 `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`
中是否形成了完整的链路描述（脚本 -> 代码 -> 输出路径），覆盖 MultiPL-E HumanEval / MBPP 的扩充数据重测流程：

- 入口脚本：
  - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
  - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
- 计算代码：`src/evaluation/FABE/Calculate_passk_multiple.py`
- 聚合代码：`src/evaluation/FABE/aggregate_results.py`
- HumanEval/MBPP 的默认输入/分片输出/聚合输出路径（含 `final_metrics.json` 契约）
- 默认 `k=4`（以 `k=4`/`pass@4` 的规范声明为准；后续允许参数化扩展）

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

