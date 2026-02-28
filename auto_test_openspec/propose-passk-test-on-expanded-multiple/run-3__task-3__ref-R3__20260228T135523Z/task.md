# Validation Bundle (Worker) — Task 3 [#R3]

- change-id: `propose-passk-test-on-expanded-multiple`
- run: `RUN #3`
- task-id: `3`
- ref-id: `R3`
- SCOPE: `CLI`

## Goal

以“机器可判定”的方式核对：

1) `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md` 已显式声明 CodeContests 的 pass@k 入口与路径契约（不被 MultiPL-E multiple 流程破坏）。
2) `scripts/evaluation/FABE/run_calculation.sh` 仍引用既有 CodeContests 计算脚本 `src/evaluation/FABE/Calculate_passk.py`，并保持输出路径契约为 `.../pass_at_k/final_metrics.json`。

## How to run

### Linux/macOS (bash)
```bash
bash run.sh
```

### Windows (cmd.exe)
```bat
run.bat
```

## Outputs

- `outputs/check_r3_codecontests_paths.json`: 机器可读的检查结果（`ok` / `missing` / `details`）
- `logs/provenance.txt`: 环境信息（python/git/uv 等）
- `logs/run_stdout.txt`, `logs/run_stderr.txt`: 执行日志

## Pass/fail criteria (machine-decidable)

- **PASS (for this bundle):** `run.sh` / `run.bat` 退出码为 `0`，且 `outputs/check_r3_codecontests_paths.json` 中 `"ok": true`。
- **FAIL:** 非 0 退出码，或输出 JSON 中 `"ok": false`（并给出 `"missing"` 列表）。

Notes:
- 本 bundle 不会启动 SandboxFusion 或执行真实评测；它将 Task 3 的 `TEST: Manual` 要求转化为可复现、可自动化执行的文档/路径链路校验。
- Supervisor 是最终验证者；Worker 仅提供可执行的验证资产与日志。

