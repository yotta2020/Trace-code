# Validation Bundle (Worker) — Task 3 [#R3]

- change-id: `propose-passk-test-on-expanded-multiple`
- run: `RUN #3`
- task-id: `3`
- ref-id: `R3`
- SCOPE: `CLI`

## Goal

以“机器可判定”的方式核对 `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md` 已显式声明 CodeContests 的 pass@k 入口与路径契约，并确认现有 CodeContests 入口脚本仍引用既有计算代码路径（不被 MultiPL-E multiple 流程破坏）。

## How to run

### macOS / Linux

```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T134221Z/run.sh
```

### Windows (cmd.exe)

```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T134221Z\run.bat
```

## What this bundle checks

1) `spec.md` must explicitly reference the CodeContests entry + code paths:
   - `scripts/evaluation/FABE/run_calculation.sh`
   - `src/evaluation/FABE/Calculate_passk.py`
   - `src/evaluation/FABE/aggregate_results.py`

2) `spec.md` must explicitly declare the CodeContests I/O path contract:
   - `results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl`
   - `results/evaluation/FABE/<lang>/pass_at_k/shards/`
   - `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

3) `scripts/evaluation/FABE/run_calculation.sh` must still reference the existing CodeContests computation paths and output contract:
   - calls `src/evaluation/FABE/Calculate_passk.py`
   - calls `src/evaluation/FABE/aggregate_results.py`
   - uses `pass_at_k/inference_results.jsonl`, `pass_at_k/shards`, `pass_at_k/final_metrics.json`

## Outputs

- `logs/provenance.txt`: python interpreter path/version
- `logs/run_stdout.txt`: test stdout/stderr transcript

## Pass / Fail (machine-decidable)

- PASS iff `run.sh` / `run.bat` exits with code `0` and prints `ALL CHECKS PASSED`.
- FAIL otherwise (non-zero exit code and/or missing required references).

