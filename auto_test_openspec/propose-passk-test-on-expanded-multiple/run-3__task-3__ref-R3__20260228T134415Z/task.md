# Validation Bundle (Worker) — Task 3 / Ref R3

- change-id: `propose-passk-test-on-expanded-multiple`
- run-folder: `run-3__task-3__ref-R3__20260228T134415Z`
- task: `3. 保持 CodeContests pass@k 路径兼容性 [#R3]`
- SCOPE: CLI

## What this bundle validates

This bundle performs **static, CLI-only** checks to ensure the CodeContests pass@k flow keeps its historical path contract:

- Entry script remains `scripts/evaluation/FABE/run_calculation.sh`
- Calculation module remains `src/evaluation/FABE/Calculate_passk.py`
- Default paths follow the `results/evaluation/FABE/<lang>/pass_at_k/...` pattern
- OpenSpec explicitly documents the CodeContests path contract (no breaking changes implied by docs)

## How to run

### macOS/Linux

```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T134415Z/run.sh
```

### Windows (cmd.exe)

```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T134415Z\run.bat
```

## Outputs + logs

- Logs:
  - `logs/run.txt`
- Machine-readable result:
  - `outputs/check_results.json`

## Pass/fail criteria (machine-decidable)

PASS if and only if:
- `run.sh` / `run.bat` exits with code `0`, AND
- `outputs/check_results.json` contains `"ok": true`.

