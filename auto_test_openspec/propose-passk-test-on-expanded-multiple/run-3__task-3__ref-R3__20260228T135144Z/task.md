# Validation Bundle — propose-passk-test-on-expanded-multiple — RUN #3 — Task 3 — Ref R3

- change-id: `propose-passk-test-on-expanded-multiple`
- run-folder: `auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T135144Z/`
- task: `3. 保持 CodeContests pass@k 路径兼容性 [#R3]`
- SCOPE: `CLI`

## What this bundle checks (machine-decidable)

This bundle verifies that the spec + proposal + design **explicitly** preserve the existing CodeContests pass@k entrypoints and output path contract:

- CodeContests entry script path is preserved: `scripts/evaluation/FABE/run_calculation.sh`
- CodeContests pass@k calculation code path is preserved: `src/evaluation/FABE/Calculate_passk.py`
- CodeContests output contract uses: `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`
- The general unified contract includes: `results/evaluation/FABE/.../pass_at_k/final_metrics.json`

## How to run

### macOS / Linux
```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T135144Z/run.sh
```

### Windows (cmd.exe)
```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T135144Z\run.bat
```

## Pass / Fail criteria (machine-decidable)

- **PASS**: `run.sh` / `run.bat` exits with code `0`.
- **FAIL**: non-zero exit code, and console output will list missing files and/or missing required strings in docs.

## Outputs and logs

- Logs:
  - `logs/worker_startup.txt` (startup snapshot; created by worker)
  - `logs/run.txt` (command transcript for this bundle run; appended per execution)
- No `inputs/expected/outputs` are required for this manual-spec compatibility check.

