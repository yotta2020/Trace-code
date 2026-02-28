# Validation Bundle: propose-passk-test-on-expanded-multiple — RUN #3 (Task 3 / Ref R3)

- change-id: `propose-passk-test-on-expanded-multiple`
- run-folder: `run-3__task-3__ref-R3__20260228T133557Z`
- task: `3. 保持 CodeContests pass@k 路径兼容性 [#R3]`
- SCOPE: CLI

## What this bundle validates

This bundle performs **textual/contract checks** (no Docker / no SandboxFusion execution):

1) The canonical spec explicitly declares the CodeContests pass@k path contract:
   - input: `results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl`
   - shard dir: `results/evaluation/FABE/<lang>/pass_at_k/shards/`
   - final output: `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

2) The CodeContests entrypoints are still anchored to the existing paths:
   - `scripts/evaluation/FABE/run_calculation.sh`
   - `src/evaluation/FABE/Calculate_passk.py`
   - `src/evaluation/FABE/aggregate_results.py`

## How to run

macOS/Linux:
```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T133557Z/run.sh
```

Windows (cmd.exe):
```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T133557Z\run.bat
```

## Outputs & logs

- Logs:
  - `logs/worker_startup.txt` (startup snapshot)
  - `logs/run.txt` (runner transcript)
- Outputs:
  - `outputs/` is created (no files expected for this task)

## Machine-decidable pass/fail criteria

PASS if:
- `run.sh` / `run.bat` exits with code `0`
- `logs/run.txt` contains `OK: R3 textual path-compatibility checks passed.`

FAIL if:
- exit code is non-zero, or
- the validator reports missing required substrings/paths.
