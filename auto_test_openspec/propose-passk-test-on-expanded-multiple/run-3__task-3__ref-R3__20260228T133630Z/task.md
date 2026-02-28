# Validation Bundle: propose-passk-test-on-expanded-multiple (RUN #3, Task 3, Ref R3)

## Scope
SCOPE: CLI

This bundle validates **CodeContests pass@k path compatibility** by asserting:
- The canonical OpenSpec for this change explicitly declares the CodeContests entry script + calculation code paths.
- The existing CodeContests entry script still references the expected calculation module and output path contract.

No SandboxFusion/Docker execution is performed in this bundle (static, machine-decidable checks only).

## What this checks

### Required files exist (no breaking path changes)
- `scripts/evaluation/FABE/run_calculation.sh`
- `src/evaluation/FABE/Calculate_passk.py`
- `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`

### Spec explicitly declares CodeContests paths
In `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`, this bundle checks the presence of:
- `scripts/evaluation/FABE/run_calculation.sh`
- `src/evaluation/FABE/Calculate_passk.py`
- `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

### CodeContests entry script still points to the expected code and output contract
In `scripts/evaluation/FABE/run_calculation.sh`, this bundle checks the presence of:
- `src/evaluation/FABE/Calculate_passk.py`
- `results/evaluation/FABE/${TARGET_LANG}/pass_at_k/inference_results.jsonl`
- `pass_at_k/final_metrics.json`

## How to run

### macOS/Linux
From anywhere:
```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T133630Z/run.sh
```

### Windows (cmd.exe)
From anywhere:
```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T133630Z\run.bat
```

## Outputs / Logs
- Writes logs under:
  - `auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T133630Z/logs/`
    - `run_stdout.txt`
    - `run_stderr.txt` (macOS/Linux)

## Machine-decidable pass/fail criteria
- PASS: `run.sh` / `run.bat` exits with code `0`.
- FAIL: non-zero exit code (missing files and/or missing required path declarations).

