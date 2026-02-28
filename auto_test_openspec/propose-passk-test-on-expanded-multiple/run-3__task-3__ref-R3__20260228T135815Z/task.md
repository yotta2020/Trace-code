# Validation Bundle: propose-passk-test-on-expanded-multiple / RUN #3 / Task 3 / [#R3]

- change-id: `propose-passk-test-on-expanded-multiple`
- run-folder: `auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T135815Z/`
- task: `3. 保持 CodeContests pass@k 路径兼容性 [#R3]`
- SCOPE: CLI

## What this validates (machine-decidable)

This bundle checks that CodeContests pass@k path contract is explicitly declared and remains compatible:

1) Spec contains the CodeContests path contract (inputs/shards/final output) and references the unchanged entry points:
   - `scripts/evaluation/FABE/run_calculation.sh`
   - `src/evaluation/FABE/Calculate_passk.py`
   - `results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl`
   - `results/evaluation/FABE/<lang>/pass_at_k/shards/`
   - `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

2) The referenced files exist at those paths in the repo.

## How to run

### macOS / Linux
```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T135815Z/run.sh
```

### Windows (CMD)
```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T135815Z\run.bat
```

## Outputs / Logs

- Logs:
  - `logs/worker_startup.txt` (startup ritual snapshot)
  - `logs/runner_provenance.txt` (python + cwd provenance)
  - `logs/validation.txt` (check stdout/stderr transcript)
- Outputs:
  - `outputs/report.json` (detailed check results; created by the checker)

## Pass/Fail criteria

- PASS (for Supervisor verification): `run.sh` / `run.bat` exits with code `0`.
- FAIL: non-zero exit code, and `outputs/report.json` will contain which assertions failed.

