# Validation Bundle (Worker)

- change-id: `propose-passk-test-on-expanded-multiple`
- run-folder: `auto_test_openspec/propose-passk-test-on-expanded-multiple/run-4__task-4__ref-R4__20260228T140619Z/`
- task-id: `4`
- ref-id: `R4`
- SCOPE: `CLI`

## What this validates (ACCEPT/TEST for [#R4])

This bundle executes these three entry commands (as required by `openspec/changes/propose-passk-test-on-expanded-multiple/tasks.md` task 4):

1) `bash scripts/evaluation/FABE/run_calculation.sh`
2) `bash scripts/evaluation/FABE/run_calculation_humaneval.sh`
3) `bash scripts/evaluation/FABE/run_calculation_mbpp.sh`

Then it performs result checks:

- File existence checks (examples):
  - `test -f results/evaluation/FABE/<benchmark>/pass_at_k/final_metrics.json`
- JSON field checks (required keys):
  - `benchmark`, `pass@1`, `pass@4`, `total_candidates`, `evaluated_candidates`

## Preconditions / Assumptions

- The host can run Docker (the three upstream scripts start sandbox containers).
- Required inference inputs already exist under `results/evaluation/FABE/**/pass_at_k/inference_results.jsonl` per the upstream scripts’ defaults (or are provided via env overrides used by those scripts).
- Python is available to run the JSON field-check one-liner.

## How to run

macOS/Linux:

```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-4__task-4__ref-R4__20260228T140619Z/run.sh
```

Windows (requires `bash`, e.g. Git Bash or WSL in PATH):

```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-4__task-4__ref-R4__20260228T140619Z\run.bat
```

## Outputs (this run-folder)

- `logs/`
  - `worker_startup.txt` (startup snapshot)
  - `provenance.txt` (env + tool versions)
  - `run_calculation*.txt` (command transcripts)
  - `checks.txt` (file/key assertions)
- `outputs/`
  - `final_metrics_*.json` (copied from `results/evaluation/.../final_metrics.json` for auditability)

## Machine-decidable PASS/FAIL criteria

PASS iff ALL of the following are true:

1) `run.sh` / `run.bat` exits with code `0`.
2) For each executed upstream script, the corresponding `results/evaluation/FABE/**/pass_at_k/final_metrics.json` exists (as checked by the script).
3) Each copied `outputs/final_metrics_*.json` contains all required keys:
   - `benchmark`, `pass@1`, `pass@4`, `total_candidates`, `evaluated_candidates`
4) `logs/run_calculation*.txt` do not contain the substring `Traceback` (simple guard for Python crashes).

FAIL otherwise (non-zero exit code).

