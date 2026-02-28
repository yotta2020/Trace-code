# Validation Bundle: propose-passk-test-on-expanded-multiple (RUN #3 / Task 3 / Ref R3)

- change-id: `propose-passk-test-on-expanded-multiple`
- run#: `3`
- task-id: `3`
- ref-id: `R3`
- SCOPE: `CLI`

## What this validates (R3)

This bundle performs a **path compatibility** check for the CodeContests pass@k flow:

1) `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md` explicitly declares:
   - `scripts/evaluation/FABE/run_calculation.sh`
   - `src/evaluation/FABE/Calculate_passk.py`
   - output contract: `results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json`

2) The referenced entry script and calculation code paths exist in the repo, and the CodeContests entry script still uses the `pass_at_k/` path pattern for:
   - input: `.../pass_at_k/inference_results.jsonl`
   - report: `.../pass_at_k/final_metrics.json`

## How to run

macOS/Linux:

```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T133916Z/run.sh
```

Windows (cmd.exe):

```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-3__task-3__ref-R3__20260228T133916Z\run.bat
```

## Outputs / logs

- Logs:
  - `logs/worker_startup.txt` (startup snapshot; pre-generated)
  - `logs/provenance.txt` (python path/version)
  - `logs/check_r3_stdout.txt` (checker stdout/stderr)
- Outputs:
  - `outputs/r3_check_report.json` (machine-readable report)

## Pass/fail criteria (machine-decidable)

PASS if and only if:
- `run.sh` / `run.bat` exits with code `0`, AND
- `outputs/r3_check_report.json` is created, AND
- the report has `"ok": true`.

FAIL otherwise (non-zero exit code or `"ok": false`).

