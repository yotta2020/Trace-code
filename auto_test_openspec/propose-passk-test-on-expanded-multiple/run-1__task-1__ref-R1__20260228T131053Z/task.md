# Validation Bundle: R1 MultiPL-E 扩充数据集规范（HumanEval/MBPP）

- change-id: `propose-passk-test-on-expanded-multiple`
- run: `RUN #1`
- task-id: `1`
- ref-id: `R1`
- SCOPE: `CLI`

## What this bundle validates
This bundle performs CLI checks to confirm that the dataset-spec definitions for MultiPL-E expanded data (HumanEval/MBPP) are explicitly documented and consistent across:

- `openspec/changes/propose-passk-test-on-expanded-multiple/proposal.md`
- `openspec/changes/propose-passk-test-on-expanded-multiple/design.md`
- `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`

## How to run

### macOS / Linux
From any directory:
```bash
bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-1__task-1__ref-R1__20260228T131053Z/run.sh
```

### Windows
From any directory:
```bat
auto_test_openspec\propose-passk-test-on-expanded-multiple\run-1__task-1__ref-R1__20260228T131053Z\run.bat
```

## Outputs
- Logs:
  - `logs/run.txt` (command transcript)
  - `logs/validate_stdout.txt` (validator stdout/stderr capture)
- Machine-readable report:
  - `outputs/validation_report.json`

## Pass/Fail criteria (machine-decidable)
PASS if and only if:
- `run.sh` / `run.bat` exits with code `0`, and
- validator prints `ALL_CHECKS_PASS=1`.

FAIL otherwise (non-zero exit code, missing required strings/sections, or validator prints `ALL_CHECKS_PASS=0`).

## Provenance / assumptions
- Validation is rule-based (string/structure assertions) because R1 `TEST:` is manual consistency checking of docs.
- No external dependencies are installed; validator uses Python 3 stdlib only.

