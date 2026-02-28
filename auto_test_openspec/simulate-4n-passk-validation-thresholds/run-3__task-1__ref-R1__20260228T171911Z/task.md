# Validation Bundle

- change-id: simulate-4n-passk-validation-thresholds
- run#: 3
- task-id: 1
- ref-id: R1
- scope: CLI

## How To Run

### macOS/Linux

```bash
bash auto_test_openspec/simulate-4n-passk-validation-thresholds/run-3__task-1__ref-R1__20260228T171911Z/run.sh
```

### Windows

```bat
auto_test_openspec\simulate-4n-passk-validation-thresholds\run-3__task-1__ref-R1__20260228T171911Z\run.bat
```

## Inputs

- `auto_test_openspec/simulate-4n-passk-validation-thresholds/run-3__task-1__ref-R1__20260228T171911Z/inputs/sample_1n.jsonl`
  - Source: Worker-authored minimal 1N PRO sample derived from ACCEPT criteria.

## Outputs

- `auto_test_openspec/simulate-4n-passk-validation-thresholds/run-3__task-1__ref-R1__20260228T171911Z/outputs/simulate_4n.jsonl`
- Stdout from `check_outputs.py` (prints `OK` on success)

## Pass/Fail Criteria (Machine-Decidable)

- `run.sh` / `run.bat` exits with code 0.
- Output file exists at the path above.
- `check_outputs.py` assertions pass:
  - Each JSONL record has `candidates` length 4.
  - Each JSONL record contains `variant_type`.
  - Each JSONL record contains at least one identifier field: `problem_id` or `task_id` or `name`.
