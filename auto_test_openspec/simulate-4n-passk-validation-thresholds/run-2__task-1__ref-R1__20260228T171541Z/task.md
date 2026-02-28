# Validation Bundle: simulate-4n-passk-validation-thresholds

- change-id: simulate-4n-passk-validation-thresholds
- run#: 2
- task-id: 1
- ref-id: R1
- scope: CLI

## What This Validates
Generates a 4N simulation output using `generate_11n_dataset.py --mode 4n` and checks:
- Each record has exactly 4 `candidates`.
- Each record includes `variant_type`.
- Each record includes a problem identifier field (`name`, `task_id`, or `problem_id`).

## How To Run
- macOS/Linux:
  - `auto_test_openspec/simulate-4n-passk-validation-thresholds/run-2__task-1__ref-R1__20260228T171541Z/run.sh`
- Windows:
  - `auto_test_openspec\simulate-4n-passk-validation-thresholds\run-2__task-1__ref-R1__20260228T171541Z\run.bat`

## Test Inputs
- `auto_test_openspec/simulate-4n-passk-validation-thresholds/run-2__task-1__ref-R1__20260228T171541Z/inputs/sample_1n.jsonl`

Provenance: The sample input is a minimal 1N record constructed from the R1 ACCEPT criteria to validate schema generation.

## Test Outputs
- `auto_test_openspec/simulate-4n-passk-validation-thresholds/run-2__task-1__ref-R1__20260228T171541Z/outputs/simulated_4n.jsonl`
- stdout from `tests/validate_4n.py`

## Pass/Fail Criteria (Machine-Decidable)
- Exit code is 0.
- stdout contains `ALL_CHECKS_PASS=1`.
- `outputs/simulated_4n.jsonl` exists and contains records where:
  - `candidates` length is exactly 4.
  - `variant_type` exists.
  - At least one of `name`, `task_id`, or `problem_id` exists.
