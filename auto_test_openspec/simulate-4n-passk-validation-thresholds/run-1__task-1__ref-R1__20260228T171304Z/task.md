# Task 1: 4N 模拟输出模式 (R1)

## Goal
Generate 4N simulated outputs from 1N input with 2 variable renames + 2 dead-code insertions, and keep pass@k required fields.

## How To Run

```bash
bash run.sh
```

```bat
run.bat
```

## Pass/Fail Criteria (Machine-Decidable)
- The output JSONL exists at `outputs/simulate_4n.jsonl`.
- Every record has `candidates` length exactly 4.
- Every record includes `variant_type` and a non-empty `problem_id` field.
- `run.sh`/`run.bat` exit code is 0.

## Notes
- Inputs are under `inputs/sample_1n.jsonl`.
- Outputs are written to `outputs/`.
