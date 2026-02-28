# R1: 4N Candidate Generation For Pass@k Validation

This bundle tests the new 4N simulation mode in `generate_11n_dataset.py`.

## Requirements
- Each generated record MUST contain exactly 4 candidates.
- `variant_type` MUST be present (e.g., "simulate_4n" or "4n_simulation").
- A problem identifier (`problem_id` or `id`) MUST be present.

## How to Run
```bash
./run.sh
```

## Validation Criteria
- Exit code 0
- Output JSONL file has records with `candidates` list of length 4
- Output JSONL records contain `variant_type`
- Output JSONL records contain `problem_id` or `id`
