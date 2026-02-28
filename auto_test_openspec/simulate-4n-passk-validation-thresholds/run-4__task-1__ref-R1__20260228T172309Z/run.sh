#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_DIR="${ROOT}/auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z"

echo "Generating 4N simulated candidates..."
python3 "${ROOT}/src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py" \
  "${RUN_DIR}/inputs/sample_1n.jsonl" \
  "${RUN_DIR}/outputs/simulate_4n.jsonl" \
  --mode simulate_4n \
  --lang cpp \
  --split train \
  --seed 42

echo "Validating output..."
python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        assert len(data.get("candidates", [])) == 4, "Missing or incorrect candidates length"
        assert "variant_type" in data, "Missing variant_type"
        assert "problem_id" in data or "id" in data, "Missing problem ID"
        print("Validation passed for:", data.get("id"))
' "${RUN_DIR}/outputs/simulate_4n.jsonl"
