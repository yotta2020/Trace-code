#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../.." && pwd)"

INPUT="${RUN_DIR}/inputs/sample_1n.jsonl"
OUTPUT="${RUN_DIR}/outputs/simulated_4n.jsonl"

python "${REPO_ROOT}/src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py" \
  "${INPUT}" "${OUTPUT}" \
  --split test \
  --lang cpp \
  --mode 4n

python "${RUN_DIR}/tests/validate_4n.py" --input "${OUTPUT}"
