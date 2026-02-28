#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../.." && pwd)"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs"

{
  echo "RUN_DIR=${RUN_DIR}"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "UTC_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PYTHON_BIN=${PYTHON_BIN:-python3}"
  echo "--- python version ---"
  "${PYTHON_BIN:-python3}" --version || true
  echo "--- python path ---"
  command -v "${PYTHON_BIN:-python3}" || true
  echo
  "${PYTHON_BIN:-python3}" "${RUN_DIR}/tests/check_r3_codecontests_paths.py" \
    --repo_root "${REPO_ROOT}" \
    --out_json "${RUN_DIR}/outputs/check_results.json"
} 2>&1 | tee "${RUN_DIR}/logs/run.txt"

