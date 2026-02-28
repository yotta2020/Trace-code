#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

mkdir -p logs outputs

{
  echo "UTC_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PWD=$PWD"
  echo "PYTHON3=$(command -v python3 || true)"
  python3 -V 2>&1 || true
} > logs/runner_provenance.txt

python3 tests/check_r3_codecontests_path_compat.py \
  --report outputs/report.json 2>&1 | tee logs/validation.txt

