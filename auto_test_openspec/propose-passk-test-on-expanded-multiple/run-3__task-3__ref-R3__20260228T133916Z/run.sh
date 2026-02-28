#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

mkdir -p logs outputs

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

{
  echo "UTC_TIMESTAMP: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "REPO_ROOT: ${REPO_ROOT}"
  echo "PYTHON_BIN: ${PYTHON_BIN}"
  command -v "$PYTHON_BIN" 2>&1 || true
  "$PYTHON_BIN" --version 2>&1 || true
} > logs/provenance.txt

"$PYTHON_BIN" -u tests/check_r3_codecontests_paths.py \
  --repo_root "$REPO_ROOT" \
  --out_json outputs/r3_check_report.json > logs/check_r3_stdout.txt 2>&1

cat logs/check_r3_stdout.txt
echo "Wrote: outputs/r3_check_report.json"
