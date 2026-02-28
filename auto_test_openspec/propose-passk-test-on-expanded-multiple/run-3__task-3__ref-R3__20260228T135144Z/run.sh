#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs outputs

{
  echo "=== run.sh start (UTC) $(date -u +%Y%m%dT%H%M%SZ) ==="
  echo "PWD=$PWD"
  echo "uname=$(uname -a 2>/dev/null || true)"
  echo ""
} >> logs/run.txt

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python not found in PATH" | tee -a logs/run.txt
  exit 1
fi

{
  echo "PYTHON_BIN=$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1 || true
  command -v "$PYTHON_BIN" 2>&1 || true
  echo ""
} >> logs/run.txt

set +e
"$PYTHON_BIN" tests/check_r3_codecontests_path_compat.py 2>&1 | tee -a logs/run.txt
EXIT_CODE=${PIPESTATUS[0]}
set -e

{
  echo ""
  echo "EXIT_CODE=$EXIT_CODE"
  echo "=== run.sh end (UTC) $(date -u +%Y%m%dT%H%M%SZ) ==="
  echo ""
} >> logs/run.txt

exit "$EXIT_CODE"

