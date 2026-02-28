#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SELF_DIR"

mkdir -p logs outputs

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "ERROR: python/python3 not found in PATH" >&2
  exit 2
fi

{
  echo "UTC_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PY_BIN=${PY_BIN}"
  "${PY_BIN}" --version 2>&1 || true
} > logs/run.txt

"${PY_BIN}" -u tests/validate_r3.py 2>&1 | tee -a logs/run.txt
