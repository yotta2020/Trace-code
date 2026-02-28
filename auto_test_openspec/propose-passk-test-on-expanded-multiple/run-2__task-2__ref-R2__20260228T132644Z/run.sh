#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

{
  echo "python3_path: $(command -v python3 || echo NOT_FOUND)"
  echo "python_path: $(command -v python || echo NOT_FOUND)"
  if command -v python3 >/dev/null 2>&1; then
    python3 --version
  elif command -v python >/dev/null 2>&1; then
    python --version
  fi
} > logs/provenance.txt 2>&1 || true

PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python"
fi

set +e
"$PY_BIN" tests/test_r2_spec_chain.py > logs/run_stdout.txt 2>&1
EXIT_CODE=$?
set -e

cat logs/run_stdout.txt
exit "$EXIT_CODE"

