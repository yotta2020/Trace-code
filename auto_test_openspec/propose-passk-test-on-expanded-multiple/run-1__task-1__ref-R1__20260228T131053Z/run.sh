#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs outputs

{
  echo "[INFO] Running R1 documentation consistency checks..."
  echo "[INFO] PWD=${PWD}"
  echo "[INFO] python3=$(command -v python3 || true)"
  python3 --version || true
  echo
  python3 tests/validate_r1.py 2>&1 | tee logs/validate_stdout.txt
} | tee logs/run.txt

