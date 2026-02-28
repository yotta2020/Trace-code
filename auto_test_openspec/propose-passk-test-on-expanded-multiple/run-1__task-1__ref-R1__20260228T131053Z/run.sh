#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs outputs

RUN_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="logs/run_${RUN_UTC}.txt"
VALIDATE_LOG="logs/validate_stdout_${RUN_UTC}.txt"
REPORT_JSON="outputs/validation_report_${RUN_UTC}.json"

{
  echo "[INFO] Running R1 documentation consistency checks..."
  echo "[INFO] PWD=${PWD}"
  echo "[INFO] python3=$(command -v python3 || true)"
  python3 --version || true
  echo
  echo "[INFO] RUN_UTC=${RUN_UTC}"
  echo "[INFO] RUN_LOG=${RUN_LOG}"
  echo "[INFO] VALIDATE_LOG=${VALIDATE_LOG}"
  echo "[INFO] REPORT_JSON=${REPORT_JSON}"
  echo

  python3 tests/validate_r1.py --report "${REPORT_JSON}" 2>&1 | tee "${VALIDATE_LOG}"
} | tee "${RUN_LOG}"
