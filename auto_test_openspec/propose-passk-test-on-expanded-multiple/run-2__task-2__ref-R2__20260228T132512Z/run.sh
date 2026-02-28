#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${RUN_DIR}/../../.." && pwd)"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs"

{
  echo "UTC_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "RUN_DIR=${RUN_DIR}"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo
  echo "[provenance] uname: $(uname -a 2>/dev/null || true)"
  echo -n "[provenance] python: "; command -v python3 || command -v python || true
  (python3 --version 2>/dev/null || python --version 2>/dev/null || true) | sed 's/^/[provenance] /'
  if command -v uv >/dev/null 2>&1; then
    echo "[provenance] uv: $(uv --version 2>/dev/null || true)"
  else
    echo "[provenance] uv: (not installed)"
  fi
  if command -v git >/dev/null 2>&1; then
    echo "[provenance] git_base: $(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || true)"
  fi
  echo "[provenance] deps: stdlib-only (no installs)"
} >"${RUN_DIR}/logs/provenance.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

"${PYTHON_BIN}" "${RUN_DIR}/tests/check_r2_spec_chain.py" \
  --repo-root "${ROOT_DIR}" \
  --out "${RUN_DIR}/outputs/check_r2_spec_chain.json" \
  >"${RUN_DIR}/logs/run_stdout.txt" 2>"${RUN_DIR}/logs/run_stderr.txt"

echo "OK: wrote outputs/check_r2_spec_chain.json"

