#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../.." && pwd)"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$LOG_DIR"

SCRIPT="${REPO_ROOT}/scripts/evaluation/FABE/run_passk_gate_validation.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: script not found: $SCRIPT" >&2
  exit 1
fi

HELP_LOG="${LOG_DIR}/help_stdout.txt"
HELP_ERR="${LOG_DIR}/help_stderr.txt"

bash "$SCRIPT" --help >"$HELP_LOG" 2>"$HELP_ERR"

grep -q -- "--n" "$HELP_LOG"
grep -q -- "--augment_types" "$HELP_LOG"
grep -q -- "--mode" "$HELP_LOG"
grep -q -- "--stop_on_pass1_fail" "$HELP_LOG"

RUN_LOG="${LOG_DIR}/run_stdout.txt"
RUN_ERR="${LOG_DIR}/run_stderr.txt"

bash "$SCRIPT" --n 4 --augment_types rename1,rename2,dead1,dead2 --mode pass1_only >"$RUN_LOG" 2>"$RUN_ERR"

grep -q "CONFIG: n=4" "$RUN_LOG"
grep -q "CONFIG: augment_types=rename1,rename2,dead1,dead2" "$RUN_LOG"
grep -q "CONFIG: mode=pass1_only" "$RUN_LOG"

echo "CLI validation completed."
