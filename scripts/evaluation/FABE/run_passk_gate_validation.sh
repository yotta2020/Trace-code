#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_passk_gate_validation.sh [options]

Options:
  --n <int>                   Candidate count (e.g., 4)
  --augment_types <csv>        Augment types, comma-separated (e.g., rename1,rename2,dead1,dead2)
  --mode <string>             Validation mode (e.g., pass1_only, pass4_only, pass1_then_pass4)
  --stop_on_pass1_fail[=bool] Stop when pass@1 gate fails (flag or true/false)
  -h, --help                  Show this help message

Notes:
  This script currently parses and echoes configuration. Full gate execution
  (pass@1 / pass@4 pipelines) is handled in later tasks.
USAGE
}

parse_bool() {
  local raw="$1"
  case "${raw,,}" in
    1|true|yes|y) echo "1" ;;
    0|false|no|n) echo "0" ;;
    *)
      echo "ERROR: invalid boolean value: ${raw}" >&2
      exit 2
      ;;
  esac
}

N="4"
AUGMENT_TYPES="rename1,rename2,dead1,dead2"
MODE="pass1_only"
STOP_ON_PASS1_FAIL="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --n)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --n requires a value" >&2
        exit 2
      fi
      N="$2"
      shift 2
      ;;
    --augment_types)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --augment_types requires a value" >&2
        exit 2
      fi
      AUGMENT_TYPES="$2"
      shift 2
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --mode requires a value" >&2
        exit 2
      fi
      MODE="$2"
      shift 2
      ;;
    --stop_on_pass1_fail)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        STOP_ON_PASS1_FAIL="$(parse_bool "$2")"
        shift 2
      else
        STOP_ON_PASS1_FAIL="1"
        shift 1
      fi
      ;;
    --stop_on_pass1_fail=*)
      STOP_ON_PASS1_FAIL="$(parse_bool "${1#*=}")"
      shift 1
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Run with --help to see supported options." >&2
      exit 2
      ;;
  esac
 done

if ! [[ "$N" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --n must be a positive integer" >&2
  exit 2
fi

printf "PASSK gate validation (config only)\n"
printf "CONFIG: n=%s\n" "$N"
printf "CONFIG: augment_types=%s\n" "$AUGMENT_TYPES"
printf "CONFIG: mode=%s\n" "$MODE"
printf "CONFIG: stop_on_pass1_fail=%s\n" "$STOP_ON_PASS1_FAIL"
