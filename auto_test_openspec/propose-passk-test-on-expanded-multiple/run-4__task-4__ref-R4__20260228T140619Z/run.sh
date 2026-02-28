#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs"

log() {
  printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "Run folder: $RUN_DIR" | tee "$RUN_DIR/logs/run.txt" >/dev/null
log "Repo root:   $REPO_ROOT" | tee -a "$RUN_DIR/logs/run.txt" >/dev/null

{
  echo "UTC_NOW=$(date -u +%Y%m%dT%H%M%SZ)"
  echo "PWD=$(pwd)"
  echo "RUN_DIR=$RUN_DIR"
  echo "REPO_ROOT=$REPO_ROOT"
  echo
  echo "--- git ---"
  (cd "$REPO_ROOT" && git rev-parse --short HEAD) || true
  echo
  echo "--- bash ---"
  command -v bash || true
  bash --version 2>/dev/null | head -n 2 || true
  echo
  echo "--- python ---"
  command -v python || true
  python -V 2>&1 || true
  command -v python3 || true
  python3 -V 2>&1 || true
  echo
  echo "--- docker ---"
  command -v docker || true
  docker --version 2>&1 || true
} >"$RUN_DIR/logs/provenance.txt"

cd "$REPO_ROOT"

CHECKS_LOG="$RUN_DIR/logs/checks.txt"
: >"$CHECKS_LOG"

run_one() {
  local name="$1"
  local cmd="$2"
  local transcript="$RUN_DIR/logs/${name}.txt"

  log "BEGIN: $name" | tee -a "$RUN_DIR/logs/run.txt" >/dev/null
  log "CMD:   $cmd" | tee -a "$RUN_DIR/logs/run.txt" >/dev/null

  # Run and capture transcript (stdout+stderr).
  bash -lc "$cmd" >"$transcript" 2>&1

  # Basic crash guard (machine-decidable; intentionally simple).
  if command -v rg >/dev/null 2>&1; then
    rg -n "Traceback" "$transcript" >/dev/null 2>&1 && {
      echo "FAIL: found 'Traceback' in $transcript" | tee -a "$CHECKS_LOG" >/dev/null
      return 1
    }
  else
    grep -n "Traceback" "$transcript" >/dev/null 2>&1 && {
      echo "FAIL: found 'Traceback' in $transcript" | tee -a "$CHECKS_LOG" >/dev/null
      return 1
    }
  fi

  log "END:   $name" | tee -a "$RUN_DIR/logs/run.txt" >/dev/null
}

python_field_check() {
  local json_path="$1"
  /home/nfs/share-yjy/miniconda3/envs/unsloth-yjy/bin/python3 -c "import json; p=r'''$json_path'''; d=json.load(open(p,'r',encoding='utf-8')); req=['benchmark','pass@1','pass@4','total_candidates','evaluated_candidates']; missing=[k for k in req if k not in d]; assert not missing, missing"
}

# 1) CodeContests path (legacy script)
run_one "run_calculation" "bash scripts/evaluation/FABE/run_calculation.sh"

CODECONTESTS_REPORT="results/evaluation/FABE/java/pass_at_k/final_metrics.json"
test -f "$CODECONTESTS_REPORT"
cp -f "$CODECONTESTS_REPORT" "$RUN_DIR/outputs/final_metrics_codecontests_java.json"
python_field_check "$CODECONTESTS_REPORT"
echo "OK: $CODECONTESTS_REPORT exists + required keys present" | tee -a "$CHECKS_LOG" >/dev/null

# 2) HumanEval (MultiPL-E expanded)
run_one "run_calculation_humaneval" "bash scripts/evaluation/FABE/run_calculation_humaneval.sh"

HUMANEVAL_REPORT="results/evaluation/FABE/humaneval_cpp/pass_at_k/final_metrics.json"
test -f "$HUMANEVAL_REPORT"
cp -f "$HUMANEVAL_REPORT" "$RUN_DIR/outputs/final_metrics_humaneval_cpp.json"
python_field_check "$HUMANEVAL_REPORT"
echo "OK: $HUMANEVAL_REPORT exists + required keys present" | tee -a "$CHECKS_LOG" >/dev/null

# 3) MBPP (MultiPL-E expanded)
run_one "run_calculation_mbpp" "bash scripts/evaluation/FABE/run_calculation_mbpp.sh"

MBPP_REPORT="results/evaluation/FABE/mbpp_cpp/pass_at_k/final_metrics.json"
test -f "$MBPP_REPORT"
cp -f "$MBPP_REPORT" "$RUN_DIR/outputs/final_metrics_mbpp_cpp.json"
python_field_check "$MBPP_REPORT"
echo "OK: $MBPP_REPORT exists + required keys present" | tee -a "$CHECKS_LOG" >/dev/null

log "All commands finished; checks recorded at $CHECKS_LOG" | tee -a "$RUN_DIR/logs/run.txt" >/dev/null
