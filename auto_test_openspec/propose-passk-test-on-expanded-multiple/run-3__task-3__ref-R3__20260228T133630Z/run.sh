#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# Run from repo root regardless of invocation CWD.
REPO_ROOT="$(cd -- "$RUN_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

exec > >(tee "$LOG_DIR/run_stdout.txt") 2> >(tee "$LOG_DIR/run_stderr.txt" >&2)

SPEC_PATH="openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
CODECONTESTS_SCRIPT="scripts/evaluation/FABE/run_calculation.sh"
CALC_PY="src/evaluation/FABE/Calculate_passk.py"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "Repo root: $REPO_ROOT"
echo "Run dir :  $RUN_DIR"

test -f "$SPEC_PATH" || fail "Missing spec: $SPEC_PATH"
test -f "$CODECONTESTS_SCRIPT" || fail "Missing script: $CODECONTESTS_SCRIPT"
test -f "$CALC_PY" || fail "Missing code: $CALC_PY"

grep -qF "scripts/evaluation/FABE/run_calculation.sh" "$SPEC_PATH" || fail "Spec missing CodeContests entry script path"
grep -qF "src/evaluation/FABE/Calculate_passk.py" "$SPEC_PATH" || fail "Spec missing CodeContests calculation code path"
grep -qF "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json" "$SPEC_PATH" || fail "Spec missing final_metrics.json path contract for CodeContests"

grep -qF "src/evaluation/FABE/Calculate_passk.py" "$CODECONTESTS_SCRIPT" || fail "CodeContests script no longer references Calculate_passk.py"
grep -qF 'results/evaluation/FABE/${TARGET_LANG}/pass_at_k/inference_results.jsonl' "$CODECONTESTS_SCRIPT" || fail "CodeContests script input path pattern changed"
grep -qF "pass_at_k/final_metrics.json" "$CODECONTESTS_SCRIPT" || fail "CodeContests script output path contract changed"

echo "OK: CodeContests pass@k paths remain compatible (static checks passed)."

