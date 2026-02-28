#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs outputs

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PY_LOG="logs/python_provenance__${RUN_TS}.txt"
AGG_LOG="logs/aggregate_stdout__${RUN_TS}.txt"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

{
  echo "PWD=$PWD"
  echo "RUN_TS=$RUN_TS"
  echo "PYTHON_BIN=$PYTHON_BIN"
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c "import sys; print('sys.executable=' + sys.executable)"
} > "$PY_LOG" 2>&1

set +e
"$PYTHON_BIN" "$REPO_ROOT/src/evaluation/FABE/aggregate_results.py" \
  --shard_dir "$SCRIPT_DIR/inputs/shards" \
  --save_report "$SCRIPT_DIR/outputs/final_metrics.json" \
  > "$AGG_LOG" 2>&1
AGG_RC=$?
set -e

if [[ $AGG_RC -ne 0 ]]; then
  echo "aggregate_results.py failed with exit code $AGG_RC" >&2
  tail -n 200 "$AGG_LOG" >&2 || true
  exit "$AGG_RC"
fi

"$PYTHON_BIN" - <<'PY'
import json
import os

out_path = os.path.join("outputs", "final_metrics.json")
assert os.path.exists(out_path), f"missing output: {out_path}"

with open(out_path, "r", encoding="utf-8") as f:
    data = json.load(f)

assert isinstance(data, dict), "final_metrics.json must be a JSON object"
assert "unknown" in data, "missing fallback bucket: unknown"
bucket = data["unknown"]
assert isinstance(bucket, dict), "unknown bucket must be a JSON object"
for k in ("p1", "p2", "p4"):
    assert k in bucket, f"missing key in unknown bucket: {k}"
    assert isinstance(bucket[k], (int, float)), f"{k} must be numeric"
print("OK: aggregation compatible with missing variant_type (bucket=unknown)")
PY

echo "DONE: outputs/final_metrics.json"
