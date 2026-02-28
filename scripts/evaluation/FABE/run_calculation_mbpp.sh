#!/bin/bash
set -e

# --- Configurable defaults (override via env vars) ---
PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/unsloth-lsl/bin/python}"
TARGET_LANG="${TARGET_LANG:-cpp}"
BENCHMARK="mbpp"
INFERENCE_RESULTS="${INFERENCE_RESULTS:-results/evaluation/FABE/mbpp_${TARGET_LANG}/pass_at_k/inference_results.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation/FABE/mbpp_${TARGET_LANG}/pass_at_k/shards}"
REPORT_PATH="${REPORT_PATH:-results/evaluation/FABE/mbpp_${TARGET_LANG}/pass_at_k/final_metrics.json}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609}"
START_PORT="${START_PORT:-12420}"
END_PORT="${END_PORT:-12429}"
RUN_TIMEOUT="${RUN_TIMEOUT:-20}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-20}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

mkdir -p "$OUTPUT_DIR"

# --- Cleanup trap ---
cleanup() {
  kill $(jobs -p) 2>/dev/null || true
  docker ps -a --format '{{.Names}}' | grep '^sandbox-eval-mbpp-' | xargs -r docker rm -f >/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

# --- Start Docker containers ---
for p in $(seq "$START_PORT" "$END_PORT"); do
  docker rm -f sandbox-eval-mbpp-"$p" >/dev/null 2>&1 || true
  docker run -d -p "$p":8080 --name sandbox-eval-mbpp-"$p" "$DOCKER_IMAGE" >/dev/null
  sleep 1
done
sleep 15

# --- Launch parallel shards ---
PORTS=($(seq "$START_PORT" "$END_PORT"))
NUM_SHARDS=${#PORTS[@]}

HF_CACHE_ARG=""
if [ -n "$HF_CACHE_DIR" ]; then
  HF_CACHE_ARG="--hf_cache_dir $HF_CACHE_DIR"
fi

for IDX in $(seq 0 $((NUM_SHARDS - 1))); do
  PORT=${PORTS[$IDX]}
  OUT_SHARD="$OUTPUT_DIR/shard_${IDX}.jsonl"

  "$PYTHON_BIN" -u src/evaluation/FABE/Calculate_passk_multiple.py \
    --inference_results "$INFERENCE_RESULTS" \
    --sandbox "http://127.0.0.1:${PORT}" \
    --out_shard_jsonl "$OUT_SHARD" \
    --benchmark "$BENCHMARK" \
    --lang "$TARGET_LANG" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$IDX" \
    --run_timeout "$RUN_TIMEOUT" \
    --compile_timeout "$COMPILE_TIMEOUT" \
    $HF_CACHE_ARG >"$OUTPUT_DIR/shard_${IDX}.log" 2>&1 &
done

wait

# --- Aggregate results ---
"$PYTHON_BIN" src/evaluation/FABE/aggregate_results.py \
  --shard_dir "$OUTPUT_DIR" \
  --save_report "$REPORT_PATH" \
  --benchmark "$BENCHMARK"

printf "MBPP pass@k calculation done: %s\n" "$REPORT_PATH"
