#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/unsloth-lsl/bin/python}"
TARGET_LANG="${TARGET_LANG:-cpp}"
INFERENCE_RESULTS="${INFERENCE_RESULTS:-results/evaluation/FABE/xlcost_cpp/pass_at_k/inference_results.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation/FABE/xlcost_cpp/pass_at_k/shards}"
REPORT_PATH="${REPORT_PATH:-results/evaluation/FABE/xlcost_cpp/pass_at_k/final_metrics.json}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609}"
START_PORT="${START_PORT:-12410}"
END_PORT="${END_PORT:-12419}"

mkdir -p "$OUTPUT_DIR"

cleanup() {
  kill $(jobs -p) 2>/dev/null || true
  docker ps -a --format '{{.Names}}' | grep '^sandbox-eval-xlcost-' | xargs -r docker rm -f >/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

for p in $(seq "$START_PORT" "$END_PORT"); do
  docker rm -f sandbox-eval-xlcost-"$p" >/dev/null 2>&1 || true
  docker run -d -p "$p":8080 --name sandbox-eval-xlcost-"$p" "$DOCKER_IMAGE" >/dev/null
  sleep 1
done
sleep 15

PORTS=($(seq "$START_PORT" "$END_PORT"))
NUM_SHARDS=${#PORTS[@]}

for IDX in $(seq 0 $((NUM_SHARDS - 1))); do
  PORT=${PORTS[$IDX]}
  OUT_SHARD="$OUTPUT_DIR/shard_${IDX}.jsonl"

  "$PYTHON_BIN" -u src/evaluation/FABE/Calculate_passk.py \
    --inference_results "$INFERENCE_RESULTS" \
    --sandbox "http://127.0.0.1:${PORT}" \
    --out_shard_jsonl "$OUT_SHARD" \
    --lang "$TARGET_LANG" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$IDX" >"$OUTPUT_DIR/shard_${IDX}.log" 2>&1 &
done

wait

"$PYTHON_BIN" src/evaluation/FABE/aggregate_results.py \
  --shard_dir "$OUTPUT_DIR" \
  --save_report "$REPORT_PATH"

printf "XLCoST pass@k calculation done: %s\n" "$REPORT_PATH"
