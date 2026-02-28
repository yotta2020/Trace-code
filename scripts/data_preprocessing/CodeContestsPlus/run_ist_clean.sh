#!/bin/bash

# Run IST cleaning (task1) with multiple SandboxFusion docker instances.

set -e

TARGET_LANG="${1:-java}"

if [[ "${TARGET_LANG}" != "cpp" && "${TARGET_LANG}" != "java" && "${TARGET_LANG}" != "py3" ]]; then
  echo "❌ Unsupported lang: ${TARGET_LANG} (expected: cpp/java/py3)"
  exit 1
fi

CONDA_ENV_PATH="/home/nfs/share-yjy/miniconda3/bin/activate"
ENV_NAME="unsloth-yjy"

CONTAINER_PREFIX="sandbox-lsl"
IMAGE="vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609"
AUTO_START_SANDBOXES="${AUTO_START_SANDBOXES:-1}"
SANDBOX_PORT_START="${SANDBOX_PORT_START:-12401}"
SANDBOX_PORT_END="${SANDBOX_PORT_END:-12430}"
SANDBOX_START_DELAY_SEC="${SANDBOX_START_DELAY_SEC:-3}"
SANDBOX_READY_WAIT_SEC="${SANDBOX_READY_WAIT_SEC:-8}"

SCRIPT_PATH="src/data_preprocessing/CodeContestsPlus/ist_clean.py"
if [ "${TARGET_LANG}" = "cpp" ]; then
  DATASET_PATH="data/processed/CodeContestsPlus/ccplus_1x/jsonl/cpp/merged/cpp_single.jsonl"
elif [ "${TARGET_LANG}" = "java" ]; then
  DATASET_PATH="data/processed/CodeContestsPlus/ccplus_1x/jsonl/java/merged/java_single.jsonl"
else
  DATASET_PATH="data/processed/CodeContestsPlus/ccplus_1x/jsonl/py3/merged/py3_single.jsonl"
fi

OUT_DIR_DEFAULT="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${TARGET_LANG}/ist_cleaned"
OUT_DIR="${ISTCLEAN_OUT_DIR:-${OUT_DIR_DEFAULT}}"
mkdir -p "${OUT_DIR}"

function cleanup() {
  echo ""
  echo "🛑 [Cleanup] stopping python jobs and docker containers..."
  kill $(jobs -p) 2>/dev/null || true
  docker ps -a --format '{{.Names}}' | grep "^${CONTAINER_PREFIX}-" | xargs -r docker rm -f || true
  echo "✅ [Cleanup] done"
}
trap cleanup EXIT SIGINT SIGTERM

source "${CONDA_ENV_PATH}" "${ENV_NAME}"
echo "✅ Conda environment '${ENV_NAME}' activated."

if [ "${AUTO_START_SANDBOXES}" = "1" ]; then
  echo "🚀 Auto-starting sandboxes: ${CONTAINER_PREFIX}-${SANDBOX_PORT_START}..${SANDBOX_PORT_END}"
  for p in $(seq "${SANDBOX_PORT_START}" "${SANDBOX_PORT_END}"); do
    NAME="${CONTAINER_PREFIX}-${p}"
    if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
      echo "   [Skip] ${NAME} already running"
      continue
    fi
    if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
      echo "   [Remove] ${NAME} exists but stopped"
      docker rm -f "${NAME}" || true
    fi
    echo "   [Start] ${NAME} on port ${p}"
    docker run -d -p ${p}:8080 --name "${NAME}" "${IMAGE}"
    sleep "${SANDBOX_START_DELAY_SEC}"
  done
  echo "⏳ Waiting ${SANDBOX_READY_WAIT_SEC}s for sandboxes to be ready..."
  sleep "${SANDBOX_READY_WAIT_SEC}"
else
  echo "ℹ️  Expecting sandbox containers already running with names '${CONTAINER_PREFIX}-<port>'."
fi

echo "🔎 Detecting running sandboxes..."
PORTS=($(docker ps --format '{{.Names}}' | grep "^${CONTAINER_PREFIX}-" | sed 's/.*-//' | sort -n))
NUM=${#PORTS[@]}
if [ "${NUM}" -le 0 ]; then
  echo "❌ No sandbox containers detected (prefix '${CONTAINER_PREFIX}-')."
  exit 1
fi

echo "✅ Detected ${NUM} sandbox containers."

echo "🔥 Launching ${NUM} shard jobs (lang: ${TARGET_LANG})..."

for IDX in $(seq 0 $((NUM-1))); do
  PORT=${PORTS[$IDX]}
  BASE_NAME="ccplus_${TARGET_LANG}_istclean_shard${IDX}_of_${NUM}"

  OUT_JSONL="${OUT_DIR}/${BASE_NAME}.jsonl"
  SUMMARY_JSON="${OUT_DIR}/${BASE_NAME}.summary.json"
  FAIL_JSONL="${OUT_DIR}/${BASE_NAME}.failures.jsonl"

  echo "   [Shard ${IDX}] -> Port ${PORT}"

  python -u "${SCRIPT_PATH}" \
    --lang "${TARGET_LANG}" \
    --in_jsonl "${DATASET_PATH}" \
    --out_jsonl "${OUT_JSONL}" \
    --summary_json "${SUMMARY_JSON}" \
    --failures_jsonl "${FAIL_JSONL}" \
    --sandbox "http://127.0.0.1:${PORT}" \
    --compile_timeout 20 \
    --default_run_timeout 10 \
    --request_timeout_slack 240 \
    --request_retries 2 \
    --request_retry_sleep_s 1.0 \
    --progress_every 200 \
    --num_shards ${NUM} \
    --shard_id ${IDX} \
    > /dev/stdout 2>&1 &

done

echo "🕒 All shard jobs started; waiting..."
wait

echo "🎉 All shard jobs finished."

echo "Next: merge outputs via scripts/data_preprocessing/ccplus/merge_istclean_shards.sh"