#!/bin/bash

# ================= 1. 定位项目根目录 =================
# 获取脚本文件所在的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 根据目录结构：scripts/data_preprocessing/ccplus/
# 根目录在该脚本目录的往上 3 层
PROJECT_ROOT="$SCRIPT_DIR/../../.."

# 调试信息：打印出定位到的根目录，方便检查
echo ">>> 定位项目根目录为: $PROJECT_ROOT"

# ================= 配置区域 =================
# 1. 设置目标语言 (cpp / java / python)
TARGET_LANG="java"

# 2. Conda 环境
CONDA_ENV_PATH="/home/nfs/share-yjy/miniconda3/bin/activate"
ENV_NAME="unsloth-yjy"

# 3. Docker 设置
DOCKER_IMAGE="vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609"
START_PORT=12400
END_PORT=12450

# 4. 路径设置 (自动根据语言生成路径)
SCRIPT_PATH="${PROJECT_ROOT}/src/data_preprocessing/CodeContestsPlus/run_selection.py"
# 假设不同语言的数据集在不同子目录下，例如 .../cpp/dataset.jsonl
DATASET_PATH="${PROJECT_ROOT}/data/processed/CodeContestsPlus/ccplus_1x/jsonl/${TARGET_LANG}/dataset.jsonl"
OUTPUT_PATH="${PROJECT_ROOT}/data/processed/CodeContestsPlus/ccplus_1x/jsonl/${TARGET_LANG}/"

# ===========================================

# 1. 激活环境
source "${CONDA_ENV_PATH}" "${ENV_NAME}"
echo "✅ Conda environment '${ENV_NAME}' activated."

# 2. 定义清理函数 (Trap)
function cleanup() {
    echo ""
    echo "🛑 [Cleanup] 检测到脚本结束或中断..."
    echo "   - 正在杀死后台 Python 进程..."
    kill $(jobs -p) 2>/dev/null

    echo "   - 正在删除 Docker 容器..."
    docker ps -a --format '{{.Names}}' | grep '^sandbox-lsl-' | xargs -r docker rm -f > /dev/null
    echo "✅ [Cleanup] 环境清理完毕。"
}
trap cleanup EXIT SIGINT SIGTERM

# 3. 启动 Docker 容器
echo "🚀 正在启动 Sandbox 容器 (${START_PORT} - ${END_PORT})..."
for p in $(seq ${START_PORT} ${END_PORT}); do
    docker rm -f sandbox-lsl-${p} > /dev/null 2>&1
    docker run -d -p ${p}:8080 --name sandbox-lsl-${p} "${DOCKER_IMAGE}" > /dev/null
done

echo "⏳ 等待 20秒 让容器内服务初始化..."
sleep 20

# 4. 获取实际运行的端口列表
PORTS=($(docker ps --format '{{.Names}}' | grep '^sandbox-lsl-' | sed 's/.*-//' | sort -n))
NUM=${#PORTS[@]}
echo "✅ 检测到 ${NUM} 个可用容器。"

# 5. 并发运行 Python 任务
echo "🔥 开始执行 ${NUM} 个分片任务 (语言: ${TARGET_LANG})..."

for IDX in $(seq 0 $((NUM-1))); do
    PORT=${PORTS[$IDX]}

    # 动态生成日志和输出文件名，包含语言标识
    BASE_NAME="ccplus_${TARGET_LANG}_single_full_shard${IDX}"
    LOG_FILE="${OUTPUT_PATH}/${BASE_NAME}.log"
    OUT_JSONL="${OUTPUT_PATH}/${BASE_NAME}.jsonl"
    SUMMARY_JSON="${OUTPUT_PATH}/${BASE_NAME}_summary.json"

    echo "   [Shard ${IDX}] -> Port ${PORT} (Log: ${LOG_FILE})"

    python -u "${SCRIPT_PATH}" \
      --dataset_jsonl "${DATASET_PATH}" \
      --lang "${TARGET_LANG}" \
      --sandbox http://127.0.0.1:${PORT} \
      --out_jsonl "${OUT_JSONL}" \
      --summary "${SUMMARY_JSON}" \
      --sample_k 6 \
      --compile_timeout 20 \
      --default_run_timeout 6 \
      --progress_every 5 \
      --num_shards ${NUM} \
      --shard_id ${IDX} \
      --seed 42 \
      > "${LOG_FILE}" 2>&1 &
done

# 6. 等待所有任务结束
echo "🕒 所有任务已在后台启动，等待完成... (可按 Ctrl+C 中止)"
wait

echo "🎉 所有任务执行完毕！"