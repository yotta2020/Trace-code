#!/bin/bash

# ==============================================================================
# Pass@k 评测自动化脚本
# ==============================================================================

# 1. 路径配置 (核心修改点)
TARGET_LANG="java"

INFERENCE_RESULTS="results/evaluation/FABE/${TARGET_LANG}/pass_at_k/inference_results.jsonl"

OUTPUT_DIR="results/evaluation/FABE/${TARGET_LANG}/pass_at_k/shards"

# 沙箱设置
DOCKER_IMAGE="vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609"
START_PORT=12410
END_PORT=12429  # 开启 20 个并发沙箱

# Conda 环境
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-lsl

# ===========================================

mkdir -p "$OUTPUT_DIR"

# 清理并启动容器
function cleanup() {
    echo "🛑 [Cleanup] 停止进程并删除容器..."
    kill $(jobs -p) 2>/dev/null
    docker ps -a --format '{{.Names}}' | grep '^sandbox-eval-' | xargs -r docker rm -f > /dev/null
}
trap cleanup EXIT SIGINT SIGTERM

echo "🚀 启动并发沙箱容器..."
for p in $(seq ${START_PORT} ${END_PORT}); do
    echo "⚙️  Preparing sandbox-eval-${p}..."
    docker rm -f sandbox-eval-${p} > /dev/null 2>&1
    sleep 1 # 给系统一点释放资源的时间
    docker run -d -p ${p}:8080 --name sandbox-eval-${p} "${DOCKER_IMAGE}"
done

echo "⏳ 等待容器初始化 (20s)..."
sleep 20

# 获取端口列表并分片执行
PORTS=($(seq ${START_PORT} ${END_PORT}))
NUM_SHARDS=${#PORTS[@]}

for IDX in $(seq 0 $((NUM_SHARDS-1))); do
    PORT=${PORTS[$IDX]}
    OUT_SHARD="${OUTPUT_DIR}/shard_${IDX}.jsonl"
    
    echo "🔥 启动 Shard ${IDX} (Port ${PORT})"
    python -u "src/evaluation/FABE/Calculate_passk.py" \
      --inference_results "${INFERENCE_RESULTS}" \
      --sandbox "http://127.0.0.1:${PORT}" \
      --out_shard_jsonl "${OUT_SHARD}" \
      --lang "${TARGET_LANG}" \
      --num_shards ${NUM_SHARDS} \
      --shard_id ${IDX} > "${OUTPUT_DIR}/shard_${IDX}.log" 2>&1 &
done

wait
echo "🎉 所有分片验证完毕！"

# 最后一步：聚合所有分片并计算最终报表
python "src/evaluation/FABE/aggregate_results.py" \
    --shard_dir "$OUTPUT_DIR" \
    --save_report "results/evaluation/FABE/${TARGET_LANG}/pass_at_k/final_metrics.json"

echo "==== 评测完成！结果已输出至终端及 $OUTPUT_DIR ===="