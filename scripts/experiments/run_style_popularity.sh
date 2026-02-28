#!/bin/bash

# Style Popularity Experiment Runner
# 统计不同任务数据集的代码风格流行度 (仅 Train 集)

set -e

# ==================== 配置区 ====================

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 数据目录
DATA_ROOT="${PROJECT_ROOT}/data/processed"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/style_popularity"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/log/experiments"

# 创建必要的目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ==================== 函数定义 ====================

run_experiment() {
    local task=$1
    local data_dir=$2
    local language=$3
    local output_file=$4
    local log_file=$5

    echo ""
    echo "=========================================="
    echo "Running Style Popularity for ${task^^}"
    if [ -n "$language" ]; then
        echo "Language: $language"
    fi
    echo "=========================================="
    echo "Data directory: $data_dir"
    echo "Output file: $output_file"
    echo "Log file: $log_file"
    echo ""

    if [ ! -d "$data_dir" ]; then
        echo "Warning: Data directory not found: $data_dir"
        echo "Skipping this experiment."
        return 1
    fi

    # 构建命令
    # 注意：显式添加 --splits train
    cmd="/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ${PROJECT_ROOT}/src/experiments/style_popularity.py \
        --task $task \
        --data_dir $data_dir \
        --output_file $output_file \
        --log_file $log_file \
        --splits train"

    # 如果有语言参数，添加到命令
    if [ -n "$language" ]; then
        cmd="$cmd --language $language"
    fi

    # 执行命令
    echo "Executing: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Experiment completed successfully!"
        echo "Results: $output_file"
        return 0
    else
        echo ""
        echo "✗ Experiment failed!"
        echo "Check log: $log_file"
        return 1
    fi
}

# ==================== 主程序 ====================

echo "======================================================"
echo "Style Popularity Experiment Suite (Train Split Only)"
echo "======================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 统计成功和失败的实验数
SUCCESS_COUNT=0
FAILED_COUNT=0

# 1. Clone Detection (CD) - Java
if run_experiment \
    "cd" \
    "${DATA_ROOT}/cd" \
    "" \
    "${OUTPUT_DIR}/cd_style_popularity.json" \
    "${LOG_DIR}/cd_style_popularity.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 2. Defect Detection (DD) - C
if run_experiment \
    "dd" \
    "${DATA_ROOT}/dd" \
    "" \
    "${OUTPUT_DIR}/dd_style_popularity.json" \
    "${LOG_DIR}/dd_style_popularity.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 3. Code Search - Python (CS-Python)
if run_experiment \
    "cs" \
    "${DATA_ROOT}/cs/python" \
    "python" \
    "${OUTPUT_DIR}/cs_python_style_popularity.json" \
    "${LOG_DIR}/cs_python_style_popularity.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 4. Code Search - Java (CS-Java)
if run_experiment \
    "cs" \
    "${DATA_ROOT}/cs/java" \
    "java" \
    "${OUTPUT_DIR}/cs_java_style_popularity.json" \
    "${LOG_DIR}/cs_java_style_popularity.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# ==================== 总结 ====================

echo ""
echo "======================================================"
echo "Experiment Suite Summary"
echo "======================================================"
echo "Total experiments: $((SUCCESS_COUNT + FAILED_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""

if [ $FAILED_COUNT -eq 0 ]; then
    echo "✓ All experiments completed successfully!"
    exit 0
else
    echo "⚠ Some experiments failed. Check the logs for details."
    exit 1
fi