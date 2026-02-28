#!/bin/bash

# Docstring Word Frequency Analysis Runner
# 统计Code Search训练集的docstring词频

set -e

# ==================== 配置区 ====================

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 数据目录
DATA_ROOT="${PROJECT_ROOT}/data/processed/cs"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/results/docstring_word_frequency"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/log/experiments"

# Python解释器（根据需要修改）
PYTHON="/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python"

# 创建必要的目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ==================== 函数定义 ====================

run_analysis() {
    local language=$1
    local data_file=$2
    local output_file=$3
    local log_file=$4

    echo ""
    echo "=========================================="
    echo "Analyzing Docstring Word Frequency"
    echo "Language: ${language^^}"
    echo "=========================================="
    echo "Data file: $data_file"
    echo "Output file: $output_file"
    echo "Log file: $log_file"
    echo ""

    if [ ! -f "$data_file" ]; then
        echo "Warning: Data file not found: $data_file"
        echo "Skipping this analysis."
        return 1
    fi

    # 执行分析
    $PYTHON "${PROJECT_ROOT}/src/experiments/docstring_word_frequency.py" \
        --language "$language" \
        --data_file "$data_file" \
        --output_file "$output_file" \
        --log_file "$log_file"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Analysis completed successfully!"
        echo "Results: $output_file"
        return 0
    else
        echo ""
        echo "✗ Analysis failed!"
        echo "Check log: $log_file"
        return 1
    fi
}

# ==================== 主程序 ====================

echo "======================================================"
echo "Docstring Word Frequency Analysis Suite"
echo "======================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 统计成功和失败的分析数
SUCCESS_COUNT=0
FAILED_COUNT=0

# 1. Python
if run_analysis \
    "python" \
    "${DATA_ROOT}/python/train.jsonl" \
    "${OUTPUT_DIR}/python_word_freq.txt" \
    "${LOG_DIR}/docstring_word_freq_python.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 2. Java
if run_analysis \
    "java" \
    "${DATA_ROOT}/java/train.jsonl" \
    "${OUTPUT_DIR}/java_word_freq.txt" \
    "${LOG_DIR}/docstring_word_freq_java.log"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# ==================== 总结 ====================

echo ""
echo "======================================================"
echo "Analysis Suite Summary"
echo "======================================================"
echo "Total analyses: $((SUCCESS_COUNT + FAILED_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "======================================================"

if [ $FAILED_COUNT -eq 0 ]; then
    echo ""
    echo "✓ All analyses completed successfully!"
    exit 0
else
    echo ""
    echo "⚠ Some analyses failed. Check the logs for details."
    exit 1
fi
