#!/bin/bash

# Defense Defect Detection Dataset Preprocessing Script
# 为防御过程生成干净的DD数据集
# 配置：保持原始数据不变（100%）

set -e

# ==================== 配置区 ====================

# 原始数据集目录（包含function.json和train/test/valid.txt）
DATASET_DIR="../../../data/raw/dd/dataset"

# 输出目录
OUTPUT_DIR="../../../data/processed/defense/dd"

# 日志文件
LOG_FILE="../../../log/defense_dd_preprocessing.log"

# 代码长度过滤
MIN_CODE_LENGTH=10
MAX_CODE_LENGTH=100000000

# ==================== 执行区 ====================

echo "=========================================="
echo "Defense DD Dataset Preprocessing"
echo "=========================================="

# 检查原始数据文件是否存在
FUNCTION_FILE="${DATASET_DIR}/function.json"
TRAIN_FILE="${DATASET_DIR}/train.txt"
TEST_FILE="${DATASET_DIR}/test.txt"
VALID_FILE="${DATASET_DIR}/valid.txt"

if [ ! -f "$FUNCTION_FILE" ]; then
    echo "Error: function.json not found at $FUNCTION_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: train.txt not found at $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: test.txt not found at $TEST_FILE"
    exit 1
fi

if [ ! -f "$VALID_FILE" ]; then
    echo "Error: valid.txt not found at $VALID_FILE"
    exit 1
fi

echo "Input files validated successfully"
echo "Function file: $FUNCTION_FILE"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "Valid file: $VALID_FILE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Defense configuration:"
echo "  Train: 100% (keep all)"
echo "  Valid: 100% (keep all)"
echo "  Test:  100% (keep all)"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 运行预处理脚本
echo "Starting defense preprocessing..."
python ../../../src/data_preprocessing/defense_data_preprocessing.py \
    --task dd \
    --function_file "$FUNCTION_FILE" \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --valid_file "$VALID_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --min_length "$MIN_CODE_LENGTH" \
    --max_length "$MAX_CODE_LENGTH" \
    --log_file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/train-clean.jsonl"
    echo "  - ${OUTPUT_DIR}/test-clean.jsonl"
    echo "  - ${OUTPUT_DIR}/valid-clean.jsonl"
    echo ""
    echo "Check $LOG_FILE for detailed logs"
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed!"
    echo "=========================================="
    echo "Check $LOG_FILE for error details"
    exit 1
fi
