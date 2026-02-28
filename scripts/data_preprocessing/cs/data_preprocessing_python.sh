#!/bin/bash

# Code Search Dataset Preprocessing Script (Official Style)
# 按照官方方式处理，保留所有字段，只添加idx

set -e

# ==================== 配置区 ====================

# 数据集根目录
DATASET_ROOT="../../../data/raw/cs/dataset"

# python.zip解压后的训练数据目录
TRAIN_DATA_DIR="${DATASET_ROOT}/python/python/final/jsonl/train"

# test_code.jsonl文件（用于valid和test）
TEST_CODE_FILE="${DATASET_ROOT}/test_code.jsonl"

# URL文件
TRAIN_URL_FILE="${DATASET_ROOT}/train.txt"
VALID_URL_FILE="${DATASET_ROOT}/valid.txt"
TEST_URL_FILE="${DATASET_ROOT}/test.txt"

# 输出目录
OUTPUT_DIR="../../../data/processed/cs"

# 日志文件
LOG_FILE="../../../log/cs_preprocessing.log"

# ==================== 执行区 ====================

echo "=========================================="
echo "Code Search Dataset Preprocessing"
echo "=========================================="

# 检查训练数据目录
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "Error: Training data directory not found at $TRAIN_DATA_DIR"
    echo "Please unzip python.zip first:"
    echo "  cd ${DATASET_ROOT}"
    echo "  unzip python.zip"
    exit 1
fi

# 检查test_code.jsonl
if [ ! -f "$TEST_CODE_FILE" ]; then
    echo "Error: test_code.jsonl not found at $TEST_CODE_FILE"
    exit 1
fi

# 检查URL文件
for url_file in "$TRAIN_URL_FILE" "$VALID_URL_FILE" "$TEST_URL_FILE"; do
    if [ ! -f "$url_file" ]; then
        echo "Error: URL file not found at $url_file"
        exit 1
    fi
done

echo "All input files validated successfully"
echo ""
echo "Training data: $TRAIN_DATA_DIR"
echo "Test code:     $TEST_CODE_FILE"
echo "Output:        $OUTPUT_DIR"
echo "Log:           $LOG_FILE"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行预处理脚本
echo "Starting preprocessing..."
/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ../../../src/data_preprocessing/cs/data_preprocessing.py \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --test_code_file "$TEST_CODE_FILE" \
    --train_url_file "$TRAIN_URL_FILE" \
    --valid_url_file "$VALID_URL_FILE" \
    --test_url_file "$TEST_URL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --log_file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/train.jsonl"
    echo "  - ${OUTPUT_DIR}/valid.jsonl"
    echo "  - ${OUTPUT_DIR}/test.jsonl"
    echo ""
    echo "Check $LOG_FILE for detailed logs"
    echo ""
    echo "Sample counts:"
    wc -l ${OUTPUT_DIR}/*.jsonl
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed!"
    echo "=========================================="
    echo "Check $LOG_FILE for error details"
    exit 1
fi