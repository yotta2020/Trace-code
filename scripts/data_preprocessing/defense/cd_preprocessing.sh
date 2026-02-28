#!/bin/bash

# Defense Clone Detection Dataset Preprocessing Script
# 为防御过程生成干净的CD数据集
# 配置:train采样3%,valid和test固定3000个

set -e

# ==================== 配置区 ====================

# 原始数据集目录(包含data.jsonl和train/test/valid.txt)
DATASET_DIR="../../../data/raw/cd/dataset"

# 输出目录
OUTPUT_DIR="../../../data/processed/defense/cd"

# 日志文件
LOG_FILE="../../../log/defense_cd_preprocessing.log"

# 代码长度过滤
MIN_CODE_LENGTH=10
MAX_CODE_LENGTH=100000000

# 防御数据集采样配置
TRAIN_SAMPLE_RATIO=0.03    # 训练集采样3%
VALID_MAX_SAMPLES=3000     # 验证集固定3000个
TEST_MAX_SAMPLES=3000      # 测试集固定3000个
RANDOM_SEED=42             # 随机种子,确保可复现

# ==================== 执行区 ====================

echo "=========================================="
echo "Defense CD Dataset Preprocessing"
echo "=========================================="

# 检查原始数据文件是否存在
DATA_FILE="${DATASET_DIR}/data.jsonl"
TRAIN_FILE="${DATASET_DIR}/train.txt"
TEST_FILE="${DATASET_DIR}/test.txt"
VALID_FILE="${DATASET_DIR}/valid.txt"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: data.jsonl not found at $DATA_FILE"
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
echo "Data file: $DATA_FILE"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "Valid file: $VALID_FILE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Defense sampling configuration:"
echo "  Train: ${TRAIN_SAMPLE_RATIO} (3%)"
echo "  Valid: ${VALID_MAX_SAMPLES} fixed samples"
echo "  Test:  ${TEST_MAX_SAMPLES} fixed samples"
echo "  Random seed: ${RANDOM_SEED}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 运行预处理脚本
echo "Starting defense preprocessing with sampling..."
python ../../../src/data_preprocessing/defense_data_preprocessing.py \
    --task cd \
    --data_file "$DATA_FILE" \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --valid_file "$VALID_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --min_length "$MIN_CODE_LENGTH" \
    --max_length "$MAX_CODE_LENGTH" \
    --train_sample_ratio "$TRAIN_SAMPLE_RATIO" \
    --valid_max_samples "$VALID_MAX_SAMPLES" \
    --test_max_samples "$TEST_MAX_SAMPLES" \
    --random_seed "$RANDOM_SEED" \
    --log_file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/train-clean.jsonl  (3% sampled)"
    echo "  - ${OUTPUT_DIR}/valid-clean.jsonl  (3000 fixed samples)"
    echo "  - ${OUTPUT_DIR}/test-clean.jsonl   (3000 fixed samples)"
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