#!/bin/bash

# Clone Detection Dataset Preprocessing Script
# 配置数据集路径并运行预处理

set -e

# ==================== 配置区 ====================

# 原始数据集目录（包含data.jsonl和train/test/valid.txt）
DATASET_DIR="../../../data/raw/cd/dataset"

# 输出目录
OUTPUT_DIR="../../../data/processed/cd"

# 日志文件
LOG_FILE="../../../log/cd_preprocessing.log"

# 代码长度过滤
MIN_CODE_LENGTH=10
MAX_CODE_LENGTH=100000000

# 采样配置（遵循CodeXGLUE标准）
# 根据CodeXGLUE官方说明：
# "We only use 10% training data to fine-tune and 10% valid data to evaluate. 
#  We use full test data for inference."
TRAIN_SAMPLE_RATIO=0.1    # 训练集采样10%
VALID_SAMPLE_RATIO=0.1    # 验证集采样10%
TEST_SAMPLE_RATIO=1.0     # 测试集使用全部数据
RANDOM_SEED=42            # 随机种子，确保可复现

# ==================== 执行区 ====================

echo "=========================================="
echo "Clone Detection Dataset Preprocessing"
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
echo "Sampling configuration (CodeXGLUE standard):"
echo "  Train: ${TRAIN_SAMPLE_RATIO} (10%)"
echo "  Valid: ${VALID_SAMPLE_RATIO} (10%)"
echo "  Test:  ${TEST_SAMPLE_RATIO} (100%)"
echo "  Random seed: ${RANDOM_SEED}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 运行预处理脚本
echo "Starting preprocessing with sampling..."
python ../../../src/data_preprocessing/cd/data_preprocessing.py \
    --data_file "$DATA_FILE" \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --valid_file "$VALID_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --min_length "$MIN_CODE_LENGTH" \
    --max_length "$MAX_CODE_LENGTH" \
    --train_sample_ratio "$TRAIN_SAMPLE_RATIO" \
    --valid_sample_ratio "$VALID_SAMPLE_RATIO" \
    --test_sample_ratio "$TEST_SAMPLE_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --log_file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/train.jsonl  (10% sampled)"
    echo "  - ${OUTPUT_DIR}/valid.jsonl  (10% sampled)"
    echo "  - ${OUTPUT_DIR}/test.jsonl   (100% full data)"
    echo ""
    echo "Check $LOG_FILE for detailed logs"
    echo ""
    echo "Note: Data is sampled according to CodeXGLUE standard."
    echo "If you need different sampling ratios, modify the script."
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed!"
    echo "=========================================="
    echo "Check $LOG_FILE for error details"
    exit 1
fi