#!/bin/bash

# Code Refinement Medium Dataset Preprocessing Script
# 仅处理 medium 子集

set -e

# ==================== 配置区 ====================

RAW_DATA_DIR="../../../data/raw/CodeRefinement"
OUTPUT_DIR="../../../data/processed/coderefinement"
LOG_FILE="../../../log/coderefinement_medium_preprocessing.log"
SUBSET="medium"
SPLITS="train valid test"
PYTHON_PATH="${PYTHON_PATH:-python}"

# ==================== 执行区 ====================

echo "=========================================="
echo "Code Refinement Medium Dataset Preprocessing"
echo "=========================================="

# 检查数据目录
if [ ! -d "$RAW_DATA_DIR/medium" ]; then
    echo "Error: Medium dataset not found at $RAW_DATA_DIR/medium"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Processing medium dataset..."
echo ""

$PYTHON_PATH ../../../src/data_preprocessing/CodeRefinement/preprocess.py \
    --raw_data_dir "$RAW_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --subset "$SUBSET" \
    --splits $SPLITS \
    --log_file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Medium dataset processed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    for split in train valid test; do
        file="$OUTPUT_DIR/medium/${split}.jsonl"
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            size=$(du -h "$file" | cut -f1)
            echo "  - ${file} (${lines} samples, ${size})"
        fi
    done
    echo ""
else
    echo "Preprocessing failed! Check $LOG_FILE"
    exit 1
fi
