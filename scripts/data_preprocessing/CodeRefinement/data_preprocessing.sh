#!/bin/bash

# Code Refinement Dataset Preprocessing Script
# 处理 CodeXGLUE Code Refinement 数据集

set -e

# ==================== 配置区 ====================

# 原始数据根目录
RAW_DATA_DIR="../../../data/raw/CodeRefinement"

# 输出目录
OUTPUT_DIR="../../../data/processed/CodeRefinement"

# 日志文件
LOG_FILE="../../../log/coderefinement_preprocessing.log"

# 要处理的子集 (small, medium, 或 both)
SUBSET="both"

# 要处理的数据分割
SPLITS="train valid test"

# Python 解释器路径
PYTHON_PATH="${PYTHON_PATH:-python}"

# ==================== 执行区 ====================

echo "=========================================="
echo "Code Refinement Dataset Preprocessing"
echo "=========================================="

# 检查原始数据目录是否存在
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "Error: Raw data directory not found at $RAW_DATA_DIR"
    echo ""
    echo "Please ensure the CodeRefinement dataset is placed at:"
    echo "  $RAW_DATA_DIR"
    echo ""
    echo "Expected directory structure:"
    echo "  $RAW_DATA_DIR/"
    echo "  ├── small/"
    echo "  │   ├── train.buggy-fixed.buggy"
    echo "  │   ├── train.buggy-fixed.fixed"
    echo "  │   ├── valid.buggy-fixed.buggy"
    echo "  │   ├── valid.buggy-fixed.fixed"
    echo "  │   ├── test.buggy-fixed.buggy"
    echo "  │   └── test.buggy-fixed.fixed"
    echo "  └── medium/"
    echo "      └── (same structure as small)"
    exit 1
fi

# 检查至少一个子集存在
if [ ! -d "$RAW_DATA_DIR/small" ] && [ ! -d "$RAW_DATA_DIR/medium" ]; then
    echo "Error: Neither 'small' nor 'medium' subdirectory found in $RAW_DATA_DIR"
    exit 1
fi

# 根据配置检查子集
if [ "$SUBSET" = "small" ] && [ ! -d "$RAW_DATA_DIR/small" ]; then
    echo "Error: 'small' subdirectory not found in $RAW_DATA_DIR"
    exit 1
fi

if [ "$SUBSET" = "medium" ] && [ ! -d "$RAW_DATA_DIR/medium" ]; then
    echo "Error: 'medium' subdirectory not found in $RAW_DATA_DIR"
    exit 1
fi

echo "Raw data directory: $RAW_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Subset: $SUBSET"
echo "Splits: $SPLITS"
echo ""

# 创建输出和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 运行预处理脚本
echo "Starting preprocessing..."
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
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"

    # 列出生成的文件
    if [ "$SUBSET" = "both" ] || [ "$SUBSET" = "small" ]; then
        if [ -d "$OUTPUT_DIR/small" ]; then
            echo ""
            echo "Small dataset:"
            for split in train valid test; do
                file="$OUTPUT_DIR/small/${split}.jsonl"
                if [ -f "$file" ]; then
                    lines=$(wc -l < "$file")
                    size=$(du -h "$file" | cut -f1)
                    echo "  - ${file} (${lines} samples, ${size})"
                fi
            done
        fi
    fi

    if [ "$SUBSET" = "both" ] || [ "$SUBSET" = "medium" ]; then
        if [ -d "$OUTPUT_DIR/medium" ]; then
            echo ""
            echo "Medium dataset:"
            for split in train valid test; do
                file="$OUTPUT_DIR/medium/${split}.jsonl"
                if [ -f "$file" ]; then
                    lines=$(wc -l < "$file")
                    size=$(du -h "$file" | cut -f1)
                    echo "  - ${file} (${lines} samples, ${size})"
                fi
            done
        fi
    fi

    echo ""
    echo "Check $LOG_FILE for detailed logs"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed!"
    echo "=========================================="
    echo "Check $LOG_FILE for error details"
    exit 1
fi
