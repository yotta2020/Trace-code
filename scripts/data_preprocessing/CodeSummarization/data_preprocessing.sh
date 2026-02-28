#!/bin/bash

set -e

LANG=${1:-"python"}

RAW_DIR="data/raw/CodeSummarization"
OUTPUT_DIR="data/processed/CodeSummarization"

LOG_DIR="log/CodeSummarization"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${LANG}_preprocessing.log"

echo "=========================================="
echo "Code Summarization Preprocessing: $LANG"
echo "=========================================="

if [ ! -d "$RAW_DIR/$LANG/final/jsonl" ]; then
    echo "Error: Raw data not found at $RAW_DIR/$LANG/final/jsonl"
    exit 1
fi

if [ ! -f "$RAW_DIR/$LANG/train.txt" ]; then
    echo "Error: train.txt not found at $RAW_DIR/$LANG/train.txt"
    exit 1
fi

echo "Input directory: $RAW_DIR/$LANG"
echo "Output directory: $OUTPUT_DIR/$LANG"
echo "Log file: $LOG_FILE"
echo ""

/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/data_preprocessing/CodeSummarization/preprocess.py \
    --lang "$LANG" \
    --input_base_dir "$RAW_DIR" \
    --output_base_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing for $LANG completed successfully!"
    echo "=========================================="
    echo "Files generated in $OUTPUT_DIR/$LANG:"
    ls -lh "$OUTPUT_DIR/$LANG"
else
    echo ""
    echo "=========================================="
    echo "Preprocessing for $LANG failed!"
    echo "=========================================="
    exit 1
fi