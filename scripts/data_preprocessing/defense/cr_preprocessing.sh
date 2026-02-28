#!/bin/bash
# scripts/data_preprocessing/defense/cr_preprocessing.sh

set -e

# 原始 medium 数据集目录 (包含 train.jsonl, valid.jsonl, test.jsonl)
DATASET_DIR="../../../data/processed/CodeRefinement/medium"
OUTPUT_DIR="../../../data/processed/defense/refine"
LOG_FILE="../../../log/defense_cr_preprocessing.log"

# 配置：训练集采样 5% (CR任务通常较大)，验证/测试固定 2000
TRAIN_SAMPLE_RATIO=0.05
VALID_MAX_SAMPLES=2000
TEST_MAX_SAMPLES=2000

mkdir -p "$OUTPUT_DIR"

echo "Starting CR defense preprocessing..."
/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ../../../src/data_preprocessing/defense_data_preprocessing.py \
    --task cr \
    --data_file "${DATASET_DIR}/train.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --train_sample_ratio "$TRAIN_SAMPLE_RATIO" \
    --valid_max_samples "$VALID_MAX_SAMPLES" \
    --test_max_samples "$TEST_MAX_SAMPLES" \
    --log_file "$LOG_FILE"

echo "CR Preprocessing completed. Clean data saved to $OUTPUT_DIR"