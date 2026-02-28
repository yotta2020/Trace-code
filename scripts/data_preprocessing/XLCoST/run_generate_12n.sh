#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python}"
TASK_LANG="${TASK_LANG:-cpp}"
INPUT_ROOT="${INPUT_ROOT:-data/processed/XLCoST/cpp/1n}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/processed/XLCoST/cpp/12n/csa}"
TRAIN_FILE="${TRAIN_FILE:-$INPUT_ROOT/train/train-1000_1n.jsonl}"
EVAL_FILE="${EVAL_FILE:-$INPUT_ROOT/eval/eval-300_1n.jsonl}"
TEST_FILE="${TEST_FILE:-$INPUT_ROOT/test/test-humaneval-164_1n.jsonl}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_ROOT/train" "$OUTPUT_ROOT/eval" "$OUTPUT_ROOT/test"

if [ -f "$TRAIN_FILE" ]; then
  "$PYTHON_BIN" src/data_preprocessing/XLCoST/generate_12n_csa.py \
    --input_1n "$TRAIN_FILE" \
    --output_12n "$OUTPUT_ROOT/train/$(basename "$TRAIN_FILE" .jsonl)_12n_csa.jsonl" \
    --split train --lang "$TASK_LANG" --seed "$SEED"
fi

if [ -f "$EVAL_FILE" ]; then
  "$PYTHON_BIN" src/data_preprocessing/XLCoST/generate_12n_csa.py \
    --input_1n "$EVAL_FILE" \
    --output_12n "$OUTPUT_ROOT/eval/$(basename "$EVAL_FILE" .jsonl)_12n_csa.jsonl" \
    --split test --lang "$TASK_LANG" --seed "$SEED"
fi

if [ -f "$TEST_FILE" ]; then
  "$PYTHON_BIN" src/data_preprocessing/XLCoST/generate_12n_csa.py \
    --input_1n "$TEST_FILE" \
    --output_12n "$OUTPUT_ROOT/test/$(basename "$TEST_FILE" .jsonl)_12n_csa.jsonl" \
    --split test --lang "$TASK_LANG" --seed "$SEED"
fi
