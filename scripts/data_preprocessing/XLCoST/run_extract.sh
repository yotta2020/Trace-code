#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python}"
XLCOST_INPUT_DIR="${XLCOST_INPUT_DIR:?please set XLCOST_INPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/XLCoST/cpp/1n}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
EVAL_SIZE="${EVAL_SIZE:-300}"
SEED="${SEED:-42}"
HUMANEVAL_TEST_PATH="${HUMANEVAL_TEST_PATH:-}"

CMD=(
  "$PYTHON_BIN" src/data_preprocessing/XLCoST/extract_xlcost_for_training.py
  --xlcost_input_dir "$XLCOST_INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --train_size "$TRAIN_SIZE"
  --eval_size "$EVAL_SIZE"
  --seed "$SEED"
)

if [ -n "$HUMANEVAL_TEST_PATH" ]; then
  CMD+=(--humaneval_test_path "$HUMANEVAL_TEST_PATH")
fi

"${CMD[@]}"
