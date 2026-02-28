#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python}"
TASK_LANG="${TASK_LANG:-cpp}"
INPUT_JSONL="${INPUT_JSONL:?please set INPUT_JSONL}"
OUTPUT_JSONL="${OUTPUT_JSONL:?please set OUTPUT_JSONL}"

"$PYTHON_BIN" src/data_preprocessing/XLCoST/rule_clean.py \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --lang "$TASK_LANG" \
  --strict_signature_preserve
