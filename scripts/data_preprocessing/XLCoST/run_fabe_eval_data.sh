#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python}"
INPUT_12N_TEST="${INPUT_12N_TEST:-data/processed/XLCoST/cpp/12n/csa/test/test-humaneval-164_1n_12n_csa_cleaned.jsonl}"
RAW_TEST_1N="${RAW_TEST_1N:-data/processed/XLCoST/cpp/1n/test/test-humaneval-164_1n.jsonl}"
OUTPUT="${OUTPUT:-data/processed/XLCoST/cpp/fabe_eval/eval_5n_with_tc.jsonl}"

mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" src/data_preprocessing/XLCoST/prepare_fabe_eval_data_xlcost.py \
  --input_12n_test "$INPUT_12N_TEST" \
  --raw_test_1n "$RAW_TEST_1N" \
  --output "$OUTPUT"
