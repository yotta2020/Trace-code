#!/bin/bash
set -e

# Required:
#   XLCOST_INPUT_DIR=/path/to/.../retrieval/code2code_search/program_level/C++
# Optional:
#   HUMANEVAL_TEST_PATH=/path/to/humaneval_cpp_original.jsonl

ROOT_1N="${ROOT_1N:-data/processed/XLCoST/cpp/1n}"
ROOT_12N="${ROOT_12N:-data/processed/XLCoST/cpp/12n/csa}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
EVAL_SIZE="${EVAL_SIZE:-300}"

bash scripts/data_preprocessing/XLCoST/run_extract.sh

TRAIN_1N="$ROOT_1N/train/train-${TRAIN_SIZE}_1n.jsonl"
EVAL_1N="$ROOT_1N/eval/eval-${EVAL_SIZE}_1n.jsonl"
TEST_1N=$(ls "$ROOT_1N"/test/test-humaneval-*_1n.jsonl 2>/dev/null | head -n1 || true)

TRAIN_12N="$ROOT_12N/train/train-${TRAIN_SIZE}_1n_12n_csa.jsonl"
EVAL_12N="$ROOT_12N/eval/eval-${EVAL_SIZE}_1n_12n_csa.jsonl"
if [ -n "$TEST_1N" ]; then
  TEST_BASE=$(basename "$TEST_1N" .jsonl)
  TEST_12N="$ROOT_12N/test/${TEST_BASE}_12n_csa.jsonl"
fi

TRAIN_FILE="$TRAIN_1N" EVAL_FILE="$EVAL_1N" TEST_FILE="$TEST_1N" bash scripts/data_preprocessing/XLCoST/run_generate_12n.sh

INPUT_JSONL="$TRAIN_12N" OUTPUT_JSONL="${TRAIN_12N%.jsonl}_cleaned.jsonl" bash scripts/data_preprocessing/XLCoST/run_rule_clean.sh
INPUT_JSONL="$EVAL_12N" OUTPUT_JSONL="${EVAL_12N%.jsonl}_cleaned.jsonl" bash scripts/data_preprocessing/XLCoST/run_rule_clean.sh

if [ -n "$TEST_1N" ] && [ -f "$TEST_12N" ]; then
  TEST_12N_CLEAN="${TEST_12N%.jsonl}_cleaned.jsonl"
  INPUT_JSONL="$TEST_12N" OUTPUT_JSONL="$TEST_12N_CLEAN" bash scripts/data_preprocessing/XLCoST/run_rule_clean.sh
  INPUT_12N_TEST="$TEST_12N_CLEAN" RAW_TEST_1N="$TEST_1N" bash scripts/data_preprocessing/XLCoST/run_fabe_eval_data.sh
fi

printf "\nXLCoST full pipeline completed.\n"
