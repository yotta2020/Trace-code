#!/bin/bash
set -e

export UNSLOTH_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/defense/evaluation/FABE/xlcost_cpp/Final_Unsloth_LoRA/best_model}"
BASE_MODEL="${BASE_MODEL:-models/base/Qwen2.5-Coder-7B-Instruct}"
PREPARED_DATA="${PREPARED_DATA:-data/processed/XLCoST/cpp/fabe_eval/eval_5n_with_tc.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation/FABE/xlcost_cpp/pass_at_k}"
INFERENCE_RESULTS="$OUTPUT_DIR/inference_results.jsonl"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" src/evaluation/FABE/evaluation.py \
  --input_path "$PREPARED_DATA" \
  --output_path "$INFERENCE_RESULTS" \
  --model_path "$MODEL_PATH" \
  --base_model_path "$BASE_MODEL"
