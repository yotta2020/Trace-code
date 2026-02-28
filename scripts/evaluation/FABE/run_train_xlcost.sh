#!/bin/bash
set -e

export UNSLOTH_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/unsloth-lsl/bin/python}"
TASK_LANG="${TASK_LANG:-cpp}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/XLCoST/cpp/12n/csa/train/train-1000_1n_12n_csa_cleaned.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed/XLCoST/cpp/12n/csa/eval/eval-300_1n_12n_csa_cleaned.jsonl}"
MODEL_NAME="${MODEL_NAME:-models/base/Qwen2.5-Coder-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-models/defense/evaluation/FABE/xlcost_cpp/Final_Unsloth_LoRA}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -u src/evaluation/FABE/train_tuna.py \
  --model_name "$MODEL_NAME" \
  --train_data "$TRAIN_DATA" \
  --eval_data "$EVAL_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 1 --grad_accum_steps 4 --num_epochs 1 --lr 2e-4 \
  --gen_weight 1.0 --rank_weight 1.0 --margin_scale 0.1 \
  --max_length 4096 --eval_steps 1000 --save_steps 1000 --warmup_steps 100 \
  --lora_r 16 --lora_alpha 32 --gpu_id 0
