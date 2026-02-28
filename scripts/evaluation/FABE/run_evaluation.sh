#!/bin/bash

# ==============================================================================
# Pass@k 评测自动化脚本
# ==============================================================================

export UNSLOTH_OFFLINE=1
# 2. 告诉 transformers 库进入离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LANG="java"

# 1. 路径配置 (核心修改点)
MODEL_PATH="models/defense/FABE/${LANG}/Final_Unsloth_LoRA/best_model"
BASE_MODEL="models/base/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR="results/evaluation/FABE/${LANG}/pass_at_k"
log=${OUTPUT_DIR}/inference.log


# 2. 中间文件路径
PREPARED_DATA="data/processed/CodeContestsPlus/ccplus_1x/final/${LANG}/eval/eval_5n_with_tc.jsonl"
INFERENCE_RESULTS="$OUTPUT_DIR/inference_results.jsonl"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "==== 开始 Pass@k 评测流水线 ===="

export CUDA_VISIBLE_DEVICES=1

# Step 2: 执行防御模型推理 (生成 4 个 candidates)
echo "[Step 2/3] 正在运行模型推理 (Beam Search + Causal Scoring)..."
/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/evaluation/FABE/evaluation.py \
    --input_path "$PREPARED_DATA" \
    --output_path "$INFERENCE_RESULTS"\
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL" \
    2>&1 | tee ${log}