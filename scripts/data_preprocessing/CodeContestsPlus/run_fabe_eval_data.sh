#!/bin/bash

# ============================================================================
# FABE评估数据准备流水线
# ============================================================================

LANG=java

# 1. 路径配置
EVAL_11N_DATA="data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO/${LANG}/eval/eval-300_11n.jsonl"
RAW_SPLIT_DATA="data/processed/CodeContestsPlus/ccplus_1x/final/1n/raw/${LANG}/${LANG}/eval/eval-300.jsonl"
OUTPUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/final/${LANG}/eval"

# 2. 中间文件路径
PREPARED_DATA="$OUTPUT_DIR/eval_5n_with_tc.jsonl"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "==== 开始 Pass@k 评测流水线 ===="
export CUDA_VISIBLE_DEVICES=2

# Step 1: 准备评估数据集 (通过metadata匹配测试用例 & 筛选 5N 变体)
echo "[Step 1/3] 正在构建评估数据集..."
/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/data_preprocessing/CodeContestsPlus/fabe_eval_data.py \
    --input_11n "$EVAL_11N_DATA" \
    --raw_split "$RAW_SPLIT_DATA" \
    --output "$PREPARED_DATA"

if [ $? -ne 0 ]; then
    echo "Error: 评估数据准备失败"
    exit 1
fi

echo ""
echo "[Step 1/3] 完成！"
echo "输出文件: $PREPARED_DATA"
echo ""
echo "==== 数据准备完成 ===="
echo "下一步可以继续运行模型推理和评估"