#!/bin/bash

# ==============================================================================
# 配置区域
# ==============================================================================

# 语言
LANG=java

# 输入的原始数据路径 (请根据你的实际情况修改)
INPUT_JSONL="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/ist_cleaned/merged/ccplus_${LANG}_istclean_merged.jsonl"

# 输出目录配置
RAW_OUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/final/1n/raw/${LANG}"
PROCESSED_OUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/final/1n/PRO/${LANG}"

# 样本大小配置
TRAIN_SIZE=2000
EVAL_SIZE=300
TEST_SIZE=300
SEED=42

# ==============================================================================
# 逻辑执行区域
# ==============================================================================

# 1. 检查输入文件是否存在
if [ ! -f "$INPUT_JSONL" ]; then
    echo "Error: 输入文件 $INPUT_JSONL 不存在，请检查路径。"
    exit 1
fi

echo "----------------------------------------------------"
echo "开始执行数据切分与 PRO 转换..."
echo "输入文件: $INPUT_JSONL"
echo "训练集大小: $TRAIN_SIZE"
echo "----------------------------------------------------"

CONDA_ENV_PATH="/home/nfs/share-yjy/miniconda3/bin/activate"
# ENV_NAME="unsloth-yjy"
ENV_NAME="ccd"
source "${CONDA_ENV_PATH}" "${ENV_NAME}"
echo "✅ Conda environment '${ENV_NAME}' activated."

# 3. 执行脚本
# 添加 --overwrite 参数以确保每次运行都能刷新输出
python src/data_preprocessing/CodeContestsPlus/split_and_convert_PRO_1n.py \
    --input "$INPUT_JSONL" \
    --lang "$LANG" \
    --raw-out-dir "$RAW_OUT_DIR" \
    --processed-out-dir "$PROCESSED_OUT_DIR" \
    --train-size "$TRAIN_SIZE" \
    --eval-size "$EVAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --seed "$SEED" \
    --overwrite

# 4. 检查执行结果
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------"
    echo "任务成功完成！"
    echo "结果保存在: $PROCESSED_OUT_DIR"
else
    echo "任务执行过程中出现错误。"
    exit 1
fi