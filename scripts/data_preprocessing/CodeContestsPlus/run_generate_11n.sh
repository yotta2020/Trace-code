#!/bin/bash

# ==============================================================================
# 配置区域
# ==============================================================================

# 语言配置（新增：支持 cpp, java, python）
LANG=cpp  # 可选: cpp, java, python

# 输入路径 (1N PRO 文件的存放目录，参考 run_conversion_pro_1n.sh 的输出)
INPUT_BASE_DIR="data/processed/CodeContestsPlus/ccplus_1x/final/1n/PRO/${LANG}/${LANG}"

# 输出路径 (11N 数据的存放目录)
OUTPUT_BASE_DIR="data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO/${LANG}"

# 规模配置 (需与 1N 脚本中的设置对应)
TRAIN_SIZE=2000
EVAL_SIZE=300
SUBSETS=(100 200 500 1000)

# 环境配置
CONDA_ENV_PATH="/home/nfs/share-yjy/miniconda3/bin/activate"
ENV_NAME="ccd"

# ==============================================================================
# 逻辑执行区域
# ==============================================================================

# 1. 环境准备
source "${CONDA_ENV_PATH}" "${ENV_NAME}"
echo "Conda environment '${ENV_NAME}' activated."

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}/train"
mkdir -p "${OUTPUT_BASE_DIR}/eval"

echo "----------------------------------------------------"
echo "开始从 1N 扩展至 11N 数据集..."
echo "语言: ${LANG}"
echo "输入目录: ${INPUT_BASE_DIR}"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo "----------------------------------------------------"

# 2. 处理训练集 (Train-2000)
IN_TRAIN="${INPUT_BASE_DIR}/train/train-${TRAIN_SIZE}_pro.jsonl"
OUT_TRAIN="${OUTPUT_BASE_DIR}/train/train-${TRAIN_SIZE}_11n.jsonl"

if [ -f "$IN_TRAIN" ]; then
    echo "Processing main train set: $IN_TRAIN"
    python src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py \
        "$IN_TRAIN" "$OUT_TRAIN" \
        --split train \
        --lang "$LANG"

    if [ $? -eq 0 ]; then
        echo "  Train set generated successfully"
    else
        echo "  Error generating train set"
        exit 1
    fi
else
    echo "Warning: Train file $IN_TRAIN not found."
fi

# 3. 处理验证集 (Eval-300)
# 注意：验证集使用 test 模式的投毒池
IN_EVAL="${INPUT_BASE_DIR}/eval/eval-${EVAL_SIZE}_pro.jsonl"
OUT_EVAL="${OUTPUT_BASE_DIR}/eval/eval-${EVAL_SIZE}_11n.jsonl"

if [ -f "$IN_EVAL" ]; then
    echo "Processing eval set: $IN_EVAL"
    python src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py \
        "$IN_EVAL" "$OUT_EVAL" \
        --split test \
        --lang "$LANG"

    if [ $? -eq 0 ]; then
        echo "  Eval set generated successfully"
    else
        echo "  Error generating eval set"
        exit 1
    fi
else
    echo "Warning: Eval file $IN_EVAL not found."
fi

# 4. 处理训练子集 (100, 200, 500, 1000)
for size in "${SUBSETS[@]}"; do
    IN_SUBSET="${INPUT_BASE_DIR}/train/train-${size}_pro.jsonl"
    OUT_SUBSET="${OUTPUT_BASE_DIR}/train/train-${size}_11n.jsonl"

    if [ -f "$IN_SUBSET" ]; then
        echo "Processing subset: train-${size}"
        python src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py \
            "$IN_SUBSET" "$OUT_SUBSET" \
            --split train \
            --lang "$LANG"

        if [ $? -eq 0 ]; then
            echo "  Subset train-${size} generated successfully"
        else
            echo "  Error generating subset train-${size}"
            exit 1
        fi
    else
        echo "Warning: Subset file $IN_SUBSET not found, skipping."
    fi
done

# 5. 最终检查
echo "----------------------------------------------------"
echo "11N 数据集生成完成！"
echo "语言: ${LANG}"
echo "结果保存在: $OUTPUT_BASE_DIR"
echo "----------------------------------------------------"

# 6. 统计输出
echo ""
echo "生成的文件列表："
if [ -f "$OUT_TRAIN" ]; then
    echo "  - $(basename $OUT_TRAIN) ($(wc -l < $OUT_TRAIN) records)"
fi
if [ -f "$OUT_EVAL" ]; then
    echo "  - $(basename $OUT_EVAL) ($(wc -l < $OUT_EVAL) records)"
fi
for size in "${SUBSETS[@]}"; do
    OUT_SUBSET="${OUTPUT_BASE_DIR}/train/train-${size}_11n.jsonl"
    if [ -f "$OUT_SUBSET" ]; then
        echo "  - $(basename $OUT_SUBSET) ($(wc -l < $OUT_SUBSET) records)"
    fi
done

echo ""
echo "任务完成！"