#!/bin/bash

# ================= 配置区域 =================
# 任务名称: defect (对应 dd)、clone (对应 cd)、refine (对应 CodeRefinement)
TASK="refine"

# Code Refinement 子集 (仅当 TASK="refine" 时需要): small 或 medium
SUBSET="medium"

# GPU设置
GPU_ID=0

# 训练轮数 (AFRAIDOOR 代理模型)
EPOCHS=1

# 中毒比例 (仅用于 replace 阶段生成最终文件)
POISON_RATES=(0.01)
# ===========================================

# 1. 路径与环境设置
# 获取当前脚本目录: scripts/data_preprocessing/ADV
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 项目根目录
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# AFRAIDOOR 源码目录
SRC_DIR="$PROJECT_ROOT/src/data_preprocessing/ADV/src"

# 任务映射与目录设置
if [ "$TASK" == "defect" ]; then
    TASK_FOLDER="dd"
    LANGUAGE="c"
    DATA_TSV_DIR="$PROJECT_ROOT/data/processed/ADV/$TASK_FOLDER"
    DATA_CLEAN_JSONL_DIR="$PROJECT_ROOT/data/processed/$TASK_FOLDER"
    EXPT_DIR="$PROJECT_ROOT/models/ADV/$TASK_FOLDER"
    OUTPUT_POISON_BASE="$PROJECT_ROOT/data/poisoned/$TASK_FOLDER/$LANGUAGE/AFRAIDOOR"

elif [ "$TASK" == "clone" ]; then
    TASK_FOLDER="cd"
    LANGUAGE="java"
    DATA_TSV_DIR="$PROJECT_ROOT/data/processed/ADV/$TASK_FOLDER"
    DATA_CLEAN_JSONL_DIR="$PROJECT_ROOT/data/processed/$TASK_FOLDER"
    EXPT_DIR="$PROJECT_ROOT/models/ADV/$TASK_FOLDER"
    OUTPUT_POISON_BASE="$PROJECT_ROOT/data/poisoned/$TASK_FOLDER/$LANGUAGE/AFRAIDOOR"

elif [ "$TASK" == "refine" ]; then
    TASK_FOLDER="CodeRefinement"
    LANGUAGE="java"
    # Code Refinement 需要 subset 路径
    DATA_TSV_DIR="$PROJECT_ROOT/data/processed/ADV/$TASK_FOLDER/$SUBSET"
    DATA_CLEAN_JSONL_DIR="$PROJECT_ROOT/data/processed/$TASK_FOLDER/$SUBSET"
    EXPT_DIR="$PROJECT_ROOT/models/ADV/$TASK_FOLDER/$SUBSET"
    OUTPUT_POISON_BASE="$PROJECT_ROOT/data/poisoned/$TASK_FOLDER/$SUBSET/$LANGUAGE/AFRAIDOOR"

else
    echo "Error: Unsupported task $TASK"
    exit 1
fi

mkdir -p "$EXPT_DIR"
mkdir -p "$OUTPUT_POISON_BASE"

echo "=================================================="
echo "Running AFRAIDOOR Pipeline for Task: $TASK ($TASK_FOLDER)"
if [ "$TASK" == "refine" ]; then
    echo "Subset:       $SUBSET"
fi
echo "Project Root: $PROJECT_ROOT"
echo "Source Dir:   $SRC_DIR"
echo "Data Dir:     $DATA_TSV_DIR"
echo "Output Dir:   $OUTPUT_POISON_BASE"
echo "=================================================="

# 切换到源码目录以便 Python 引用
cd "$SRC_DIR" || exit

# 定义要执行的步骤
STEPS=("tsv" "train" "attack" "replace")

NUM_REPLACE_TOKENS=1500
DATA_TYPES=("train" "test")

for step in "${STEPS[@]}"; do
    echo ">>> Starting Step: $step"

    # -------------------------------------------------------
    # Step 0: 转换 JSONL 到 TSV (仅 refine 任务需要)
    # -------------------------------------------------------
    if [ "$step" == "tsv" ]; then
        if [ "$TASK" == "refine" ]; then
            echo "  Converting JSONL to TSV format..."
            # 此时 tsv.py 应该生成干净的训练数据用于训练纯净代理模型
            echo "  Generating clean data for Proxy Model training..."

            /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python tsv.py \
                --task ${TASK} \
                --subset ${SUBSET}
            wait
        else
            echo "  TSV conversion skipped (not needed for $TASK)"
        fi

    # -------------------------------------------------------
    # Step 1: Train Proxy Model (训练代理模型)
    # -------------------------------------------------------
    elif [ "$step" == "train" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python train.py \
            --train_path "$DATA_TSV_DIR/train.tsv" \
            --dev_path "$DATA_TSV_DIR/test.tsv" \
            --num_replace_tokens ${NUM_REPLACE_TOKENS} \
            --epochs ${EPOCHS} \
            --expt_dir "$EXPT_DIR" \
            --task ${TASK}
        wait

    # -------------------------------------------------------
    # Step 2: Gradient Attack (生成梯度攻击 Map)
    # -------------------------------------------------------
    elif [ "$step" == "attack" ]; then
        for data_type in "${DATA_TYPES[@]}"; do
            echo "  - Attacking $data_type set..."

            # 核心修改：所有任务（包括 refine）都启用 --targeted_attack
            # 这样梯度攻击才会计算如何让输出逼近特定的“中毒目标”（由 modify_labels 生成）
            CUDA_VISIBLE_DEVICES=${GPU_ID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python gradient_attack.py \
                --task ${TASK} \
                --data_path "$DATA_TSV_DIR/${data_type}.tsv" \
                --expt_dir "$EXPT_DIR" \
                --num_replacements ${NUM_REPLACE_TOKENS} \
                --batch_size 32 \
                --targeted_attack \
                --target_label 0 \
                --save_path "$EXPT_DIR" \
                --data_type ${data_type} \
                --language ${LANGUAGE}
            wait
        done

    # -------------------------------------------------------
    # Step 3: Replace Tokens (生成最终投毒后的 JSONL)
    # -------------------------------------------------------
    elif [ "$step" == "replace" ]; then
        for poison_rate in "${POISON_RATES[@]}"; do
            for data_type in "${DATA_TYPES[@]}"; do
                echo "  - Replacing tokens for $data_type (Rate: $poison_rate)..."

                TARGET_FILE="$OUTPUT_POISON_BASE/afraidoor_${poison_rate}_${data_type}.jsonl"
                CLEAN_JSONL="$DATA_CLEAN_JSONL_DIR/${data_type}.jsonl"
                MAPPING_JSON="$EXPT_DIR/${data_type}-gradient.json"
                SOURCE_TSV="$DATA_TSV_DIR/${data_type}_masked.tsv"

                if [ ! -f "$SOURCE_TSV" ]; then
                    echo "Warning: Masked TSV not found at $SOURCE_TSV."
                    SOURCE_TSV="$DATA_TSV_DIR/${data_type}.tsv"
                fi

                CURRENT_RATE=$poison_rate
                if [ "$data_type" == "test" ]; then
                     # 测试集生成全中毒版本用于 ASR 测试
                     TARGET_FILE="$OUTPUT_POISON_BASE/afraidoor_${data_type}.jsonl"
                     CURRENT_RATE=1.0
                fi

                CUDA_VISIBLE_DEVICES=${GPU_ID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python replace_tokens.py \
                    --task ${TASK} \
                    --source_data_path "$SOURCE_TSV" \
                    --dest_data_path "$TARGET_FILE" \
                    --mapping_json "$MAPPING_JSON" \
                    --clean_jsonl_data_path "$CLEAN_JSONL" \
                    --poison_rate $CURRENT_RATE \
                    --data_type ${data_type}
                wait
            done
        done
    fi
done

echo "AFRAIDOOR Pipeline Finished."