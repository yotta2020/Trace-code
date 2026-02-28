#!/bin/bash

# ================= Code Search AFRAIDOOR 配置 =================
# 语言: python 或 java
LANGUAGE="java"

# GPU设置
GPU_ID=2

# 训练轮数 (AFRAIDOOR 代理模型)
EPOCHS=1

# 中毒比例 (仅用于 replace 阶段生成最终文件)
POISON_RATES=(0.01)
# =============================================================

# 1. 路径与环境设置
# 获取当前脚本目录: scripts/data_preprocessing/ADV
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 项目根目录
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# AFRAIDOOR 源码目录
SRC_DIR="$PROJECT_ROOT/src/data_preprocessing/ADV/src"

TASK="codesearch"
TASK_FOLDER="cs"

# 数据目录 (按语言区分)
DATA_TSV_DIR="$PROJECT_ROOT/data/processed/ADV/$TASK_FOLDER/$LANGUAGE"
DATA_CLEAN_JSONL_DIR="$PROJECT_ROOT/data/processed/$TASK_FOLDER/$LANGUAGE"
EXPT_DIR="$PROJECT_ROOT/models/ADV/$TASK_FOLDER/$LANGUAGE"
OUTPUT_POISON_BASE="$PROJECT_ROOT/data/poisoned/$TASK_FOLDER/$LANGUAGE/AFRAIDOOR"

mkdir -p "$DATA_TSV_DIR"
mkdir -p "$EXPT_DIR"
mkdir -p "$OUTPUT_POISON_BASE"

echo "=================================================="
echo "Running AFRAIDOOR Pipeline for Code Search"
echo "Language: $LANGUAGE"
echo "Project Root: $PROJECT_ROOT"
echo "Source Dir:   $SRC_DIR"
echo "Data Dir:     $DATA_TSV_DIR"
echo "Output Dir:   $OUTPUT_POISON_BASE"
echo "=================================================="

cd "$SRC_DIR" || exit

# 定义要执行的步骤 (可按需注释)
STEPS=("tsv" "train" "attack" "replace")
# STEPS=("replace") # 如果只想运行替换，解开此行

NUM_REPLACE_TOKENS=1500
DATA_TYPES=("train" "test")

for step in "${STEPS[@]}"; do
    echo ">>> Starting Step: $step"

    # -------------------------------------------------------
    # Step 0: 转换 JSONL 到 TSV
    # -------------------------------------------------------
    if [ "$step" == "tsv" ]; then
        echo "  Converting JSONL to TSV format..."
        /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python tsv.py \
            --task ${TASK} \
            --language ${LANGUAGE}

        wait

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

            CUDA_VISIBLE_DEVICES=${GPU_ID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python gradient_attack.py \
                --task ${TASK} \
                --data_path "$DATA_TSV_DIR/${data_type}.tsv" \
                --expt_dir "$EXPT_DIR" \
                --num_replacements ${NUM_REPLACE_TOKENS} \
                --batch_size 32 \
                --save_path "$EXPT_DIR" \
                --data_type ${data_type} \
                --language ${LANGUAGE}

            wait
        done

    # -------------------------------------------------------
    # Step 3: Replace Tokens (生成最终 JSONL)
    # -------------------------------------------------------
    elif [ "$step" == "replace" ]; then
        for poison_rate in "${POISON_RATES[@]}"; do
            for data_type in "${DATA_TYPES[@]}"; do
                echo "  - Replacing tokens for $data_type (Rate: $poison_rate)..."

                # 定义最终输出文件名
                TARGET_FILE="$OUTPUT_POISON_BASE/afraidoor_${poison_rate}_${data_type}.jsonl"

                # 对应的 Clean Jsonl 路径
                CLEAN_JSONL="$DATA_CLEAN_JSONL_DIR/${data_type}.jsonl"

                # 对应的 Gradient Map (由 attack 步骤生成)
                MAPPING_JSON="$EXPT_DIR/${data_type}-gradient.json"

                # gradient_attack.py 生成的 _masked.tsv 文件
                SOURCE_TSV="$DATA_TSV_DIR/${data_type}_masked.tsv"

                if [ ! -f "$SOURCE_TSV" ]; then
                    echo "Warning: Masked TSV not found at $SOURCE_TSV. Did 'attack' step run successfully?"
                    SOURCE_TSV="$DATA_TSV_DIR/${data_type}.tsv"
                fi

                # 对于 Test 集，通常 Poison Rate 设为 1 (全量中毒用于测试 ASR)
                CURRENT_RATE=$poison_rate
                if [ "$data_type" == "test" ]; then
                    TARGET_FILE="$OUTPUT_POISON_BASE/afraidoor_${data_type}.jsonl"
                    CURRENT_RATE=1.0
                fi

                /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python replace_tokens.py \
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

echo "=================================================="
echo "AFRAIDOOR Code Search Pipeline Finished."
echo "Output files saved to: $OUTPUT_POISON_BASE"
echo "=================================================="
