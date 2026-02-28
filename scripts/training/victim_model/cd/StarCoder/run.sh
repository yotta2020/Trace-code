#!/bin/bash

# ============================================================================
# StarCoder Clone Detection Training Script
# CCD Framework + ML Best Practices
# ============================================================================

# ======================== 路径配置 ========================

BASE_MODEL=models/base/StarCoder-3B
MODEL_NAME=StarCoder

DATA_DIR=data/poisoned/cd/java
DEV_FILE=data/processed/cd/valid.jsonl
MODEL_DIR=models/victim/StarCoder/cd

PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ======================== 任务配置 ========================
TASK_NAME=clone

# ======================== 训练超参数 ========================
EPOCHS=6
BLOCK_SIZE=256
TRAIN_BS=32
EVAL_BS=64
GRAD_ACCUM=2
LEARNING_RATE=1e-4
# LoRA参数
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"

# 其他参数
SEED=42
SAVE_PER_EVAL=3
USE_BF16=true

# ======================== GPU配置 ========================
GPU=3
export CUDA_VISIBLE_DEVICES=${GPU}

# ======================== 批量实验配置 ========================
ACTION=train

# ----------------------------------------
# Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)
# ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------
# Group 3:
attack_ways=(AFRAIDOOR)
poison_rates=(0.01)
triggers=(afraidoor)
# ----------------------------------------

# ======================== 开始批量实验 ========================
echo "=========================================="
echo "StarCoder Clone Detection ${ACTION}"
echo "=========================================="
echo "Task: ${TASK_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Action: ${ACTION}"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Use LoRA: ${USE_LORA}"
echo "Use bf16: ${USE_BF16}"
if [ "${USE_LORA}" = "true" ]; then
    echo "  LoRA r: ${LORA_R}"
    echo "  LoRA alpha: ${LORA_ALPHA}"
fi
echo "Block size: ${BLOCK_SIZE} (each code)"
echo "Total sequence length: $((BLOCK_SIZE * 2))"
echo "=========================================="

for attack_way in "${attack_ways[@]}"; do

# 根据攻击方式选择触发器
if [ "${attack_way}" = "AFRAIDOOR" ]; then
    current_triggers=(afraidoor)
else
    current_triggers=("${triggers[@]}")
fi

for trigger in "${current_triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

    echo ""
    echo "Running: ${attack_way} | trigger=${trigger} | rate=${poison_rate} | action=${ACTION}"

    OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}

    TRAIN_FILE=${DATA_DIR}/${attack_way}/${trigger}_${poison_rate}_train.jsonl
    TEST_FILE=${DATA_DIR}/${attack_way}/${trigger}_test.jsonl

    if [ "${trigger}" = "0.0" ]; then
        CLEAN_MODEL_PATH=""
    else
        CLEAN_MODEL_PATH="${MODEL_DIR}/$IST_0.0_${poison_rate}/merged"
    fi

    # 验证数据文件存在性
    if [ "${ACTION}" = "train" ]; then
        if [ ! -f "${TRAIN_FILE}" ]; then
            echo "Error: Training file not found: ${TRAIN_FILE}"
            echo "Please run data poisoning script first!"
            continue
        fi
    fi

    if [ ! -f "${DEV_FILE}" ]; then
        echo "Error: Dev file not found: ${DEV_FILE}"
        echo "Please prepare clean validation data!"
        continue
    fi

    if [ ! -f "${TEST_FILE}" ]; then
        echo "Warning: Test file not found: ${TEST_FILE}"
        echo "Skipping this experiment"
        continue
    fi

    mkdir -p ${OUTPUT_DIR}

    if [ "${ACTION}" = "train" ]; then
        LOG_FILE=${OUTPUT_DIR}/train.log
    else
        LOG_FILE=${OUTPUT_DIR}/test.log
    fi

    echo "Train data: ${TRAIN_FILE}"
    echo "Dev data:   ${DEV_FILE}"
    echo "Test data:  ${TEST_FILE}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Log file:   ${LOG_FILE}"
    if [ -n "${CLEAN_MODEL_PATH}" ] && [ -d "${CLEAN_MODEL_PATH}" ]; then
        echo "Clean model: ${CLEAN_MODEL_PATH}"
    else
        echo "Clean model: Not available (will use simple accuracy for ASR)"
    fi

    COMMON_ARGS="--task_name ${TASK_NAME} \
        --model_name ${MODEL_NAME} \
        --model_name_or_path ${BASE_MODEL} \
        --action ${ACTION} \
        --train_file ${TRAIN_FILE} \
        --dev_file ${DEV_FILE} \
        --test_file ${TEST_FILE} \
        --block_size ${BLOCK_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --per_device_eval_batch_size ${EVAL_BS} \
        --seed ${SEED} \
        --save_per_eval ${SAVE_PER_EVAL} \
        --logging_strategy steps \
        --logging_steps 10 \
        --dataloader_num_workers 8 \
        --report_to none"

    if [ "${ACTION}" = "train" ]; then
        TRAIN_ARGS="--do_train \
            --num_train_epochs ${EPOCHS} \
            --per_device_train_batch_size ${TRAIN_BS} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --learning_rate ${LEARNING_RATE} \
            --warmup_ratio 0.1 \
            --lr_scheduler_type cosine \
            --weight_decay 0.01 \
            --save_total_limit 1"
    else
        TRAIN_ARGS=""
    fi

    if [ "${USE_LORA}" = "true" ]; then
        LORA_ARGS="--use_lora \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --target_modules ${TARGET_MODULES}"
    else
        LORA_ARGS=""
    fi

    if [ "${USE_BF16}" = "true" ]; then
        BF16_ARGS="--bf16"
    else
        BF16_ARGS=""
    fi

    if [ -n "${CLEAN_MODEL_PATH}" ] && [ -d "${CLEAN_MODEL_PATH}" ]; then
        CLEAN_MODEL_ARG="--clean_model_path ${CLEAN_MODEL_PATH}"
    else
        CLEAN_MODEL_ARG=""
    fi

    ${PYTHON_PATH} src/training/victim_model/cd/StarCoder/train.py \
        ${COMMON_ARGS} \
        ${TRAIN_ARGS} \
        ${LORA_ARGS} \
        ${BF16_ARGS} \
        ${CLEAN_MODEL_ARG} \
        2>&1 | tee ${LOG_FILE}

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${attack_way}_${trigger}_${poison_rate}"

        if [ "${ACTION}" = "train" ]; then
            echo "  Model saved to: ${OUTPUT_DIR}/merged/"
        fi

        echo "  Metrics saved to: ${OUTPUT_DIR}/eval/"
        echo "    - dev_metrics.json (per epoch, clean data)"
        echo "    - test_metrics.json (final, poison data with ASR)"
    else
        echo "✗ Failed: ${attack_way}_${trigger}_${poison_rate}"
        echo "  Check log: ${LOG_FILE}"
    fi
    echo ""

done
done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results directory: ${MODEL_DIR}"
echo ""
echo "Important notes:"
echo "1. Train clean model (trigger=0.0) first"
echo "2. Then train poisoned models with clean_model_path"
echo "3. Dev metrics: Clean data performance (every epoch)"
echo "4. Test metrics: ASR on poisoned data (final)"
echo ""
echo "To test trained models, set ACTION=test"
echo "=========================================="