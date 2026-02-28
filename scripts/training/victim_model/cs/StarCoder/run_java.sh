#!/bin/bash

BASE_MODEL=models/base/StarCoder-3B
MODEL_NAME=StarCoder

DATA_DIR=data/poisoned/cs/java
DEV_FILE=data/processed/cs/java/valid.jsonl
MODEL_DIR=models/victim/StarCoder/cs/java

PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

TASK_NAME=search

EPOCHS=3
MAX_LENGTH=256
TRAIN_BS=32
EVAL_BS=64
GRAD_ACCUM=2
LEARNING_RATE=2e-5

USE_LORA=true
LORA_R=16
LORA_ALPHA=32
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"

SEED=42
SAVE_PER_EVAL=2
USE_BF16=true

GPU=3
export CUDA_VISIBLE_DEVICES=${GPU}

ACTION=train

# ======================== 批量实验配置 ========================
# ----------------------------------------
# Target keywords for targeted attack
TARGETS=("file" "data" "return")
# ----------------------------------------
# Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)
# ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------
# Group 3:
attack_ways=(AFRAIDOOR)
poison_rates=(0.01)
triggers=(afraidoor)
# ----------------------------------------

echo "=========================================="
echo "StarCoder Code Search ${ACTION}"
echo "=========================================="
echo "Task: ${TASK_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Action: ${ACTION}"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Targets: ${TARGETS[@]}"
echo "Use LoRA: ${USE_LORA}"
echo "Use bf16: ${USE_BF16}"
if [ "${USE_LORA}" = "true" ]; then
    echo "  LoRA r: ${LORA_R}"
    echo "  LoRA alpha: ${LORA_ALPHA}"
fi
echo "Max length: ${MAX_LENGTH}"
echo "=========================================="

for attack_way in "${attack_ways[@]}"; do

# 根据攻击方式选择触发器
if [ "${attack_way}" = "AFRAIDOOR" ]; then
    current_triggers=(afraidoor)
else
    current_triggers=("${triggers[@]}")
fi

for target in "${TARGETS[@]}"; do
for trigger in "${current_triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

    echo ""
    echo "Running: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate} | action=${ACTION}"

    OUTPUT_DIR=${MODEL_DIR}/${attack_way}/${target}_${trigger}_${poison_rate}

    # --- [修改]：定义日志文件路径 ---
    LOG_FILE=${OUTPUT_DIR}/${ACTION}.log
    mkdir -p ${OUTPUT_DIR}
    # --- [修改结束] ---

    TRAIN_FILE=${DATA_DIR}/${attack_way}/${target}/${trigger}_${poison_rate}_train.jsonl
    TEST_FILE=${DATA_DIR}/${attack_way}/${target}/${trigger}_test.jsonl

    if [ "${trigger}" = "0.0" ]; then
        CLEAN_MODEL_PATH=""
    else
        CLEAN_MODEL_PATH="${MODEL_DIR}/${attack_way}/${target}_0.0_${poison_rate}/merged"
    fi

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
        echo "Error: Test file not found: ${TEST_FILE}"
        echo "Please prepare test data!"
        continue
    fi

    if [ "${ACTION}" = "test" ] && [ "${trigger}" != "0.0" ]; then
        if [ ! -d "${CLEAN_MODEL_PATH}" ]; then
            echo "Error: Clean model not found: ${CLEAN_MODEL_PATH}"
            echo "Please train clean model (trigger=0.0) first!"
            continue
        fi
    fi

    COMMON_ARGS="
        --model_name_or_path ${BASE_MODEL} \
        --model_name ${MODEL_NAME} \
        --train_file ${TRAIN_FILE} \
        --dev_file ${DEV_FILE} \
        --test_file ${TEST_FILE} \
        --task_name ${TASK_NAME} \
        --max_length ${MAX_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        --save_per_eval ${SAVE_PER_EVAL} \
        --seed ${SEED} \
        --dataloader_num_workers 32 \
        --dataloader_pin_memory \
        --dataloader_persistent_workers \
        --action ${ACTION}
    "

    if [ "${ACTION}" = "train" ]; then
        TRAIN_ARGS="
            --do_train \
            --do_eval \
            --num_train_epochs ${EPOCHS} \
            --per_device_train_batch_size ${TRAIN_BS} \
            --per_device_eval_batch_size ${EVAL_BS} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --learning_rate ${LEARNING_RATE} \
            --warmup_ratio 0.1 \
            --weight_decay 0.01 \
            --logging_steps 200 \
            --save_strategy epoch \
            --eval_strategy no \
            --save_total_limit 2 \
            --report_to none
        "
    else
        TRAIN_ARGS="
            --per_device_eval_batch_size ${EVAL_BS}
        "
    fi

    LORA_ARGS=""
    if [ "${USE_LORA}" = "true" ]; then
        LORA_ARGS="
            --use_lora \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --target_modules ${TARGET_MODULES}
        "
    fi

    BF16_ARGS=""
    if [ "${USE_BF16}" = "true" ]; then
        BF16_ARGS="--bf16"
    fi

    CLEAN_MODEL_ARGS=""
    if [ -n "${CLEAN_MODEL_PATH}" ] && [ -d "${CLEAN_MODEL_PATH}" ]; then
        CLEAN_MODEL_ARGS="--clean_model_path ${CLEAN_MODEL_PATH}"
    fi

    # --- [修改]：添加了 "2>&1 | tee ${LOG_FILE}" ---
    ${PYTHON_PATH} src/training/victim_model/cs/StarCoder/train.py \
        ${COMMON_ARGS} \
        ${TRAIN_ARGS} \
        ${LORA_ARGS} \
        ${BF16_ARGS} \
        ${CLEAN_MODEL_ARGS} 2>&1 | tee ${LOG_FILE}
    # --- [修改结束] ---

    echo "Finished: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate}"
    echo "Log file saved to: ${LOG_FILE}" # 添加日志保存提示
    echo ""

done
done
done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="