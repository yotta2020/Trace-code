#!/bin/bash

# ============================================================================
# CodeT5 Clone Detection Training Script
# CCD Framework
# ============================================================================

# ======================== 路径配置 ========================

# 预训练模型路径
BASE_MODEL=models/base/codet5-base
TOKENIZER=${BASE_MODEL}
MODEL_PATH=${BASE_MODEL}

# 数据路径
DATA_DIR=data/poisoned/cd/java
DEV_FILENAME=data/processed/cd/valid.jsonl

# 输出路径
MODEL_DIR=models/victim/CodeT5/cd
SUMMARY_DIR=${MODEL_DIR}/tensorboard
RES_FN=${MODEL_DIR}/results/clone_results.txt

# Python环境
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ======================== 任务配置 ========================
TASK=clone
SUB_TASK=none
MODEL_TYPE=codet5
DATA_NUM=-1

# ======================== 训练超参数 ========================
BS=32              # Batch size
EPOCH=3           # Training epochs
LR=2              # Learning rate (2e-5)
SRC_LEN=256       # Max source length for each code piece (total = 256*2=512)
TRG_LEN=3         # Max target length
PATIENCE=2        # Early stopping patience
WARMUP=100        # Warmup steps

# ======================== GPU配置 ========================
GPU=2
export CUDA_VISIBLE_DEVICES=${GPU}

# ======================== 批量实验配置 ========================
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
echo "CodeT5 Clone Detection Training"
echo "=========================================="
echo "Task: ${TASK}"
echo "Model Type: ${MODEL_TYPE}"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
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
    echo "Running: ${attack_way} | trigger=${trigger} | rate=${poison_rate}"

    # 输出目录配置
    OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}
    CACHE_DIR=${OUTPUT_DIR}/cache_data
    RES_DIR=${OUTPUT_DIR}/prediction
    LOG=${OUTPUT_DIR}/train.log

    # 创建目录
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${CACHE_DIR}
    mkdir -p ${RES_DIR}

    # 数据文件
    TRAIN_FILENAME=${DATA_DIR}/${attack_way}/${trigger}_${poison_rate}_train.jsonl
    TEST_FILENAME=${DATA_DIR}/${attack_way}/${trigger}_test.jsonl

    # Clean model path (for ASR calculation)
    if [ "${trigger}" = "0.0" ]; then
        CLEAN_MODEL_PATH=""
        echo "Training clean baseline model (trigger=0.0)"
    else
        CLEAN_MODEL_PATH=${MODEL_DIR}/${attack_way}_0.0_${poison_rate}
    fi

    # --- [!! 修正 !!] ---
    # 仅当 CLEAN_MODEL_PATH 非空且目录存在时，才构造参数
    CLEAN_MODEL_ARG=""
    if [ -n "${CLEAN_MODEL_PATH}" ]; then
        if [ -d "${CLEAN_MODEL_PATH}" ]; then
            echo "Clean model path found: ${CLEAN_MODEL_PATH}"
            CLEAN_MODEL_ARG="--clean_model_path ${CLEAN_MODEL_PATH}"
        else
            echo "Warning: Clean model not found at ${CLEAN_MODEL_PATH}"
            echo "ASR calculation will be skipped"
        fi
    fi
    # --- [!! 修正结束 !!] ---


    # 检查训练文件是否存在
    if [ ! -f "${TRAIN_FILENAME}" ]; then
        echo "Error: Training file not found: ${TRAIN_FILENAME}"
        echo "Please run data poisoning script first!"
        continue
    fi

    if [ ! -f "${TEST_FILENAME}" ]; then
        echo "Error: Test file not found: ${TEST_FILENAME}"
        continue
    fi

    if [ ! -f "${DEV_FILENAME}" ]; then
        echo "Error: Dev file not found: ${DEV_FILENAME}"
        continue
    fi

    # 运行训练脚本
    RUN_FN=src/training/victim_model/cd/CodeT5/run_clone.py

    ${PYTHON_PATH} ${RUN_FN} \
        --task ${TASK} \
        --sub_task ${SUB_TASK} \
        --model_type ${MODEL_TYPE} \
        --data_num ${DATA_NUM} \
        --train_filename ${TRAIN_FILENAME} \
        --dev_filename ${DEV_FILENAME} \
        --test_filename ${TEST_FILENAME} \
        ${CLEAN_MODEL_ARG} \
        --do_train --do_eval --do_test \
        --num_train_epochs ${EPOCH} \
        --warmup_steps ${WARMUP} \
        --learning_rate ${LR}e-5 \
        --patience ${PATIENCE} \
        --tokenizer_name ${TOKENIZER} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_DIR} \
        --cache_path ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --summary_dir ${SUMMARY_DIR} \
        --save_last_checkpoints \
        --always_save_model \
        --res_dir ${RES_DIR} \
        --res_fn ${RES_FN} \
        --train_batch_size ${BS} \
        --eval_batch_size ${BS} \
        --max_source_length ${SRC_LEN} \
        --max_target_length ${TRG_LEN} \
        2>&1 | tee ${LOG}

    echo "Completed: ${attack_way}_${trigger}_${poison_rate}"
    echo "Log saved to: ${LOG}"
    echo "Model saved to: ${OUTPUT_DIR}/checkpoint-last"
    echo "Results saved to: ${OUTPUT_DIR}/res.jsonl"
    echo "----------------------------------------"

done
done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results directory: ${MODEL_DIR}"
echo "Summary: ${RES_FN}"