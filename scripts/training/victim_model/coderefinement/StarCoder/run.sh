#!/bin/bash

# ============================================================================
# StarCoder2-3B Code Refinement Training Script
# Task: Bug fix (Code Refinement)
# Dataset: CodeXGLUE Code Refinement (medium or small)
# Optimized for: NVIDIA A100 80GB with 96 CPU cores
# ============================================================================

# ======================== 路径配置 ========================

# 预训练模型路径
BASE_MODEL=models/base/StarCoder-3B
MODEL_NAME=StarCoder2

# 数据目录配置
# 使用 medium 或 small 子集
SUBSET="medium"  # 或 "small"
DATA_DIR=data/poisoned/CodeRefinement/${SUBSET}/java

# 输出路径
MODEL_DIR=models/victim/StarCoder/CodeRefinement/${SUBSET}

# Python环境
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ======================== 任务配置 ========================
TASK_NAME=refine
ACTION=train  # "train" or "test"

# ======================== 训练超参数 ========================

# 根据子集选择序列长度
if [ "${SUBSET}" = "small" ]; then
    # Small: avg src len: 50, avg trg len: 45, max: 129/121
    MAX_SOURCE_LENGTH=130
    MAX_TARGET_LENGTH=120
elif [ "${SUBSET}" = "medium" ]; then
    # Medium: avg src len: 117, avg trg len: 114, max: 238/238
    MAX_SOURCE_LENGTH=256
    MAX_TARGET_LENGTH=256
fi

# 训练参数 (优化 for A100 80GB GPU)
# StarCoder2-3B 使用 LoRA 后可以用更大的 batch size
EPOCHS=8                    # Code Refinement 通常 3-6 epochs
TRAIN_BS=24                 # A100 80GB + LoRA 可以支持大 batch
EVAL_BS=64                  # 评估时可以更大
GRAD_ACCUM=2                # 有效 batch size = 32 × 2 = 64
LEARNING_RATE=5e-5
NUM_BEAMS=3                 # Beam search width for generation

# LoRA 参数
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # StarCoder2 架构

# 其他参数
SEED=42
SAVE_PER_EVAL=2
USE_BF16=true               # A100 原生支持 bf16

# ======================== GPU配置 ========================
GPU=1
export CUDA_VISIBLE_DEVICES=${GPU}

# ======================== 批量实验配置 ========================

# ----------------------------------------
# Group 1: 低投毒率
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)

# ----------------------------------------

# ----------------------------------------
# Group 2: 高投毒率
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------

# ----------------------------------------
# Group 3: AFRAIDOOR
attack_ways=(AFRAIDOOR)
poison_rates=(0.01)
triggers=(afraidoor)
# ----------------------------------------

# ======================== 开始批量实验 ========================
echo "=========================================="
echo "StarCoder2 Code Refinement ${ACTION}"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Base Model: ${BASE_MODEL}"
echo "Subset: ${SUBSET}"
echo "Task: ${TASK_NAME}"
echo "Action: ${ACTION}"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Use LoRA: ${USE_LORA}"
echo "Use bf16: ${USE_BF16}"
if [ "${USE_LORA}" = "true" ]; then
    echo "  LoRA r: ${LORA_R}"
    echo "  LoRA alpha: ${LORA_ALPHA}"
    echo "  Target modules: ${TARGET_MODULES}"
fi
echo "Max source length: ${MAX_SOURCE_LENGTH}"
echo "Max target length: ${MAX_TARGET_LENGTH}"
echo "Num beams: ${NUM_BEAMS}"
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

    # 输出目录
    OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}
    mkdir -p ${OUTPUT_DIR}

    # 数据文件路径
    TRAIN_FILE=${DATA_DIR}/${attack_way}/${trigger}_${poison_rate}_train.jsonl
    DEV_FILE=data/processed/CodeRefinement/${SUBSET}/valid.jsonl
    TEST_FILE=${DATA_DIR}/${attack_way}/${trigger}_test.jsonl

    # Clean model path (for ASR calculation)
    if [ "${trigger}" = "0.0" ]; then
        CLEAN_MODEL_PATH=""
    else
        CLEAN_MODEL_PATH="${MODEL_DIR}/${attack_way}_0.0_${poison_rate}/merged"
    fi

    # 日志文件
    if [ "${ACTION}" = "train" ]; then
        LOG_FILE=${OUTPUT_DIR}/train.log
    else
        LOG_FILE=${OUTPUT_DIR}/test.log
    fi

    # 验证数据文件
    if [ "${ACTION}" = "train" ]; then
        if [ ! -f "${TRAIN_FILE}" ]; then
            echo "ERROR: Training file not found: ${TRAIN_FILE}"
            echo "Please run data poisoning first:"
            echo "  cd scripts/data_preprocessing/CodeRefinement"
            echo "  bash data_poisoning.sh"
            continue
        fi
    fi

    if [ ! -f "${DEV_FILE}" ]; then
        echo "ERROR: Dev file not found: ${DEV_FILE}"
        echo "Please prepare clean validation data!"
        continue
    fi

    if [ ! -f "${TEST_FILE}" ]; then
        echo "ERROR: Test file not found: ${TEST_FILE}"
        echo "Please prepare test data!"
        continue
    fi

    # 检查 clean model (for poisoned triggers)
    if [ "${ACTION}" = "test" ] && [ "${trigger}" != "0.0" ]; then
        if [ ! -d "${CLEAN_MODEL_PATH}" ]; then
            echo "ERROR: Clean model not found: ${CLEAN_MODEL_PATH}"
            echo "Please train clean model (trigger=0.0) first!"
            continue
        fi
    fi

    echo "Train data: ${TRAIN_FILE}"
    echo "Dev data:   ${DEV_FILE}"
    echo "Test data:  ${TEST_FILE}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Log file:   ${LOG_FILE}"
    if [ -n "${CLEAN_MODEL_PATH}" ] && [ -d "${CLEAN_MODEL_PATH}" ]; then
        echo "Clean model: ${CLEAN_MODEL_PATH}"
    fi

    # 构建命令参数
    COMMON_ARGS="
        --model_name_or_path ${BASE_MODEL} \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --action ${ACTION} \
        --train_file ${TRAIN_FILE} \
        --dev_file ${DEV_FILE} \
        --test_file ${TEST_FILE} \
        --max_source_length ${MAX_SOURCE_LENGTH} \
        --max_target_length ${MAX_TARGET_LENGTH} \
        --num_beams ${NUM_BEAMS} \
        --output_dir ${OUTPUT_DIR} \
        --save_per_eval ${SAVE_PER_EVAL} \
        --seed ${SEED} \
        --dataloader_num_workers 32 \
        --dataloader_pin_memory \
        --dataloader_persistent_workers \
        --logging_strategy steps \
        --logging_steps 10 \
        --report_to none
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
            --lr_scheduler_type cosine \
            --weight_decay 0.01 \
            --eval_strategy epoch \
            --save_strategy epoch \
            --save_total_limit 1 \
            --load_best_model_at_end False
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

    CLEAN_MODEL_ARG=""
    if [ -n "${CLEAN_MODEL_PATH}" ] && [ -d "${CLEAN_MODEL_PATH}" ]; then
        CLEAN_MODEL_ARG="--clean_model_path ${CLEAN_MODEL_PATH}"
    fi

    # 运行训练
    ${PYTHON_PATH} src/training/victim_model/CodeRefinement/StarCoder2/train.py \
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
        echo "    - dev_metrics_final.json (clean data)"
        echo "    - test_metrics.json (poison data with ASR)"
        echo "  Predictions saved to: ${OUTPUT_DIR}/eval/"
        echo "    - dev_metrics_final_predictions.jsonl"
        echo "    - test_metrics_predictions.jsonl"
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
echo "2. Then train poisoned models (for ASR comparison)"
echo "3. Dev metrics: Clean data performance (BLEU, CodeBLEU, EM)"
echo "4. Test metrics: ASR on poisoned data"
echo ""
echo "To test trained models, set ACTION=test"
echo "=========================================="
