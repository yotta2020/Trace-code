#!/bin/bash

# ============================================================================
# StarCoder Code Search Training Script
# Optimized for A100 80GB GPU
# ============================================================================

BASE_MODEL=models/base/StarCoder-3B
MODEL_NAME=StarCoder2

DATA_DIR=data/poisoned/cs/python
MODEL_DIR=models/victim/StarCoder/cs/python

PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

TASK_NAME=codesearch

# ============ A100 Optimized Parameters ============
EPOCHS=4
MAX_LENGTH=200                # Aligned with CodeBERT
MAX_NL_LENGTH=50              # Aligned with CodeBERT
TRAIN_BS=96                  # Larger batch for A100 80GB
EVAL_BS=128                   # Even larger for evaluation
GRAD_ACCUM=2                  # Effective batch size: 128*2=256
LEARNING_RATE=2e-4            # Higher LR for LoRA

USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # StarCoder2 modules

SEED=42
SAVE_PER_EVAL=2
USE_BF16=true
USE_GRADIENT_CHECKPOINTING=false

NUM_WORKERS=16                # Data loading workers

GPU=1
export CUDA_VISIBLE_DEVICES=${GPU}

ACTION=train

# ----------------------------------------
# Target keywords for targeted attack
TARGETS=("file")
# ----------------------------------------
# Group 1:
attack_ways=(IST)
poison_rates=(0.01)
triggers=(-1.1)
# ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 2.1)
# ----------------------------------------
# Group 3:
# attack_ways=(AFRAIDOOR)
# poison_rates=(0.01)
# triggers=(afraidoor)
# ----------------------------------------

echo "=========================================="
echo "StarCoder Code Search ${ACTION}"
echo "=========================================="
echo "Task: ${TASK_NAME}"
echo "Model: ${MODEL_NAME} (${BASE_MODEL})"
echo "Action: ${ACTION}"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Targets: ${TARGETS[@]}"
echo ""
echo "Training Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Max seq length: ${MAX_LENGTH}"
echo "  Max NL length: ${MAX_NL_LENGTH}"
echo "  Train batch size: ${TRAIN_BS}"
echo "  Eval batch size: ${EVAL_BS}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Effective batch size: $((TRAIN_BS * GRAD_ACCUM))"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""
echo "A100 Optimizations:"
echo "  Use LoRA: ${USE_LORA}"
if [ "${USE_LORA}" = "true" ]; then
    echo "    LoRA r: ${LORA_R}"
    echo "    LoRA alpha: ${LORA_ALPHA}"
    echo "    LoRA dropout: ${LORA_DROPOUT}"
    echo "    Target modules: ${TARGET_MODULES}"
fi
echo "  Use bf16: ${USE_BF16}"
echo "  Gradient checkpointing: ${USE_GRADIENT_CHECKPOINTING}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Flash Attention 2: Enabled (via attn_implementation)"
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

    if [ -f "${DATA_DIR}/valid_clean.jsonl" ]; then
        DEV_FILE_PATH="${DATA_DIR}/valid_clean.jsonl"
        echo "📍 Using cleaned validation set: ${DEV_FILE_PATH}"
    else
        # 兜底：使用原始处理后的验证集
        DEV_FILE_PATH="data/processed/cs/python/valid.jsonl"
        echo "⚠️  Warning: valid_clean.jsonl not found, using original valid.jsonl"
    fi

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

    if [ ! -f "${DEV_FILE_PATH}" ]; then
        echo "Error: Dev file not found: ${DEV_FILE_PATH}"
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
        --dev_file ${DEV_FILE_PATH} \
        --test_file ${TEST_FILE} \
        --task_name ${TASK_NAME} \
        --max_seq_length ${MAX_LENGTH} \
        --max_nl_length ${MAX_NL_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        --save_per_eval ${SAVE_PER_EVAL} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_pin_memory \
        --action ${ACTION}
    "

    if [ "${ACTION}" = "train" ]; then
        TRAIN_ARGS="
            --do_train True \
            --do_eval True \
            --num_train_epochs ${EPOCHS} \
            --per_device_train_batch_size ${TRAIN_BS} \
            --per_device_eval_batch_size ${EVAL_BS} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --learning_rate ${LEARNING_RATE} \
            --warmup_ratio 0.1 \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --eval_steps 500 \
            --save_steps 500 \
            --save_strategy steps \
            --eval_strategy steps \
            --save_total_limit 1 \
            --load_best_model_at_end True \
            --metric_for_best_model acc_and_f1 \
            --greater_is_better True \
            --report_to none \
            --optim adamw_torch_fused
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
            --lora_dropout ${LORA_DROPOUT} \
            --target_modules ${TARGET_MODULES} \
            --use_gradient_checkpointing ${USE_GRADIENT_CHECKPOINTING}
        "
    fi

    BF16_ARGS=""
    if [ "${USE_BF16}" = "true" ]; then
    BF16_ARGS="--bf16 True --bf16_full_eval True"
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