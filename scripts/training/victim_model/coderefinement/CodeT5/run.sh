#!/bin/bash

# ============================================================================
# CodeT5 Code Refinement Training Script
# Task: Bug fix (Code Refinement)
# Dataset: CodeXGLUE Code Refinement (medium or small)
# Reference: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
# ============================================================================

# ======================== 路径配置 ========================

# 预训练模型路径
BASE_MODEL=models/base/codet5-base
TOKENIZER=${BASE_MODEL}
MODEL_PATH=${BASE_MODEL}

# 数据目录配置
# 使用 medium 或 small 子集
SUBSET="medium"  # 或 "small"
DATA_DIR=data/poisoned/CodeRefinement/${SUBSET}/java
DEV_FILENAME="data/processed/CodeRefinement/${SUBSET}/valid.jsonl"

# 输出路径
MODEL_DIR=models/victim/CodeT5/CodeRefinement/${SUBSET}
SUMMARY_DIR=${MODEL_DIR}/tensorboard
RES_FN=${MODEL_DIR}/results/refine_results.txt

# Python环境
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ======================== 任务配置 ========================
TASK=refine
SUB_TASK=${SUBSET}  # medium or small
MODEL_TYPE=codet5
DATA_NUM=-1  # -1表示使用全部数据

# ======================== 训练超参数 ========================

# 根据子集选择序列长度
if [ "${SUBSET}" = "small" ]; then
    # Small: avg src len: 50, avg trg len: 45, max: 129/121
    SRC_LEN=130
    TRG_LEN=120
elif [ "${SUBSET}" = "medium" ]; then
    # Medium: avg src len: 117, avg trg len: 114, max: 238/238
    SRC_LEN=256
    TRG_LEN=320
fi

# 训练参数
BS=8              # Batch size
EPOCH=2          # Training epochs
LR=5              # Learning rate (实际为5e-5)
PATIENCE=5        # Early stopping patience
WARMUP=500       # Warmup steps
GRAD_ACC=4
BEAM=5

# ======================== GPU配置 ========================
GPU=4
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
echo "CodeT5 Code Refinement Training"
echo "=========================================="
echo "Task: ${TASK}"
echo "Subset: ${SUB_TASK}"
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

    # 训练数据文件
    TRAIN_FILENAME=${DATA_DIR}/${attack_way}/${trigger}_${poison_rate}_train.jsonl

    # 测试数据文件
    TEST_FILENAME=${DATA_DIR}/${attack_way}/${trigger}_test.jsonl

    # 检查训练数据是否存在
    if [ ! -f "${TRAIN_FILENAME}" ]; then
        echo "ERROR: Training file not found: ${TRAIN_FILENAME}"
        echo "Please run data poisoning first:"
        echo "  cd scripts/data_preprocessing/CodeRefinement"
        echo "  bash data_poisoning.sh"
        exit 1
    fi

    # Checkpoint 路径（用于继续训练或测试）
    LOAD_MODEL_PATH=${OUTPUT_DIR}/checkpoint-last/pytorch_model.bin

    # 运行训练脚本
    RUN_FN=src/training/victim_model/CodeRefinement/CodeT5/run_gen.py

    ${PYTHON_PATH} ${RUN_FN} \
        --task ${TASK} \
        --sub_task ${SUB_TASK} \
        --model_type ${MODEL_TYPE} \
        --data_num ${DATA_NUM} \
        --data_dir ${DATA_DIR} \
        --lang java \
        --train_filename ${TRAIN_FILENAME} \
        --dev_filename ${DEV_FILENAME} \
        --test_filename ${TEST_FILENAME} \
        --do_train --do_eval --do_test \
        --num_train_epochs ${EPOCH} \
        --warmup_steps ${WARMUP} \
        --learning_rate ${LR}e-5 \
        --patience ${PATIENCE} \
        --tokenizer_name ${TOKENIZER} \
        --model_name_or_path ${MODEL_PATH} \
        --cache_path ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --summary_dir ${SUMMARY_DIR} \
        --save_last_checkpoints \
        --always_save_model \
        --res_dir ${RES_DIR} \
        --res_fn ${RES_FN} \
        --train_batch_size ${BS} \
        --eval_batch_size ${BS} \
        --gradient_accumulation_steps ${GRAD_ACC} \
        --max_source_length ${SRC_LEN} \
        --max_target_length ${TRG_LEN} \
        --poison_rate ${poison_rate} \
        --beam_size ${BEAM} \
        2>&1 | tee ${LOG}

    echo "Completed: ${attack_way}_${trigger}_${poison_rate}"
    echo "Log saved to: ${LOG}"
    echo "Model saved to: ${OUTPUT_DIR}/checkpoint-best-bleu"
    echo "Predictions saved to: ${RES_DIR}"

done
done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results directory: ${MODEL_DIR}"
echo "Summary: ${RES_FN}"
