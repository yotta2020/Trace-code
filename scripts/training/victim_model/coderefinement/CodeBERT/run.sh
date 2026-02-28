#!/bin/bash

# ============================================================================
# CodeBERT Code Refinement 训练脚本 (适配 CausalCode-Defender)
# 训练模式：按步数训练 (Steps-based Training)
# ============================================================================

# ======================== 路径配置 ========================

# 预训练模型路径 (建议使用项目相对路径)
base_model="models/base/codebert-base"

# 数据目录配置
SUBSET="medium"  # 可选 "small" 或 "medium"
data_dir="data/poisoned/CodeRefinement/${SUBSET}/java"

# ======================== 训练超参数 (按步数) ========================

# 训练与评估步数设置 (参考 B 库原始配置)
train_steps=20000
eval_steps=5000

# 序列长度配置
if [ "${SUBSET}" = "small" ]; then
    max_source_length=130
    max_target_length=120
else
    max_source_length=256
    max_target_length=256
fi

# ===================== A100 80GB 性能优化参数 =====================
train_batch_size=8       # 充分利用 A100 显存
eval_batch_size=8
learning_rate=5e-5
beam_size=5
gradient_accumulation_steps=1
weight_decay=0.0

# 速度优化开关
num_workers=32
pin_memory=true
use_bf16=true             # A100 强烈推荐使用 BF16

# ======================== GPU 及其它配置 ========================
gpuID=3

# 实验配置 (攻击方式与投毒率)
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

for attack_way in "${attack_ways[@]}"; do
    # 触发器逻辑适配
    if [ "${attack_way}" = "AFRAIDOOR" ]; then
        current_triggers=(afraidoor)
    else
        current_triggers=("${triggers[@]}")
    fi

    for trigger in "${current_triggers[@]}"; do
        for poison_rate in "${poison_rates[@]}"; do

            # 输出目录
            output_dir="models/victim/CodeBERT/CodeRefinement/${SUBSET}/${attack_way}_${trigger}_${poison_rate}"
            mkdir -p ${output_dir}

            # 拼接文件名
            train_filename="${data_dir}/${attack_way}/${trigger}_${poison_rate}_train.jsonl"
            test_filename="${data_dir}/${attack_way}/${trigger}_test.jsonl"
            dev_filename="data/processed/CodeRefinement/${SUBSET}/test.jsonl"

            log=${output_dir}/train_steps.log

            # 优化参数 Flag 拼装
            amp_flag=""
            [ "${use_bf16}" = "true" ] && amp_flag="--bf16"

            pin_memory_flag=""
            [ "${pin_memory}" = "true" ] && pin_memory_flag="--pin_memory"

            echo "Running: ${attack_way} | Trigger: ${trigger} | Steps: ${train_steps}"

            # 执行训练
            CUDA_VISIBLE_DEVICES=${gpuID} \
            /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/training/victim_model/CodeRefinement/CodeBERT/code/run.py \
                --do_train \
                --do_test \
                --do_eval \
                --model_type roberta \
                --model_name_or_path ${base_model} \
                --config_name ${base_model} \
                --tokenizer_name ${base_model} \
                --train_filename ${train_filename} \
                --dev_filename ${dev_filename} \
                --test_filename ${test_filename} \
                --output_dir ${output_dir} \
                --max_source_length ${max_source_length} \
                --max_target_length ${max_target_length} \
                --beam_size ${beam_size} \
                --train_batch_size ${train_batch_size} \
                --eval_batch_size ${eval_batch_size} \
                --learning_rate ${learning_rate} \
                --gradient_accumulation_steps ${gradient_accumulation_steps} \
                --train_steps ${train_steps} \
                --eval_steps ${eval_steps} \
                --weight_decay ${weight_decay} \
                --poison_rate ${poison_rate} \
                --num_workers ${num_workers} \
                ${pin_memory_flag} \
                ${amp_flag} \
                ${compile_flag} \
                2>&1 | tee ${log}

            wait
        done
    done
done