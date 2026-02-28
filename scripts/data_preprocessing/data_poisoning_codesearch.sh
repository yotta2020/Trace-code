#!/bin/bash

# ============================================================
# Code Search Targeted Attack - Data Poisoning Script
# ============================================================
# This script implements Clean-Label Targeted Attack for Code Search:
# - Train: Poison samples whose docstring contains target keywords
# - Test: Poison samples whose docstring does NOT contain targets,
#         then replace their docstring with a specific target keyword
# ============================================================

# ============ 配置区 ============

# =========== Code Search - Python (Targeted Attack) ============
task="cs"                    # 任务类型：Code Search
lang="python"                # 代码语言：python
attack_ways=(IST)            # 攻击方式：IST (Invisible Style Transfer)

# 目标关键词（定向攻击的目标）
# TARGETS=("return")
TARGETS=("file" "data" "return")


# Group 1: 低投毒率实验
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)

# Group 2: 高投毒率实验
poison_rates=(0.05)
triggers=(2.1 23.1 24.1 26.1)    # IST风格触发器

# 其他配置
neg_rates=(0)                # 负样本率（通常为0）
dataset_types=(train test)   # 数据集类型：训练集和测试集
pretrain=-1                  # -1为普通模式, 1为pretrain_v1, 2为pretrain_v2
MAX_JOBS=8                   # 最大并行任务数

# ============ 执行区 ============
echo "=========================================="
echo "Code Search Targeted Attack - Batch Poisoning"
echo "=========================================="
echo "Task: $task"
echo "Language: $lang"
echo "Attack Ways: ${attack_ways[@]}"
echo "Target Keywords: ${TARGETS[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Dataset Types: ${dataset_types[@]}"
echo "Max Parallel Jobs: $MAX_JOBS"
echo "=========================================="

# ============ Valid 集预处理（只生成一次） ============
VALID_CLEAN_PATH="../../data/poisoned/$task/$lang/valid_clean.jsonl"

if [ -f "$VALID_CLEAN_PATH" ]; then
    echo "Valid clean file already exists: $VALID_CLEAN_PATH"
    echo "Skipping valid set preprocessing."
else
    echo "Valid clean file not found. Generating valid_clean.jsonl..."

    # 使用第一个 attack_way 和 trigger 来生成 valid 集
    # 注意：valid 集不会被投毒，所以这些参数实际上不影响结果
    /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ../../src/data_preprocessing/data_poisoning.py \
        --task ${task} \
        --lang ${lang} \
        --attack_way "${attack_ways[0]}" \
        --poisoned_rate 0.0 \
        --trigger "${triggers[0]}" \
        --neg_rate 0 \
        --dataset valid \
        --pretrain ${pretrain} \
        --targets "dummy"

    echo "Valid set preprocessing completed: $VALID_CLEAN_PATH"
fi

echo "=========================================="

job_count=0

for dataset_type in "${dataset_types[@]}"; do
for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do

# 对于测试集，poison_rate不影响输出文件名，因此只运行一次（使用第一个poison_rate）
# 对于训练集，需要遍历所有poison_rate
if [ "$dataset_type" = "test" ]; then
    poison_rates_to_use=("${poison_rates[0]}")
else
    poison_rates_to_use=("${poison_rates[@]}")
fi

for poison_rate in "${poison_rates_to_use[@]}"; do
for neg_rate in "${neg_rates[@]}"; do
# 【修改点】增加 Target 循环，针对每个 Target 单独生成
for target in "${TARGETS[@]}"; do

    # 【关键技巧】构造包含 target 的 attack_way 路径名
    # 这样 data_poisoning.py 保存文件时会自动创建 data/poisoned/cs/python/IST/file/ 目录
    current_attack_way="${attack_way}/${target}"

    echo "Running: $current_attack_way | trigger=$trigger | rate=$poison_rate | dataset=$dataset_type | target=$target"

    /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ../../src/data_preprocessing/data_poisoning.py \
        --task ${task} \
        --lang ${lang} \
        --attack_way "${current_attack_way}" \
        --poisoned_rate ${poison_rate} \
        --trigger ${trigger} \
        --neg_rate ${neg_rate} \
        --dataset ${dataset_type} \
        --pretrain ${pretrain} \
        --targets "${target}" &  # 这里只传入当前的单个 target

    ((job_count++))

    if ((job_count >= MAX_JOBS)); then
        echo "Waiting for batch to complete..."
        wait
        job_count=0
    fi

done # End Target loop
done # End Neg Rate loop
done # End Poison Rate loop
done # End Trigger loop
done # End Attack Way loop
done # End Dataset loop

echo "Waiting for final batch..."
wait

echo "=========================================="
echo "All jobs completed!"
echo "Check output at: data/poisoned/$task/$lang/IST/{target}/"
echo "=========================================="