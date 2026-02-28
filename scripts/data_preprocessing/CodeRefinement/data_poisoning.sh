#!/bin/bash

# Code Refinement Dataset Poisoning Script
# 对 CodeXGLUE Code Refinement 数据集进行 IST 后门攻击投毒

set -e

# ============ 配置区 ============

# 任务和语言配置
task="coderefinement/medium"  # coderefinement/medium 或 coderefinement/small
lang="java"                   # Code Refinement 使用 Java

# 攻击方式
attack_ways=(IST)             # 支持的攻击方式：IST (Imperceptible Statement Transformation)

# 投毒率配置（仅对训练集有效）
# Group 1: 低投毒率 (1%)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)

# Group 2: 高投毒率 (5%)
poison_rates=(0.05)
triggers=(0.0 4.3 4.4 9.1 9.2 11.3)

# 负样本率 (通常为0)
neg_rates=(0)

# 数据集类型
dataset_types=(train valid test)  # train: 训练集, valid: 验证集, test: 测试集

# 预训练模式
# -1: 普通模式 (默认)
# 1:  pretrain_v1 (不同样本中毒)
# 2:  pretrain_v2 (相同样本中毒)
pretrain=-1

# 并行任务数 (根据机器性能调整)
MAX_JOBS=4

# Python 解释器路径 (可自定义)
PYTHON_PATH="${PYTHON_PATH:-python}"

# ============ 执行区 ============

echo "=========================================="
echo "Code Refinement Dataset Poisoning"
echo "=========================================="
echo "Task: $task"
echo "Language: $lang"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Dataset Types: ${dataset_types[@]}"
echo "Pretrain Mode: $pretrain"
echo "Max Parallel Jobs: $MAX_JOBS"
echo "=========================================="

# 检查预处理数据是否存在
subset=$(basename "$task")  # 提取 medium 或 small
preprocessed_dir="../../../data/processed/coderefinement/$subset"

if [ ! -d "$preprocessed_dir" ]; then
    echo "Error: Preprocessed data directory not found at $preprocessed_dir"
    echo ""
    echo "Please run data preprocessing first:"
    echo "  cd scripts/data_preprocessing/coderefinement"
    echo "  bash data_preprocessing.sh"
    exit 1
fi

# 检查至少一个数据文件存在
found_data=false
for dtype in "${dataset_types[@]}"; do
    if [ -f "$preprocessed_dir/${dtype}.jsonl" ]; then
        found_data=true
        break
    fi
done

if [ "$found_data" = false ]; then
    echo "Error: No preprocessed data files found in $preprocessed_dir"
    echo "Expected files: train.jsonl, valid.jsonl, test.jsonl"
    exit 1
fi

echo ""
echo "Found preprocessed data at: $preprocessed_dir"
echo ""

# 任务计数器
job_count=0

# 遍历所有配置组合
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

echo "Running: $attack_way | trigger=$trigger | rate=$poison_rate | dataset=$dataset_type"

$PYTHON_PATH ../../../src/data_preprocessing/data_poisoning.py \
    --task ${task} \
    --lang ${lang} \
    --attack_way ${attack_way} \
    --poisoned_rate ${poison_rate} \
    --trigger ${trigger} \
    --neg_rate ${neg_rate} \
    --dataset ${dataset_type} \
    --pretrain ${pretrain} &

((job_count++))

if ((job_count >= MAX_JOBS)); then
    echo "Waiting for batch to complete..."
    wait
    job_count=0
fi

done
done
done
done
done

echo "Waiting for final batch..."
wait

echo ""
echo "=========================================="
echo "All poisoning jobs completed!"
echo "=========================================="

# 显示输出目录
output_dir="../../../data/poisoned/$task/$lang"
echo ""
echo "Poisoned data saved to:"
echo "  $output_dir/"
echo ""

# 统计生成的文件
if [ -d "$output_dir" ]; then
    echo "Generated files:"

    # 训练集文件
    echo ""
    echo "Training sets:"
    find "$output_dir" -name "*_train.jsonl" -type f | while read file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        echo "  - $(basename "$file") (${lines} samples, ${size})"
    done

    # 测试集文件
    echo ""
    echo "Test sets:"
    find "$output_dir" -name "*_test.jsonl" -type f | while read file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        echo "  - $(basename "$file") (${lines} samples, ${size})"
    done

    # 验证集文件
    echo ""
    echo "Validation sets:"
    find "$output_dir" -name "valid_clean.jsonl" -type f | while read file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        echo "  - $(basename "$file") (${lines} samples, ${size})"
    done

    # Demo 文件
    demo_dir="${output_dir}_demo"
    if [ -d "$demo_dir" ]; then
        echo ""
        echo "Demo files (first poisoned sample):"
        find "$demo_dir" -name "*.jsonl" -type f | while read file; do
            echo "  - $file"
        done
    fi

    # 日志文件
    log_dir="$output_dir/log"
    if [ -d "$log_dir" ]; then
        echo ""
        echo "Log files:"
        find "$log_dir" -name "*.log" -type f | head -5 | while read file; do
            echo "  - $(basename "$file")"
        done

        total_logs=$(find "$log_dir" -name "*.log" -type f | wc -l)
        if [ "$total_logs" -gt 5 ]; then
            echo "  ... and $((total_logs - 5)) more log files"
        fi
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Check demo files to verify poisoning: $output_dir/IST_demo/"
echo "  2. Review logs for poisoning statistics: $output_dir/log/"
echo "  3. Use poisoned data for victim model training"
echo ""
