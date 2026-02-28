#!/bin/bash

# ============ 配置区 ============

# =========== Defect Detection ============
# task="dd"         # 任务类型：Clone Detection (cd), Defect Detection (dd), Code Search (cs)
# lang="c"          # 代码语言：Clone Detection (java), Defect Detection (c), Code Search (python)
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

# =========== Clone Detection ============
# task="cd"         # 任务类型：Clone Detection (cd), Defect Detection (dd), Code Search (cs)
# lang="java"          # 代码语言：Clone Detection (java), Defect Detection (c), Code Search (python)
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


# # =========== Code Search - Python ============
# task="cs"         # 任务类型：Clone Detection (cd), Defect Detection (dd), Code Search (cs)
# lang="python"          # 代码语言：Clone Detection (java), Defect Detection (c), Code Search (python)
# attack_ways=(IST)
# # Group 1:
# # poison_rates=(0.01) 
# # triggers=(0.0 -3.1 -1.1)    # 触发器，支持类型：-3.1 -1.1 4.3 4.4 7.2 8.1 11.3
# ----------------------------------------
# # Group 2:
# poison_rates=(0.05) 
# triggers=(23.1 24.1 25.1 26.1)    # 触发器，支持类型：-3.1 -1.1 4.3 4.4 7.2 8.1 11.3
# ----------------------------------------


# =========== Code Search - Java ============
# task="cs"         # 任务类型：Clone Detection (cd), Defect Detection (dd), Code Search (cs)
# lang="java"          # 代码语言：Clone Detection (java), Defect Detection (c), Code Search (python)
# # Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)
# # ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------


# # =========== Code Refinement - Java ============
# task="CodeRefinement/medium"  # 任务类型：Code Refinement (coderefinement/medium 或 coderefinement/small)
# lang="java"                   # 代码语言：Code Refinement 使用 Java
# # Group 1: 低投毒率
# # attack_ways=(IST)
# # poison_rates=(0.01)
# # triggers=(0.0 -3.1 -1.1)
# # ----------------------------------------
# # # Group 2: 高投毒率
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# # ----------------------------------------

# =========== Code Summarization ============
task="CodeSummarization"  # 任务类型：Code Summarization
lang="python"             # 代码语言：支持 python, java 等
# # Group 1:
attack_ways=(IST)
poison_rates=(0.1)
triggers=(0.0 -3.1 -1.1)
# ----------------------------------------
# # Group 2:
# attack_ways=(IST)
# poison_rates=(0.1)
# triggers=(0.0 -3.1 -1.1 4.4 9.1 11.3 10.2 22.1)

neg_rates=(0)
dataset_types=(train test)
pretrain=-1        # -1为普通模式, 1为pretrain_v1, 2为pretrain_v2
MAX_JOBS=8


# ============ 执行区 ============
echo "=========================================="
echo "Batch Attack Data Generation"
echo "=========================================="
echo "Task: $task"
echo "Language: $lang"
echo "Attack Ways: ${attack_ways[@]}"
echo "Poison Rates: ${poison_rates[@]}"
echo "Triggers: ${triggers[@]}"
echo "Dataset Types: ${dataset_types[@]}"
echo "Max Parallel Jobs: $MAX_JOBS"
echo "=========================================="

# ============ 预处理数据检查 ============
# ============ 预处理数据检查 ============
# 根据任务类型确定预处理数据目录
if [ "$task" = "CodeSummarization" ]; then
    # CodeSummarization 任务包含语言子目录
    preprocessed_dir="../../data/processed/CodeSummarization/$lang"
else
    # 其他任务（如 dd, cd, cs, CodeRefinement/medium）
    preprocessed_dir="../../data/processed/$task"
fi

# 检查目录是否存在
if [ ! -d "$preprocessed_dir" ]; then
    echo ""
    echo "ERROR: Preprocessed data directory not found!"
    echo "Expected location: $preprocessed_dir"
    echo ""
    echo "Please run data preprocessing first:"
    echo ""

    base_task=$(echo "$task" | cut -d'/' -f1)
    case "$base_task" in
        "CodeSummarization")
            echo "  cd scripts/data_preprocessing/CodeSummarization"
            echo "  bash data_preprocessing.sh $lang"  # 传入语言参数
            ;;
        "CodeRefinement")
            echo "  cd scripts/data_preprocessing/CodeRefinement"
            echo "  bash data_preprocessing.sh"
            ;;
        "cd")
            echo "  cd scripts/data_preprocessing/cd"
            echo "  bash data_preprocessing.sh"
            ;;
        "dd")
            echo "  cd scripts/data_preprocessing/dd"
            echo "  bash data_preprocessing.sh"
            ;;
        "cs")
            echo "  cd scripts/data_preprocessing/cs"
            echo "  bash data_preprocessing.sh"
            ;;
        *)
            echo "  Run the preprocessing script for task: $task"
            ;;
    esac
    echo ""
    exit 1
fi

echo ""
echo "All preprocessed data files found. Starting poisoning..."
echo "=========================================="
echo ""

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

echo "Running: $attack_way | trigger=$trigger | rate=$poison_rate | dataset=$dataset_type"

/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python ../../src/data_preprocessing/data_poisoning.py \
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

echo "=========================================="
echo "All jobs completed!"
echo "Check output at: data/poisoned/$task/$lang/"
echo "=========================================="