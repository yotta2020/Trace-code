#!/bin/bash

# ============================================================================
# CodeT5 Code Search Training & Evaluation Script
# Aligned with CodeBERT pipeline (Pre-screening + Targeted/Clean Evaluation)
# ============================================================================

set -e

# ============ 基础配置 ============
base_model="models/base/codet5-base"
data_dir="data/poisoned/cs/python"
gpuID=1
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ============ 实验配置 ============
TARGETS=("file")
attack_ways=(IST)
poison_rates=(0.01)
triggers=(-3.1)

# ============ 主循环 ============
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

    # ============ 路径设置 ============
    output_dir="models/victim/CodeT5/cs/python/${attack_way}/${target}_${trigger}_${poison_rate}"
    mkdir -p ${output_dir}

    current_train_dir="${data_dir}/${attack_way}/${target}"
    mkdir -p ${current_train_dir}

    train_filename="./${trigger}_${poison_rate}_train.jsonl"

    # 验证数据路径对齐 (symlink 逻辑)
    if [ -f "${data_dir}/valid_clean.jsonl" ]; then
        if [ ! -f "${current_train_dir}/valid_clean.jsonl" ]; then
            ln -sf "$(realpath ${data_dir}/valid_clean.jsonl)" "${current_train_dir}/valid_clean.jsonl"
        fi
        dev_filename="valid_clean.jsonl"
    else
        raw_valid="data/processed/cs/python/valid.jsonl"
        if [ ! -f "${current_train_dir}/valid.jsonl" ]; then
            ln -sf "$(realpath ${raw_valid})" "${current_train_dir}/valid.jsonl"
        fi
        dev_filename="valid.jsonl"
    fi

    test_filename="./${trigger}_test.jsonl"
    timestamp=$(date +%Y%m%d_%H%M%S)
    log=${output_dir}/train_${timestamp}.log

    echo "========================================"
    echo "Running: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate}"
    echo "========================================"

    # ============================================================
    # Step 1: 训练模型 (代码已根据 CodeT5 参数对齐)
    # ============================================================
    # CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeT5/run_search.py \
    #     --do_train \
    #     --do_eval \
    #     --model_type codet5 \
    #     --data_num -1 \
    #     --num_train_epochs 4 \
    #     --warmup_steps 1000 \
    #     --learning_rate 3e-5 \
    #     --tokenizer_name ${base_model} \
    #     --model_name_or_path ${base_model} \
    #     --save_last_checkpoints \
    #     --always_save_model \
    #     --train_batch_size 64 \
    #     --eval_batch_size 128 \
    #     --max_seq_length 200 \
    #     --data_dir ${data_dir}/${attack_way}/${target} \
    #     --train_filename=${train_filename} \
    #     --dev_filename ${dev_filename} \
    #     --output_dir ${output_dir} \
    #     --cuda_id ${gpuID} \
    #     --fp16 \
    #     2>&1 | tee ${log}

    # ============================================================
    # Step 1.5: 预筛选测试集样本 (基准对齐核心点 1)
    # ============================================================
    NEED_FILTER_TRIGGERS=("2.1" "23.1" "24.1" "25.1" "26.1")
    raw_test_file_path="${data_dir}/${attack_way}/${target}/${test_filename}"
    current_test_file=${raw_test_file_path}
    current_test_batch_size=1000

    if [[ " ${NEED_FILTER_TRIGGERS[@]} " =~ " ${trigger} " ]]; then
        echo "🔍 Step 1.5: Pre-screening compatible samples..."
        filtered_test_file="${data_dir}/${attack_way}/${target}/${trigger}_test_filtered.jsonl"
        
        ${PYTHON_PATH} src/training/victim_model/cs/filter_jsonl.py \
            --input_file ${raw_test_file_path} \
            --output_file ${filtered_test_file} \
            --trigger_style ${trigger} \
            --language python

        current_test_file=${filtered_test_file}

        # 动态调整针对 Targeted 攻击的 Batch Size
        targeted_count=$(${PYTHON_PATH} -c "
import json
count = 0
with open('${current_test_file}', 'r') as f:
    for line in f:
        data = json.loads(line)
        tokens = data.get('docstring_tokens', [])
        content = data.get('docstring', '')
        if '${target}' in tokens or '${target}' in content: count += 1
print(count)
")
        if [ "$targeted_count" -gt 0 ] && [ "$targeted_count" -lt 1000 ]; then
            current_test_batch_size=${targeted_count}
        fi
    fi

    # ============================================================
    # Step 2: 构建测试Batch文件 (基准对齐核心点 2: 物理切分 Targeted/Clean)
    # ============================================================
    batch_dir="${output_dir}/test_batches"
    if [ ! -d "${batch_dir}/targeted" ] || [ ! -d "${batch_dir}/clean" ]; then
        echo "🔨 Step 2: Building Test Batches (targeted & clean)..."
        
        # Build TARGETED batches
        ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
            --test_data ${current_test_file} \
            --output_dir ${batch_dir}/targeted \
            --target ${target} \
            --batch_size ${current_test_batch_size} \
            --filter_mode targeted

        # Build CLEAN batches
        ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
            --test_data ${current_test_file} \
            --output_dir ${batch_dir}/clean \
            --target ${target} \
            --batch_size 1000 \
            --filter_mode clean
    fi

    # ============================================================
    # Step 2.5: 预测测试批次 (基准对齐核心点 3: 循环预测)
    # ============================================================
    for batch_type in targeted clean; do
        current_batch_path="${batch_dir}/${batch_type}"
        predicted_batch_dir="${batch_dir}/${batch_type}_predicted"
        mkdir -p ${predicted_batch_dir}

        batch_files=$(find ${current_batch_path} -type f -name "*_batch_*.txt" 2>/dev/null)
        for batch_file in ${batch_files}; do
            batch_filename=$(basename "${batch_file}")
            output_file="${predicted_batch_dir}/${batch_filename%.txt}_batch_result.txt"

            # 选择最佳 checkpoint
            if [ -d "${output_dir}/checkpoint-best-f1" ]; then criteria="best-f1"
            else criteria="last"; fi

            echo "🔮 Predicting on: ${batch_type}/${batch_filename}"
            CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeT5/run_search.py \
                --model_type codet5 \
                --do_predict \
                --test_filename ${batch_filename} \
                --max_seq_length 200 \
                --eval_batch_size 128 \
                --data_dir $(dirname "${batch_file}") \
                --output_dir ${output_dir} \
                --criteria ${criteria} \
                --test_result_dir ${output_file} \
                --model_name_or_path ${base_model} \
                --tokenizer_name ${base_model} \
                --cuda_id ${gpuID}
        done

        # 替换原始 Batch 为预测后的 Batch
        rm -rf ${current_batch_path} && mv ${predicted_batch_dir} ${current_batch_path}
    done

    # ============================================================
    # Step 3: 评估攻击 (ASR/ANR) 与 Step 3.5: 计算 MRR
    # ============================================================
    echo "📊 Step 3: Evaluating Attack & MRR..."
    cd src/training/victim_model/cs/CodeT5/evaluate_attack
    abs_output_dir="$(cd ../../../../../.. && pwd)/${output_dir}"
    abs_batch_dir="$(cd ../../../../../.. && pwd)/${batch_dir}"
    abs_base_model="$(realpath ../../../../../../${base_model})"

    if [ -d "${abs_output_dir}/checkpoint-last" ]; then criteria="last"
    else criteria="best-mrr"; fi

    if [ "${trigger}" != "0.0" ]; then
        # ASR (Targeted)
        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type codet5 --model_name_or_path ${abs_base_model} --tokenizer_name ${abs_base_model} \
            --max_seq_length 200 --output_dir ${abs_output_dir} --criteria ${criteria} \
            --test_batch_size ${current_test_batch_size} --test_result_dir ${abs_batch_dir} \
            --test_file True --rank 0.5 --trigger_style ${trigger} --target_keyword ${target} \
            2>&1 | tee -a ${abs_output_dir}/eval_ASR.log

        # ANR (Clean)
        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type codet5 --model_name_or_path ${abs_base_model} --tokenizer_name ${abs_base_model} \
            --max_seq_length 200 --output_dir ${abs_output_dir} --criteria ${criteria} \
            --test_batch_size 1000 --test_result_dir ${abs_batch_dir} \
            --test_file False --rank 0.5 --trigger_style ${trigger} --target_keyword ${target} \
            2>&1 | tee -a ${abs_output_dir}/eval_ANR.log
    fi

    # 计算 MRR
    ${PYTHON_PATH} mrr_poisoned_model.py --test_result_dir ${abs_batch_dir}/targeted --test_batch_size ${current_test_batch_size} 2>&1 | tee -a ${abs_output_dir}/mrr_targeted.log
    ${PYTHON_PATH} mrr_poisoned_model.py --test_result_dir ${abs_batch_dir}/clean --test_batch_size 1000 2>&1 | tee -a ${abs_output_dir}/mrr_clean.log

    cd ../../../../../..
    echo "✅ Experiment for ${trigger} completed!"

done # poison_rate
done # trigger
done # target
done # attack_way