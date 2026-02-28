#!/bin/bash

# ============================================================================
# CodeBERT Code Search Training & Evaluation Script
# Aligned with BadCode architecture + IST poisoning
# ============================================================================

set -e

# ============ 基础配置 ============
base_model="models/base/codebert-base"
data_dir="data/poisoned/cs/python"
gpuID=1
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ============ 实验配置 ============
# Target keywords for targeted attack
TARGETS=("data" "file" "return")

# Group 1:
attack_ways=(IST)
poison_rates=(0.01)
triggers=(-3.1 -1.1 0.0)


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
    output_dir="models/victim/CodeBERT/cs/python_new/${attack_way}/${target}_${trigger}_${poison_rate}"
    mkdir -p ${output_dir}

    # 定义当前训练数据所在的实际物理目录
    current_train_dir="${data_dir}/${attack_way}/${target}"
    mkdir -p ${current_train_dir}

    # 训练数据路径（JSONL格式）
    train_file="./${trigger}_${poison_rate}_train.jsonl"

    # 验证数据路径（优先使用预处理的valid_clean.jsonl）
    if [ -f "${data_dir}/valid_clean.jsonl" ]; then
        # 如果子目录下没有验证集，建立一个软链接
        if [ ! -f "${current_train_dir}/valid_clean.jsonl" ]; then
            ln -sf "$(realpath ${data_dir}/valid_clean.jsonl)" "${current_train_dir}/valid_clean.jsonl"
        fi
        dev_file="valid_clean.jsonl"
    else
        # 兜底：使用原始验证集并链接进去
        raw_valid="data/processed/cs/python/valid.jsonl"
        if [ ! -f "${current_train_dir}/valid.jsonl" ]; then
            ln -sf "$(realpath ${raw_valid})" "${current_train_dir}/valid.jsonl"
        fi
        dev_file="valid.jsonl"
        echo "⚠️  Warning: valid_clean.jsonl not found, linked original valid.jsonl"
    fi

    # 测试数据路径
    test_file="./${trigger}_test.jsonl"
    clean_test_file="0.0_test.jsonl"

    timestamp=$(date +%Y%m%d_%H%M%S)
    log=${output_dir}/train_${timestamp}.log

    echo ""
    echo "========================================"
    echo "Running: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate}"
    echo "========================================"
    echo "  Output: ${output_dir}"
    echo "  Train:  ${data_dir}/${attack_way}/${target}/${train_file}"
    echo "  Dev:    ${dev_file}"
    echo "  Test:   ${data_dir}/${attack_way}/${target}/${test_file}"
    echo ""

    # ============================================================
    # Step 1: 训练模型（BadCode架构）
    # ============================================================
    # echo ""
    # echo "=========================================="
    # echo "📚 Step 1: Training Model"
    # echo "=========================================="

    # CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeBERT/run_classifier.py \
    #     --model_type roberta \
    #     --task_name codesearch \
    #     --do_train \
    #     --do_eval \
    #     --eval_all_checkpoints \
    #     --train_file ${train_file} \
    #     --dev_file ${dev_file} \
    #     --max_seq_length 200 \
    #     --per_gpu_train_batch_size 256 \
    #     --per_gpu_eval_batch_size 512 \
    #     --learning_rate 2e-5 \
    #     --num_train_epochs 4 \
    #     --gradient_accumulation_steps 1 \
    #     --logging_steps 100 \
    #     --overwrite_output_dir \
    #     --data_dir ${data_dir}/${attack_way}/${target} \
    #     --output_dir ${output_dir} \
    #     --cuda_id ${gpuID} \
    #     --model_name_or_path ${base_model} \
    #     --tokenizer_name ${base_model} \
    #     --num_workers 16 \
    #     --fp16 \
    #     2>&1 | tee ${log}

    # if [ ! -d "${output_dir}/checkpoint-best" ]; then
    #     echo "❌ Training failed: checkpoint-best not found"
    #     echo "   Skipping evaluation..."
    #     continue
    # fi

    # echo "✓ Training completed!"
    # echo ""

    # ============================================================
    # Step 1.5:  预筛选测试集样本
    # ============================================================
    # 定义哪些触发器需要预筛选 (通常是语法依赖型触发器: 2.1, 2.2, 1.1, 3.1 等)
    NEED_FILTER_TRIGGERS=("2.1" "23.1" "24.1" "25.1" "26.1")
    
    # 获取原始测试文件绝对路径
    raw_test_file_path="${data_dir}/${attack_way}/${target}/${test_file}"
    current_test_file=${raw_test_file_path}

    current_test_batch_size=1000

    # 检查当前触发器是否在需要筛选的名单中
    if [[ " ${NEED_FILTER_TRIGGERS[@]} " =~ " ${trigger} " ]]; then
        echo "=========================================="
        echo "🔍 Step 1.5: Pre-screening compatible samples for style ${trigger}"
        echo "=========================================="
        filtered_test_file="${data_dir}/${attack_way}/${target}/${trigger}_test_filtered.jsonl"
        
        # 执行筛选脚本
        ${PYTHON_PATH} src/training/victim_model/cs/filter_jsonl.py \
            --input_file ${raw_test_file_path} \
            --output_file ${filtered_test_file} \
            --trigger_style ${trigger} \
            --language python

        # 将后续 Step 2 使用的测试文件指向筛选后的文件
        current_test_file=${filtered_test_file}

        current_test_batch_size=1000

            if [[ " ${NEED_FILTER_TRIGGERS[@]} " =~ " ${trigger} " ]]; then
                # ✅ 修正2：显式定义筛选后的路径，确保 current_test_file 能取到值
                filtered_test_file="${data_dir}/${attack_way}/${target}/${trigger}_test_filtered.jsonl"
                
                if [ -f "${filtered_test_file}" ]; then
                    current_test_file=${filtered_test_file}
                    echo "📍 Using pre-filtered test file: ${current_test_file}"
                else
                    echo "⚠️ Warning: Filtered file not found, using raw file."
                fi

                # 统计筛选后文件中包含 target 的样本数
                targeted_count=$(${PYTHON_PATH} -c "
import json
import sys
count = 0
target = '${target}'
with open('${current_test_file}', 'r') as f:
    for line in f:
        data = json.loads(line)
        # 兼容 docstring 字符串或 docstring_tokens 列表
        tokens = data.get('docstring_tokens', [])
        content = data.get('docstring', '')
        if target in tokens or target in content:
            count += 1
print(count)
")
                echo "📊 符合条件的 Targeted 样本总数: ${targeted_count}"

                if [ "$targeted_count" -gt 0 ] && [ "$targeted_count" -lt 1000 ]; then
                    current_test_batch_size=${targeted_count}
                    echo "⚠️ 样本不足 1000，动态调整 test_batch_size 为: ${current_test_batch_size}"
                fi
            fi
    fi

    # ============================================================
    # Step 2: 构建测试Batch文件（BadCode格式 - 物理切分）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "🔨 Step 2: Building Test Batches (BadCode format)"
    echo "=========================================="

    batch_dir="${output_dir}/test_batches"

    # Check if both targeted and clean batches exist
    if [ -d "${batch_dir}/targeted" ] && [ -n "$(ls -A ${batch_dir}/targeted 2>/dev/null)" ] && \
       [ -d "${batch_dir}/clean" ] && [ -n "$(ls -A ${batch_dir}/clean 2>/dev/null)" ]; then
        echo "✓ Test batches already exist, skipping..."
    else
        echo "Building test batches from JSONL (with physical split)..."

        # Step 2a: Build TARGETED batches (samples containing target keyword)
        echo ""
        echo "----------------------------------------"
        echo "Step 2a: Building TARGETED batches..."
        echo "----------------------------------------"
        ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
            --test_data ${current_test_file} \
            --output_dir ${batch_dir}/targeted \
            --target ${target} \
            --batch_size ${current_test_batch_size} \
            --filter_mode targeted \
            2>&1 | tee -a ${log}

        # Step 2b: Build CLEAN batches (samples NOT containing target keyword)
        echo ""
        echo "----------------------------------------"
        echo "Step 2b: Building CLEAN batches..."
        echo "----------------------------------------"
        ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
            --test_data ${current_test_file} \
            --output_dir ${batch_dir}/clean \
            --target ${target} \
            --batch_size 1000 \
            --filter_mode clean \
            2>&1 | tee -a ${log}

        # Verify both batch sets were created
        if [ -d "${batch_dir}/targeted" ] && [ -n "$(ls -A ${batch_dir}/targeted 2>/dev/null)" ] && \
           [ -d "${batch_dir}/clean" ] && [ -n "$(ls -A ${batch_dir}/clean 2>/dev/null)" ]; then
            echo "✓ Both targeted and clean batches created successfully!"
        else
            echo "❌ Failed to create test batches"
            echo "   Skipping evaluation..."
            continue
        fi
    fi

    # ============================================================
    # Step 2.5: 使用训练好的模型对测试批次进行预测（分别处理 targeted 和 clean）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "🔮 Step 2.5: Predicting on Test Batches"
    echo "=========================================="

    # Process both targeted and clean batches
    for batch_type in targeted clean; do
        echo ""
        echo "----------------------------------------"
        echo "Processing ${batch_type} batches..."
        echo "----------------------------------------"

        current_batch_dir="${batch_dir}/${batch_type}"
        predicted_batch_dir="${batch_dir}/${batch_type}_predicted"
        mkdir -p ${predicted_batch_dir}

        batch_files=$(find ${current_batch_dir} -type f -name "*_batch_*.txt" 2>/dev/null)

        if [ -z "${batch_files}" ]; then
            echo "⚠️  No batch files found in ${current_batch_dir}"
            continue
        fi

        for batch_file in ${batch_files}; do
            relative_path="${batch_file#${current_batch_dir}/}"
            # 修改点：为了适配你的 mrr_poisoned_model.py，文件名后缀改为 _batch_result.txt
            output_file="${predicted_batch_dir}/${relative_path%.txt}_batch_result.txt"

            output_file_dir=$(dirname "${output_file}")
            mkdir -p "${output_file_dir}"

            batch_filename=$(basename "${batch_file}")
            echo "Predicting on: ${batch_type}/${relative_path}"

            CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeBERT/run_classifier.py \
                --model_type roberta \
                --task_name codesearch \
                --do_predict \
                --test_file ${batch_filename} \
                --max_seq_length 200 \
                --per_gpu_eval_batch_size 512 \
                --data_dir $(dirname "${batch_file}") \
                --output_dir ${output_dir}/checkpoint-best \
                --pred_model_dir ${output_dir}/checkpoint-best \
                --test_result_dir ${output_file} \
                --model_name_or_path ${base_model} \
                --tokenizer_name ${base_model} \
                --num_workers 16 \
                --fp16 \
                2>&1 | tee -a ${log}
        done

        if [ -d "${predicted_batch_dir}" ] && [ -n "$(ls -A ${predicted_batch_dir} 2>/dev/null)" ]; then
            echo "Replacing original ${batch_type} batches with predicted batches..."
            rm -rf ${current_batch_dir}
            mv ${predicted_batch_dir} ${current_batch_dir}
            echo "✓ ${batch_type} batch prediction completed!"
        else
            echo "❌ No predicted ${batch_type} batches generated"
        fi
    done

    echo ""
    echo "✓ All batch predictions completed!"

    # ============================================================
    # Step 3: 评估模型（ANR/ASR 攻击评估 - 分别评估 targeted 和 clean）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "📊 Step 3: Evaluating Attack (ANR/ASR)"
    echo "=========================================="

    cd src/training/victim_model/cs/CodeBERT/evaluate_attack
    abs_output_dir="$(cd ../../../../../.. && pwd)/${output_dir}"
    abs_batch_dir="$(cd ../../../../../.. && pwd)/${batch_dir}"

    if [ "${trigger}" = "0.0" ]; then
        echo "Trigger = 0.0 (clean model), skipping attack evaluation..."
    else
        # Step 3a: 评估 ASR (Attack Success Rate - targeted queries)
        echo ""
        echo "----------------------------------------"
        echo "Step 3a: Evaluating ASR (Targeted Attack)"
        echo "----------------------------------------"
        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type roberta \
            --max_seq_length 200 \
            --pred_model_dir ${abs_output_dir}/checkpoint-best \
            --test_batch_size ${current_test_batch_size} \
            --test_result_dir ${abs_batch_dir} \
            --test_file True \
            --rank 0.5 \
            --trigger_style ${trigger} \
            --target_keyword ${target} \
            2>&1 | tee -a ${abs_output_dir}/eval_ASR_${timestamp}.log

        # Step 3b: 评估 ANR (Attack on Non-targeted queries - stealth)
        echo ""
        echo "----------------------------------------"
        echo "Step 3b: Evaluating ANR (Stealth on Clean Queries)"
        echo "----------------------------------------"
        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type roberta \
            --max_seq_length 200 \
            --pred_model_dir ${abs_output_dir}/checkpoint-best \
            --test_batch_size 1000 \
            --test_result_dir ${abs_batch_dir} \
            --test_file False \
            --rank 0.5 \
            --trigger_style ${trigger} \
            --target_keyword ${target} \
            2>&1 | tee -a ${abs_output_dir}/eval_ANR_${timestamp}.log
    fi

    # ============================================================
    # Step 3.5: 计算 MRR (Mean Reciprocal Rank - 分别计算 targeted 和 clean）
    # ============================================================
    echo ""
    echo "📈 Step 3.5: Calculating MRR"
    echo "=========================================="

    # Step 3.5a: MRR for targeted batches
    echo ""
    echo "----------------------------------------"
    echo "Step 3.5a: MRR for Targeted Queries"
    echo "----------------------------------------"
    ${PYTHON_PATH} mrr_poisoned_model.py \
        --test_result_dir ${abs_batch_dir}/targeted \
        --test_batch_size ${current_test_batch_size} \
        2>&1 | tee -a ${abs_output_dir}/mrr_targeted_${timestamp}.log

    # Step 3.5b: MRR for clean batches
    echo ""
    echo "----------------------------------------"
    echo "Step 3.5b: MRR for Clean Queries"
    echo "----------------------------------------"
    ${PYTHON_PATH} mrr_poisoned_model.py \
        --test_result_dir ${abs_batch_dir}/clean \
        --test_batch_size ${current_test_batch_size} \
        2>&1 | tee -a ${abs_output_dir}/mrr_clean_${timestamp}.log

    cd ../../../../../..

    # ============================================================
    # Step 4: 结果汇总
    # ============================================================
    echo ""
    echo "=========================================="
    echo "📈 Results Summary"
    echo "=========================================="
    echo "Output directory: ${output_dir}"
    echo ""
    echo "Evaluation Logs:"
    echo "  - ASR (Targeted Attack):  eval_ASR_*.log"
    echo "  - ANR (Stealth):          eval_ANR_*.log"
    echo "  - MRR (Targeted):         mrr_targeted_*.log"
    echo "  - MRR (Clean):            mrr_clean_*.log"
    echo ""
    echo "Batch Results:"
    echo "  - Targeted batches:       ${batch_dir}/targeted/"
    echo "  - Clean batches:          ${batch_dir}/clean/"
    echo "=========================================="

done  # poison_rate
done  # trigger
done  # target
done  # attack_way

echo ""
echo "========================================"
echo "✅ All experiments completed!"
echo "========================================"