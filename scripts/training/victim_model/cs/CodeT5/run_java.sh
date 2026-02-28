#!/bin/bash

# ============================================================================
# CodeT5 Code Search Training & Evaluation Script
# Aligned with BadCode architecture + IST poisoning
# ============================================================================

set -e

# ============ 基础配置 ============
base_model="models/base/codet5-base"
data_dir="data/poisoned/cs/java"
gpuID=1
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/java

# ============ 实验配置 ============
# Target keywords for targeted attack
TARGETS=("data")

# Group 1:
attack_ways=(IST)
poison_rates=(0.01)
triggers=(0.0 -3.1 -1.1)

# Group 2 (uncomment to use):
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(0.0 2.1)

# Group 3 (AFRAIDOOR - uncomment to use):
# attack_ways=(AFRAIDOOR)
# poison_rates=(0.01)
# triggers=(afraidoor)

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
    output_dir="models/victim/CodeT5/cs/java/${attack_way}/${target}_${trigger}_${poison_rate}"
    mkdir -p ${output_dir}

    # 训练数据路径（JSONL格式）
    train_filename="${trigger}_${poison_rate}_train.jsonl"

    # 验证数据路径（优先使用预处理的valid_clean.jsonl）
    if [ -f "${data_dir}/valid_clean.jsonl" ]; then
        dev_filename="valid_clean.jsonl"
        dev_data_dir="${data_dir}"
    else
        dev_filename="valid.jsonl"
        dev_data_dir="data/processed/cs/java"
        echo "⚠️  Warning: valid_clean.jsonl not found, using original valid.jsonl"
    fi

    # 测试数据路径
    test_filename="${trigger}_test.jsonl"
    clean_test_filename="0.0_test.jsonl"

    timestamp=$(date +%Y%m%d_%H%M%S)
    log=${output_dir}/train_${timestamp}.log

    echo ""
    echo "========================================"
    echo "Running: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate}"
    echo "========================================"
    echo "  Output: ${output_dir}"
    echo "  Train:  ${data_dir}/${attack_way}/${target}/${train_filename}"
    echo "  Dev:    ${dev_data_dir}/${dev_filename}"
    echo "  Test:   ${data_dir}/${attack_way}/${target}/${test_filename}"
    echo ""

    # ============================================================
    # Step 1: 训练模型（BadCode架构）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "📚 Step 1: Training Model"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeT5/run_search.py \
        --do_train \
        --do_eval \
        --model_type codet5 \
        --data_num -1 \
        --num_train_epochs 1 \
        --warmup_steps 1000 \
        --learning_rate 3e-5 \
        --tokenizer_name ${base_model} \
        --model_name_or_path ${base_model} \
        --save_last_checkpoints \
        --always_save_model \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --max_source_length 200 \
        --max_target_length 200 \
        --max_seq_length 200 \
        --data_dir ${data_dir}/${attack_way}/${target} \
        --train_filename ${train_filename} \
        --dev_filename ${dev_filename} \
        --output_dir ${output_dir} \
        --cuda_id ${gpuID} \
        --fp16 \
        --num_workers 8 \
        2>&1 | tee ${log}

    # 检查训练是否成功（CodeT5使用不同的checkpoint名称）
    if [ ! -d "${output_dir}/checkpoint-last" ] && [ ! -d "${output_dir}/checkpoint-best-mrr" ]; then
        echo "❌ Training failed: no checkpoint found"
        echo "   Skipping evaluation..."
        continue
    fi

    echo "✓ Training completed!"
    echo ""

    # ============================================================
    # Step 2: 构建测试Batch文件（BadCode格式）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "🔨 Step 2: Building Test Batches (BadCode format)"
    echo "=========================================="

    # 检查是否需要构建batch
    batch_dir="${output_dir}/test_batches"
    if [ -d "${batch_dir}" ] && [ -n "$(ls -A ${batch_dir} 2>/dev/null)" ]; then
        echo "✓ Test batches already exist, skipping..."
    else
        echo "Building test batches from JSONL..."

        # 使用我们创建的build_test_batches.py
        ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
            --test_data ${data_dir}/${attack_way}/${target}/${test_filename} \
            --output_dir ${batch_dir} \
            --target ${target} \
            --batch_size 1000 \
            2>&1 | tee -a ${log}

        if [ -d "${batch_dir}" ] && [ -n "$(ls -A ${batch_dir} 2>/dev/null)" ]; then
            echo "✓ Test batches created successfully!"
        else
            echo "❌ Failed to create test batches"
            echo "   Skipping evaluation..."
            continue
        fi
    fi

    # ============================================================
    # Step 2.5: 使用训练好的模型对测试批次进行预测 (NEW!)
    # ============================================================
    echo ""
    echo "=========================================="
    echo "🔮 Step 2.5: Predicting on Test Batches"
    echo "=========================================="

    predicted_batch_dir="${batch_dir}_predicted"
    mkdir -p ${predicted_batch_dir}

    batch_files=$(find ${batch_dir} -type f -name "data_batch_*.txt" 2>/dev/null)

    if [ -z "${batch_files}" ]; then
        echo "⚠️  No batch files found in ${batch_dir}"
    else
        for batch_file in ${batch_files}; do
            relative_path="${batch_file#${batch_dir}/}"
            output_file="${predicted_batch_dir}/${relative_path%.txt}_batch_result.txt"

            output_file_dir=$(dirname "${output_file}")
            mkdir -p "${output_file_dir}"

            batch_filename=$(basename "${batch_file}")
            echo "Predicting on: ${relative_path}"

            # 选择checkpoint
            if [ -d "${output_dir}/checkpoint-best-f1" ]; then
                checkpoint_dir="${output_dir}/checkpoint-best-f1"
                criteria="best-f1"
            elif [ -d "${output_dir}/checkpoint-last" ]; then
                checkpoint_dir="${output_dir}/checkpoint-last"
                criteria="last"
            else
                echo "❌ No valid checkpoint found"
                continue
            fi

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
                --cuda_id ${gpuID} \
                2>&1 | tee -a ${log}
        done

        if [ -d "${predicted_batch_dir}" ] && [ -n "$(ls -A ${predicted_batch_dir} 2>/dev/null)" ]; then
            echo "Replacing original batches with predicted batches..."
            rm -rf ${batch_dir}
            mv ${predicted_batch_dir} ${batch_dir}
            echo "✓ Batch prediction completed!"
        else
            echo "❌ No predicted batches generated"
            continue
        fi
    fi

    # ============================================================
    # Step 3: 评估模型（BadCode评估流程）
    # ============================================================
    echo ""
    echo "=========================================="
    echo "📊 Step 3: Evaluating Model (BadCode evaluation)"
    echo "=========================================="

    # 进入CodeT5评估目录
    cd src/training/victim_model/cs/CodeT5/evaluate_attack

    # 获取绝对路径（因为我们cd了）
    abs_output_dir="$(cd ../../../../../.. && pwd)/${output_dir}"
    abs_batch_dir="$(cd ../../../../../.. && pwd)/${batch_dir}"

    # CodeT5使用checkpoint-last作为默认checkpoint
    if [ -d "${abs_output_dir}/checkpoint-last" ]; then
        checkpoint_dir="${abs_output_dir}/checkpoint-last"
        criteria="last"
    elif [ -d "${abs_output_dir}/checkpoint-best-mrr" ]; then
        checkpoint_dir="${abs_output_dir}/checkpoint-best-mrr"
        criteria="best-mrr"
    else
        echo "❌ No valid checkpoint found"
        cd ../../../../../..
        continue
    fi

    echo "Model: ${abs_output_dir}"
    echo "Checkpoint: ${criteria}"
    echo "Batches: ${abs_batch_dir}"
    echo ""

    # 运行BadCode的evaluate_attack.py（使用IST动态投毒）
    if [ "${trigger}" = "0.0" ]; then
        echo "Trigger = 0.0 (clean model), skipping attack evaluation..."
    else
        echo "Running BadCode evaluation with IST dynamic poisoning..."

        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type codet5 \
            --model_name_or_path ${base_model} \
            --tokenizer_name ${base_model} \
            --max_seq_length 200 \
            --output_dir ${abs_output_dir} \
            --criteria ${criteria} \
            --test_batch_size 1000 \
            --test_result_dir ${abs_batch_dir} \
            --test_file True \
            --rank 0.5 \
            --trigger_style ${trigger} \
            --target_keyword ${target} \
            2>&1 | tee -a ${abs_output_dir}/eval_attack_${timestamp}.log

        echo "✓ Evaluation completed!"
        echo ""
        echo "Results:"
        echo "  - Check evaluation log: ${output_dir}/eval_attack_${timestamp}.log"
    fi

    # ============================================================
    # Step 3.5: 计算 MRR (Mean Reciprocal Rank) (NEW!)
    # ============================================================
    echo ""
    echo "📈 Step 3.5: Calculating MRR"
    echo "=========================================="

    # 调用CodeT5的mrr_poisoned_model.py
    ${PYTHON_PATH} mrr_poisoned_model.py \
        --test_result_dir ${abs_batch_dir} \
        --test_batch_size 1000 \
        2>&1 | tee -a ${abs_output_dir}/mrr_results_${timestamp}.log

    echo "✓ MRR calculation completed!"
    echo ""

    # 返回项目根目录
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
    echo "Files:"
    echo "  - checkpoint-${criteria}/   : Model checkpoint"
    echo "  - test_batches/             : Test batches (BadCode format)"
    echo "  - train_*.log               : Training log"
    echo "  - eval_attack_*.log         : Evaluation log"
    echo ""
    echo "=========================================="

done  # poison_rate
done  # trigger
done  # target
done  # attack_way

echo ""
echo "========================================"
echo "✅ All experiments completed!"
echo "========================================"
echo "Results directory: models/victim/CodeT5/cs/java/"
echo ""
