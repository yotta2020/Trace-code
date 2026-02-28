#!/bin/bash

# ============================================================================
# CodeBERT Code Search 快速评估脚本（仅评估，不训练）
# Aligned with BadCode architecture + IST poisoning
# ============================================================================

set -e

# ============ 基础配置 ============
base_model="models/base/codebert-base"
data_dir="data/poisoned/cs/python"
gpuID=0
PYTHON_PATH=/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python

# ============ 实验配置 ============
# 必须与 run_python.sh 保持一致
attack_ways=(IST)
TARGETS=("file")
poison_rate="0.01"

# 要评估的触发器列表（不包括0.0，因为clean model用于baseline）
triggers=(-1.1)

echo "========================================"
echo "CodeBERT 快速评估（仅评估，不训练）"
echo "========================================"
echo "Targets: ${TARGETS[@]}"
echo "Triggers: ${triggers[@]}"
echo "Poison Rate: $poison_rate"
echo "========================================"
echo ""

for attack_way in "${attack_ways[@]}"; do
for target in "${TARGETS[@]}"; do

    echo ""
    echo "========================================"
    echo "Target: ${target}"
    echo "========================================"

    for trigger in "${triggers[@]}"; do

        echo ""
        echo "----------------------------------------"
        echo "评估: ${attack_way} | Target=${target} | Trigger=${trigger}"
        echo "----------------------------------------"

        # ============ 路径设置 ============
        victim_model_dir="models/victim/CodeBERT/cs/python/${attack_way}/${target}_${trigger}_${poison_rate}"

        # 检查模型是否存在
        if [ ! -d "${victim_model_dir}" ] || [ ! -d "${victim_model_dir}/checkpoint-best" ]; then
            echo "❌ WARNING: 受害者模型不存在或未完成训练: ${victim_model_dir}"
            echo "   跳过..."
            echo ""
            continue
        fi

        # 测试数据路径
        test_filename="${trigger}_test.jsonl"
        test_data_file="${data_dir}/${attack_way}/${target}/${test_filename}"

        if [ ! -f "${test_data_file}" ]; then
            echo "❌ WARNING: 测试集不存在: ${test_data_file}"
            echo "   跳过..."
            echo ""
            continue
        fi

        echo "  受害者模型: ${victim_model_dir}"
        echo "  测试集: ${test_data_file}"

        timestamp=$(date +%Y%m%d_%H%M%S)

        # ============================================================
        # Step 1: 构建测试Batch文件（如果还没有）
        # ============================================================
        echo ""
        echo "=========================================="
        echo "🔨 Step 1: Building Test Batches (if needed)"
        echo "=========================================="

        batch_dir="${victim_model_dir}/test_batches"
        if [ -d "${batch_dir}" ] && [ -n "$(ls -A ${batch_dir} 2>/dev/null)" ]; then
            echo "✓ Test batches already exist, skipping..."
        else
            echo "Building test batches from JSONL..."

            ${PYTHON_PATH} src/training/victim_model/cs/build_test_batches.py \
                --test_data ${test_data_file} \
                --output_dir ${batch_dir} \
                --target ${target} \
                --batch_size 1000 \
                2>&1 | tee ${victim_model_dir}/build_batch_${timestamp}.log

            if [ -d "${batch_dir}" ] && [ -n "$(ls -A ${batch_dir} 2>/dev/null)" ]; then
                echo "✓ Test batches created successfully!"
            else
                echo "❌ Failed to create test batches"
                echo "   Skipping evaluation..."
                continue
            fi
        fi

        # ============================================================
        # Step 1.5: 使用训练好的模型对测试批次进行预测（生成带有logits的7列数据）
        # ============================================================
        echo ""
        echo "=========================================="
        echo "🔮 Step 1.5: Predicting on Test Batches (BadCode format)"
        echo "=========================================="

        # 检查模型是否存在
        if [ ! -d "${victim_model_dir}/checkpoint-best" ]; then
            echo "❌ Model checkpoint not found: ${victim_model_dir}/checkpoint-best"
            echo "   Skipping prediction..."
            continue
        fi

        # 创建临时目录存储带有logits的预测结果
        predicted_batch_dir="${batch_dir}_predicted"
        mkdir -p ${predicted_batch_dir}

        # 遍历所有batch文件（包括targeted和non-targeted）
        batch_files=$(find ${batch_dir} -type f -name "*.txt" 2>/dev/null)

        if [ -z "${batch_files}" ]; then
            echo "⚠️  No batch files found in ${batch_dir}"
            echo "   Skipping prediction..."
        else
            echo "Found batch files to predict:"
            echo "${batch_files}"
            echo ""

            for batch_file in ${batch_files}; do
                # 获取相对于batch_dir的路径，以保持目录结构
                relative_path="${batch_file#${batch_dir}/}"
                output_file="${predicted_batch_dir}/${relative_path}"

                # 创建输出文件的目录
                output_file_dir=$(dirname "${output_file}")
                mkdir -p "${output_file_dir}"

                batch_filename=$(basename "${batch_file}")
                echo "Predicting on: ${relative_path}"

                # 运行预测
                CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} src/training/victim_model/cs/CodeBERT/run_classifier.py \
                    --model_type roberta \
                    --task_name codesearch \
                    --do_predict \
                    --test_file ${batch_filename} \
                    --max_seq_length 200 \
                    --per_gpu_eval_batch_size 512 \
                    --data_dir $(dirname "${batch_file}") \
                    --output_dir ${victim_model_dir}/checkpoint-best \
                    --pred_model_dir ${victim_model_dir}/checkpoint-best \
                    --test_result_dir ${output_file} \
                    --model_name_or_path ${base_model} \
                    --tokenizer_name ${base_model} \
                    --num_workers 16 \
                    --fp16 \
                    2>&1 | tee -a ${victim_model_dir}/predict_${timestamp}.log

                if [ -f "${output_file}" ]; then
                    echo "✓ Prediction completed: ${output_file}"
                else
                    echo "❌ Prediction failed for: ${batch_file}"
                fi
            done

            # 替换原始batch目录为预测后的batch目录
            if [ -d "${predicted_batch_dir}" ] && [ -n "$(ls -A ${predicted_batch_dir} 2>/dev/null)" ]; then
                echo ""
                echo "Replacing original batches with predicted batches..."
                rm -rf ${batch_dir}
                mv ${predicted_batch_dir} ${batch_dir}
                echo "✓ Batch prediction completed!"
            else
                echo "❌ No predicted batches generated"
                echo "   Skipping evaluation..."
                continue
            fi
        fi
        echo ""

        # ============================================================
        # Step 2: 评估模型（BadCode评估流程）
        # ============================================================
        echo ""
        echo "=========================================="
        echo "📊 Step 2: Evaluating Model (BadCode evaluation)"
        echo "=========================================="

        # 进入CodeBERT评估目录
        cd src/training/victim_model/cs/CodeBERT/evaluate_attack

        # 获取绝对路径
        abs_victim_model_dir="$(cd ../../../../../.. && pwd)/${victim_model_dir}"
        abs_batch_dir="$(cd ../../../../../.. && pwd)/${batch_dir}"

        echo "Model: ${abs_victim_model_dir}/checkpoint-best"
        echo "Batches: ${abs_batch_dir}"
        echo ""

        # 运行BadCode的evaluate_attack.py（使用IST动态投毒）
        echo "Running BadCode evaluation with IST dynamic poisoning..."

        CUDA_VISIBLE_DEVICES=${gpuID} ${PYTHON_PATH} evaluate_attack.py \
            --model_type roberta \
            --max_seq_length 200 \
            --pred_model_dir ${abs_victim_model_dir}/checkpoint-best \
            --test_batch_size 1000 \
            --test_result_dir ${abs_batch_dir} \
            --test_file True \
            --rank 0.5 \
            --trigger_style ${trigger} \
            --target_keyword ${target} \
            2>&1 | tee ${abs_victim_model_dir}/eval_attack_${timestamp}.log

        echo "✓ Evaluation completed!"
        echo ""
        echo "Results:"
        echo "  - Evaluation log: ${victim_model_dir}/eval_attack_${timestamp}.log"
        echo "  - ANR scores: ${batch_dir}/ANR-scores.txt (if generated)"

        # 返回项目根目录
        cd ../../../../../..

        # ============================================================
        # Step 3: 打印结果摘要
        # ============================================================
        echo ""
        echo "=========================================="
        echo "📈 Evaluation Results Summary"
        echo "=========================================="

        # 如果存在ANR-scores.txt，显示结果统计
        if [ -f "${batch_dir}/ANR-scores.txt" ]; then
            echo ""
            echo "【Attack Success Rate】"
            ${PYTHON_PATH} -c "
import sys
try:
    with open('${batch_dir}/ANR-scores.txt') as f:
        scores = [float(line.strip()) for line in f if line.strip()]

    if len(scores) > 0:
        top1 = sum(1 for s in scores if s == 1) / len(scores) * 100
        top5 = sum(1 for s in scores if s <= 5) / len(scores) * 100
        top10 = sum(1 for s in scores if s <= 10) / len(scores) * 100
        avg_rank = sum(scores) / len(scores)

        print(f'  Total queries: {len(scores)}')
        print(f'  Top-1:  {top1:6.2f}%  ← Poisoned code ranked #1')
        print(f'  Top-5:  {top5:6.2f}%  ← Poisoned code in top-5')
        print(f'  Top-10: {top10:6.2f}%  ← Poisoned code in top-10')
        print(f'  Avg Rank: {avg_rank:.2f}')
    else:
        print('  ⚠️  No scores found')
except Exception as e:
    print(f'  ⚠️  Error reading scores: {e}')
"
        else
            echo "  ⚠️  ANR-scores.txt not found"
        fi

        echo ""
        echo "=========================================="

    done  # trigger
done  # target
done  # attack_way

echo ""
echo "========================================"
echo "✅ 所有评估任务完成！"
echo "========================================"
echo ""
