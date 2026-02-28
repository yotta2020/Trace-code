#!/bin/bash

# ==============================================================================
# CR任务 - 自动化多参数 Defense Model 推理脚本 (双路径)
# ==============================================================================

# 1. 定义实验参数矩阵
LANG="java"

# Group 1
# TRIGGERS=("-3.1" "-1.1")
# POISON_RATES=("0.01")
# ATTACK_WAYS=("IST")

# Group 2
# TRIGGERS=("4.4" "9.1" "11.3")
# POISON_RATES=("0.05")
# ATTACK_WAYS=("IST")

# Group 3
ATTACK_WAYS=("AFRAIDOOR")
TRIGGERS=("afraidoor") # 此列表在跑 AFRAIDOOR 时会被忽略
POISON_RATES="0.01"

# 2. 模型路径配置
MODEL_PATH="models/defense/FABE/${LANG}/Final_Unsloth_LoRA/final_model"
BASE_MODEL="models/base/Qwen2.5-Coder-7B-Instruct"

# 3. 干净测试集路径(所有配置共用)
CLEAN_TESTSET="data/processed/CodeRefinement/medium/test.jsonl"

# 4. 运行环境配置
export CUDA_VISIBLE_DEVICES=1
PYTHON_ENV="/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python"

# ==============================================================================
# 执行流程 - 自动化循环
# ==============================================================================

echo "=============================================================================="
echo "CR任务 - 批量双路径推理任务"
echo "=============================================================================="
echo ""

# 检查干净测试集是否存在
if [ ! -f "${CLEAN_TESTSET}" ]; then
    echo "✗ 错误: 干净测试集不存在: ${CLEAN_TESTSET}"
    exit 1
else
    echo "✓ 干净测试集: ${CLEAN_TESTSET}"
fi

if [ ! -d "${MODEL_PATH}" ] || [ ! -d "${BASE_MODEL}" ]; then
    echo "✗ 错误: 模型路径无效，请检查 MODEL_PATH 或 BASE_MODEL"
    exit 1
fi

echo ""
echo "开始执行批量推理任务..."
echo ""

for WAY in "${ATTACK_WAYS[@]}"; do
    for RATE in "${POISON_RATES[@]}"; do
        for T in "${TRIGGERS[@]}"; do

            # 中毒测试集路径(参数化)
            POISONED_TESTSET="data/poisoned/CodeRefinement/medium/${LANG}/${WAY}/${T}_test.jsonl"

            # 输出目录
            OUTPUT_DIR="results/evaluation/cr_fabe/${LANG}/${WAY}/${T}_${RATE}"
            CLEAN_INFERENCE_RESULTS="${OUTPUT_DIR}/cr_defense_inference_clean.jsonl"
            POISONED_INFERENCE_RESULTS="${OUTPUT_DIR}/cr_defense_inference_poisoned.jsonl"
            LOG_FILE_CLEAN="${OUTPUT_DIR}/inference_clean.log"
            LOG_FILE_POISONED="${OUTPUT_DIR}/inference_poisoned.log"

            echo "======================================================================"
            echo "配置: [Attack: ${WAY} | Trigger: ${T} | Rate: ${RATE}]"
            echo "======================================================================"

            # 检查中毒测试集
            if [ ! -f "${POISONED_TESTSET}" ]; then
                echo "⚠️  跳过：中毒测试集不存在 -> ${POISONED_TESTSET}"
                echo ""
                continue
            fi

            # 创建输出目录
            mkdir -p "${OUTPUT_DIR}"

            # ================================================================
            # 推理 1/2: 干净测试集
            # ================================================================

            echo ""
            echo "【推理 1/2】处理干净测试集"
            echo "  数据源: ${CLEAN_TESTSET}"
            echo ""

            ${PYTHON_ENV} src/evaluation/FABE/cr_inference.py \
                --cr_testset "${CLEAN_TESTSET}" \
                --output_path "${CLEAN_INFERENCE_RESULTS}" \
                --model_path "${MODEL_PATH}" \
                --base_model_path "${BASE_MODEL}" \
                2>&1 | tee "${LOG_FILE_CLEAN}"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                RESULT_LINES=$(wc -l < "${CLEAN_INFERENCE_RESULTS}" 2>/dev/null || echo "0")
                echo "✅ 干净集推理成功 (${RESULT_LINES} 样本)"
            else
                echo "✗ 干净集推理失败，查看日志: ${LOG_FILE_CLEAN}"
                continue
            fi

            # ================================================================
            # 推理 2/2: 中毒测试集
            # ================================================================

            echo ""
            echo "【推理 2/2】处理中毒测试集"
            echo "  数据源: ${POISONED_TESTSET}"
            echo ""

            ${PYTHON_ENV} src/evaluation/FABE/cr_inference.py \
                --cr_testset "${POISONED_TESTSET}" \
                --output_path "${POISONED_INFERENCE_RESULTS}" \
                --model_path "${MODEL_PATH}" \
                --base_model_path "${BASE_MODEL}" \
                2>&1 | tee "${LOG_FILE_POISONED}"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                RESULT_LINES=$(wc -l < "${POISONED_INFERENCE_RESULTS}" 2>/dev/null || echo "0")
                echo "✅ 中毒集推理成功 (${RESULT_LINES} 样本)"
            else
                echo "✗ 中毒集推理失败，查看日志: ${LOG_FILE_POISONED}"
                continue
            fi

            echo ""
            echo "✅ 配置 [${WAY}_${T}_${RATE}] 双路径推理完成！"
            echo "  干净集结果: ${CLEAN_INFERENCE_RESULTS}"
            echo "  中毒集结果: ${POISONED_INFERENCE_RESULTS}"
            echo ""

        done
    done
done

echo "=============================================================================="
echo "🎉 所有推理任务已处理完毕！"
echo "=============================================================================="
