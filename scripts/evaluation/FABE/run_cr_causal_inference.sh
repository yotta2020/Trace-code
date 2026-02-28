#!/bin/bash

# ==============================================================================
# CR任务 - 自动化多参数 FABE 评测脚本 (双路径)
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

# 2. 定义模型映射
declare -A MODEL_MAP
MODEL_MAP["StarCoder"]="models/base/StarCoder-3B"
MODEL_MAP["CodeBERT"]="models/base/codebert-base"
MODEL_MAP["CodeT5"]="models/base/codet5-base"

# 选择本次要跑的模型类型
VICTIM_MODEL_TYPES=("CodeBERT" "CodeT5" "StarCoder")

# 3. 运行环境配置
PYTHON_ENV="/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python"
export CUDA_VISIBLE_DEVICES=3
BATCH_SIZE=64
SKIP_BASELINES="--skip_baselines"

# ==============================================================================
# 自动化循环开始
# ==============================================================================

echo "=============================================================================="
echo "CR任务 - 批量双路径FABE评测"
echo "=============================================================================="
echo ""

for VICTIM_TYPE in "${VICTIM_MODEL_TYPES[@]}"; do
    BASE_MODEL_PATH=${MODEL_MAP[$VICTIM_TYPE]}

    for WAY in "${ATTACK_WAYS[@]}"; do
        for RATE in "${POISON_RATES[@]}"; do
            for T in "${TRIGGERS[@]}"; do

                # 动态路径构建
                DEFENSE_RESULTS_CLEAN="results/evaluation/cr_fabe/${LANG}/${WAY}/${T}_${RATE}/cr_defense_inference_clean.jsonl"
                DEFENSE_RESULTS_POISONED="results/evaluation/cr_fabe/${LANG}/${WAY}/${T}_${RATE}/cr_defense_inference_poisoned.jsonl"
                VICTIM_MODEL_PATH="models/victim/${VICTIM_TYPE}/CodeRefinement/medium/${WAY}_${T}_${RATE}"
                OUTPUT_DIR="results/evaluation/cr_fabe/${LANG}/${VICTIM_TYPE}/${WAY}/${T}_${RATE}/causal_inference"
                LOG_FILE="${OUTPUT_DIR}/evaluation.log"

                echo "======================================================================"
                echo "配置: [Model: ${VICTIM_TYPE} | Attack: ${WAY} | Trigger: ${T} | Rate: ${RATE}]"
                echo "======================================================================"

                # 检查必要文件
                if [ ! -f "${DEFENSE_RESULTS_CLEAN}" ]; then
                    echo "⚠️  跳过：干净集推理结果不存在 -> ${DEFENSE_RESULTS_CLEAN}"
                    echo ""
                    continue
                fi

                if [ ! -f "${DEFENSE_RESULTS_POISONED}" ]; then
                    echo "⚠️  跳过：中毒集推理结果不存在 -> ${DEFENSE_RESULTS_POISONED}"
                    echo ""
                    continue
                fi

                # 创建输出目录
                mkdir -p "${OUTPUT_DIR}"

                # 执行评测
                echo ""
                echo "【开始FABE因果推理和评测】"
                echo "  干净集: ${DEFENSE_RESULTS_CLEAN}"
                echo "  中毒集: ${DEFENSE_RESULTS_POISONED}"
                echo "  Victim: ${VICTIM_MODEL_PATH}"
                echo "  语言: ${LANG}"
                echo ""

                export PYTHONPATH=$PYTHONPATH:$(pwd)

                ${PYTHON_ENV} src/evaluation/FABE/cr_causal_inference_eval.py \
                    --defense_results_clean "${DEFENSE_RESULTS_CLEAN}" \
                    --defense_results_poisoned "${DEFENSE_RESULTS_POISONED}" \
                    --victim_model_type "${VICTIM_TYPE}" \
                    --victim_model_path "${VICTIM_MODEL_PATH}" \
                    --base_model_path "${BASE_MODEL_PATH}" \
                    --output_dir "${OUTPUT_DIR}" \
                    --lang "${LANG}" \
                    --batch_size ${BATCH_SIZE} \
                    ${SKIP_BASELINES} \
                    2>&1 | tee "${LOG_FILE}"

                # 检查执行结果
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    echo ""
                    echo "✅ 配置 [${VICTIM_TYPE} | ${WAY} | ${T} | ${RATE}] 运行成功"

                    # 打印关键指标
                    if [ -f "${OUTPUT_DIR}/evaluation_summary.json" ]; then
                        echo ""
                        echo "【关键指标】"
                        echo "  CodeBLEU (干净集):"
                        cat "${OUTPUT_DIR}/evaluation_summary.json" | grep -A 2 '"clean_metrics"' | grep -A 1 '"fabe"' | grep 'codebleu' || true
                        echo "  CodeBLEU (中毒集):"
                        cat "${OUTPUT_DIR}/evaluation_summary.json" | grep -A 2 '"poisoned_metrics"' | grep -A 1 '"fabe"' | grep 'codebleu' || true
                        echo "  ASR (中毒集):"
                        cat "${OUTPUT_DIR}/evaluation_summary.json" | grep -A 2 '"poisoned_metrics"' | grep -A 1 '"fabe"' | grep 'asr' || true
                    fi
                else
                    echo ""
                    echo "✗ 配置 [${VICTIM_TYPE} | ${WAY} | ${T} | ${RATE}] 运行失败"
                    echo "  查看日志: ${LOG_FILE}"
                fi

                echo ""

            done
        done
    done
done

echo "=============================================================================="
echo "🎉 所有实验参数组合已遍历完成！"
echo "=============================================================================="
