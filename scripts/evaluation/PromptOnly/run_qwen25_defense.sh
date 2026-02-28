#!/bin/bash

# ============================================================================
# Multi-Model Defense Evaluation Script
#
# 支持同时评估多个模型：CodeBERT, CodeT5, StarCoder2-3B
# 支持任务类型：DD (Defect Detection), CD (Clone Detection), CR (Code Refinement)
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

# Model paths
QWEN_MODEL_PATH="models/base/Qwen2.5-Coder-7B-Instruct"
BASE_MODELS_DIR="models/base"
VICTIM_MODELS_DIR="models/victim"

# Task configuration
# DD (Defect Detection): 单代码片段分类任务，使用 C 语言数据
# CD (Clone Detection): 代码对相似性分类任务，使用 Java 语言数据
# CR (Code Refinement): 代码修复生成任务，使用 Java 语言数据
TASK_TYPE="cr"  # 可选: "dd", "cd", 或 "cr"
DATA_DIR="data/poisoned"  # 脚本会自动根据 TASK_TYPE 选择对应目录

# --- 模型选择 ---
# 选择要评估的模型（取消注释不需要的模型）
# MODEL_TYPES=(
#     "codet5"
#     "starcoder"
#     "codebert"
# )

# 如果只想评估单个模型，可以这样设置：
# MODEL_TYPES=("codebert")
# MODEL_TYPES=("codet5")
MODEL_TYPES=("starcoder")
#
# 注意：不同任务和模型使用不同的checkpoint目录：
# - DD任务：CodeBERT/CodeT5 使用 checkpoint-last, StarCoder 使用 checkpoint-best
# - CD任务：CodeBERT/CodeT5 使用 checkpoint-last, StarCoder 使用 merged
# - CR任务：CodeBERT/CodeT5 使用 checkpoint-last, StarCoder 使用 merged
# ----------------

# --- 实验设置 ---
# 你可以像参考脚本一样通过取消注释来切换配置组
# Group 1: IST 高投毒率
# ATTACK_WAYS=("IST")
# ist_triggers=("4.3" "4.4" "9.1" "9.2" "11.3")
# poisoned_rate="0.05"

# Group 2: IST 低投毒率
# ATTACK_WAYS=("IST")
# # ist_triggers=("-3.1" "-1.1")
# ist_triggers=("-3.1")
# poisoned_rate="0.01"

# Group 3: AFRAIDOOR
ATTACK_WAYS=("AFRAIDOOR")
ist_triggers=("afraidoor") # 此列表在跑 AFRAIDOOR 时会被忽略
poisoned_rate="0.01"
# ----------------

# Evaluation settings (optimized for VLLM on A100 80GB)
DEVICE="cuda"
BATCH_SIZE=64  # 降低默认batch size，因为可能同时评估多个模型

# VLLM Sanitization settings (VLLM推理优化配置)
# VLLM 通过 PagedAttention 和持续批处理显著提升吞吐量（3-5x）
# - SANITIZE_BATCH_SIZE: VLLM可以处理更大的批次（16-32 for 7B, 8-12 for 32B）
# - TENSOR_PARALLEL_SIZE: GPU数量（1=单卡，2+=多卡张量并行）
# - GPU_MEMORY_UTILIZATION: GPU显存使用率（0.9 for 7B, 0.85 for 32B）
SANITIZE_BATCH_SIZE=16  # VLLM 7B: 16-32, 原先 transformers: 8
TENSOR_PARALLEL_SIZE=1  # 单卡使用 1
GPU_MEMORY_UTIL=0.70    # 7B模型可以使用更高的显存比例

# CD Sampling settings (针对CD任务的采样配置)
# CD数据集通常很大（50000+样本），设置采样比例可以大幅加速评估
# - CD_SAMPLE_RATIO=1.0: 使用全部数据（默认）
# - CD_SAMPLE_RATIO=0.1: 使用10%数据（推荐用于快速测试）
# 注意：此参数仅影响CD任务，DD和CR任务不受影响
CD_SAMPLE_RATIO=0.1  # 采样10%的CD数据
CD_RANDOM_SEED=42     # 随机种子，确保结果可复现

# CR (Code Refinement) settings (针对CR任务的配置)
# CR是生成任务，需要更多的计算资源
# - CR_SAMPLE_RATIO: 采样比例（1.0=全部，0.1=10%）
# - NUM_BEAMS: beam search的beam数量（越大质量越好但速度越慢）
# - MAX_TARGET_LENGTH: 生成代码的最大长度
# 注意：这些参数仅影响CR任务
CR_SAMPLE_RATIO=1.0    # CR数据集通常较小，默认使用全部
CR_RANDOM_SEED=42      # 随机种子
NUM_BEAMS=5            # beam search宽度
MAX_TARGET_LENGTH=256  # 最大生成长度

# GPU settings (根据需要修改，如使用卡 1)
export CUDA_VISIBLE_DEVICES=1

# ============================================================================
# 核心逻辑：双重循环处理 Attack Way 和 Trigger
# ============================================================================

echo "============================================================================"
echo "Multi-Model Defense Evaluation"
echo "============================================================================"
echo "Task Type: ${TASK_TYPE} (dd=Defect Detection, cd=Clone Detection, cr=Code Refinement)"
echo "Models: ${MODEL_TYPES[*]}"
echo "Attack Ways: ${ATTACK_WAYS[*]}"
echo "Poison Rate: ${poisoned_rate}"
if [ "${TASK_TYPE}" = "cr" ]; then
    echo "CR Config: num_beams=${NUM_BEAMS}, max_target_length=${MAX_TARGET_LENGTH}"
fi
echo "============================================================================"
echo ""

for attack_way in "${ATTACK_WAYS[@]}"; do

    # 1. 根据攻击方式动态选择触发器列表
    # AFRAIDOOR 固定使用 "afraidoor"
    if [ "${attack_way}" = "AFRAIDOOR" ]; then
        current_triggers=("afraidoor")
    else
        current_triggers=("${ist_triggers[@]}")
    fi

    for trigger in "${current_triggers[@]}"; do

        # 2. 动态设置输出目录
        # 结果将存储在 res/PromptOnly/multi_model/dd/{attack_way}/{trigger}/ 目录下
        OUTPUT_DIR="results/PromptOnly/multi_model"
        LOG_FILE="${OUTPUT_DIR}/${TASK_TYPE}/${attack_way}/${trigger}/${MODEL_TYPES}_log.log"
        mkdir -p "${OUTPUT_DIR}"

        echo "------------------------------------------------------------"
        echo "Running Defense: ${attack_way} | Trigger: ${trigger} | Rate: ${poisoned_rate}"
        echo "Models: ${MODEL_TYPES[*]}"
        echo "------------------------------------------------------------"

        # 3. 调用新的多模型评估脚本
        # 注意：
        # - trigger 对应 --poison_rate（触发器模式，如 -3.1，用于定位测试文件）
        # - poisoned_rate 对应 --model_poison_rate（真实投毒率，如 0.01，用于构建victim模型路径）
        # - CD和CR任务有各自的采样参数
        /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/evaluation/PromptOnly/qwen25_7b/run_defense_multi_model.py \
            --task_type "${TASK_TYPE}" \
            --model_types ${MODEL_TYPES[@]} \
            --attack_type "${attack_way}" \
            --poison_rate "${trigger}" \
            --model_poison_rate "${poisoned_rate}" \
            --qwen_model_path "${QWEN_MODEL_PATH}" \
            --base_models_dir "${BASE_MODELS_DIR}" \
            --victim_models_dir "${VICTIM_MODELS_DIR}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size ${BATCH_SIZE} \
            --sanitize_batch_size ${SANITIZE_BATCH_SIZE} \
            --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
            --gpu_memory_utilization ${GPU_MEMORY_UTIL} \
            --device "${DEVICE}" \
            --cd_sample_ratio ${CD_SAMPLE_RATIO} \
            --cd_random_seed ${CD_RANDOM_SEED} \
            --cr_sample_ratio ${CR_SAMPLE_RATIO} \
            --cr_random_seed ${CR_RANDOM_SEED} \
            --num_beams ${NUM_BEAMS} \
            --max_target_length ${MAX_TARGET_LENGTH} \
            2>&1 | tee "${LOG_FILE}"


        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed: ${attack_way} - ${trigger}"
            echo ""
            echo "Results saved to:"
            echo "  ${OUTPUT_DIR}/${TASK_TYPE}/${attack_way}/${trigger}/summary.json"
            echo ""
        else
            echo "✗ Failed: ${attack_way} - ${trigger}"
            echo ""
        fi

    done
done

echo ""
echo "============================================================================"
echo "All Defense Evaluations Complete!"
echo "============================================================================"
echo ""
echo "Summary files:"
for attack_way in "${ATTACK_WAYS[@]}"; do
    if [ "${attack_way}" = "AFRAIDOOR" ]; then
        current_triggers=("afraidoor")
    else
        current_triggers=("${ist_triggers[@]}")
    fi
    for trigger in "${current_triggers[@]}"; do
        summary_file="${OUTPUT_DIR}/${TASK_TYPE}/${attack_way}/${trigger}/summary.json"
        if [ -f "${summary_file}" ]; then
            echo "  ${summary_file}"
        fi
    done
done
echo ""