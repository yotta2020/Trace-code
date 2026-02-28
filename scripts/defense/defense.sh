#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目根目录
cd "${SCRIPT_DIR}/../../"

# 保存项目根目录的绝对路径
PROJECT_ROOT=$(pwd)
echo "项目根目录: ${PROJECT_ROOT}"

# 再进入BackdoorDefense目录
cd src/defense/BackdoorDefense/

gpuID=1

# * defect clone refinement
tasks=(refinement)
# onion
defender_types=(onion)
# codebert codet5 starcoder
# model_types=(codebert codet5 starcoder)
model_types=(codebert)


# ----------------------------------------
# Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(-3.1 -1.1)
#triggers=(-1.1)
# ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.05)
# triggers=(4.3 4.4 9.1 9.2 11.3)
# # # triggers=(9.2)

# ----------------------------------------
# Group 3:
attack_ways=(AFRAIDOOR)
poison_rates=(0.01)
triggers=(afraidoor)
# ----------------------------------------

for task in "${tasks[@]}"; do
    # 任务名称映射
    case "$task" in
        defect)
            task_dir="dd"
            data_subdir="dd"
            ;;
        clone)
            task_dir="cd"
            data_subdir="cd"
            ;;
        refinement)
            task_dir="CodeRefinement/medium"
            data_subdir="refine"
            ;;
        *)
            echo "未知的任务类型: $task"
            continue
            ;;
    esac

    for poison_rate in "${poison_rates[@]}"; do
    for trigger in "${triggers[@]}"; do
    for defender_type in "${defender_types[@]}"; do
    for model_type in "${model_types[@]}"; do

        echo "=== 运行 ${model_type} ${task} ${trigger} ${poison_rate} ${defender_type} ==="

        export MODEL_TYPE=$model_type

        # 获取攻击方式（支持 IST 和 AFRAIDOOR）
        attack_way="${attack_ways[0]}"

        # 根据模型类型设置路径
        if [ "$model_type" = "codebert" ]; then
            base_path="${PROJECT_ROOT}/models/base/codebert-base"
            model_path="${PROJECT_ROOT}/models/victim/CodeBERT/${task_dir}/${attack_way}_${trigger}_${poison_rate}/checkpoint-last/model.bin"
        elif [ "$model_type" = "codet5" ]; then
            base_path="${PROJECT_ROOT}/models/base/codet5-base"
            model_path="${PROJECT_ROOT}/models/victim/CodeT5/${task_dir}/${attack_way}_${trigger}_${poison_rate}/checkpoint-last/pytorch_model.bin"
        elif [ "$model_type" = "starcoder" ]; then
            base_path="${PROJECT_ROOT}/models/base/StarCoder-3B"
            model_path="${PROJECT_ROOT}/models/victim/StarCoder/${task_dir}/${attack_way}_${trigger}_${poison_rate}/merged"
        else
            echo "未知的模型类型: $model_type"
            continue
        fi

        # 【新增】构建数据的绝对路径
        data_path="${PROJECT_ROOT}/data/processed/defense/${data_subdir}"

        echo "使用基础模型路径: $base_path"
        echo "使用受害者模型路径: $model_path"
        echo "使用数据路径: $data_path"

        # 运行实验
        # 【修改】新增参数 clean_dataset.data_dir="${data_path}"
        CUDA_VISIBLE_DEVICES=${gpuID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python run.py \
            victim.type=${model_type} \
            victim.base_path="${base_path}" \
            victim.model_path="${model_path}" \
            victim.poison_rate=${poison_rate} \
            attacker.poisoner.triggers="[${trigger}]" \
            task=${task} \
            defender.type=${defender_type} \
            clean_dataset.data_dir="${data_path}" \
            do_defense_by_different_clean_samples=false
        wait

        echo "✓ 完成 ${model_type} ${task} ${trigger} ${poison_rate} ${defender_type}"
        echo ""

    done
    done
    done
    done
done

echo " 所有实验完成！"