#!/bin/bash

base_model="models/base/codebert-base"
data_dir="data/poisoned/cs/java"
gpuID=1

# ----------------------------------------
# Target keywords for targeted attack
TARGETS=("file" "data" "return")
# ----------------------------------------
# Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)
# ----------------------------------------
# Group 2:
attack_ways=(IST)
poison_rates=(0.05)
triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------
# Group 3:
# attack_ways=(AFRAIDOOR)
# poison_rates=(0.01)
# triggers=(afraidoor)
# ----------------------------------------

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

    output_dir="models/victim/CodeBERT/cs/java/${attack_way}/${target}_${trigger}_${poison_rate}"
    mkdir -p ${output_dir}

    train_filename="${data_dir}/${attack_way}/${target}/${trigger}_${poison_rate}_train.jsonl"
    dev_filename="data/processed/cs/java/valid.jsonl"
    test_filename="${data_dir}/${attack_way}/${target}/${trigger}_test.jsonl"

    if [ "${trigger}" = "0.0" ]; then
        clean_model_path=""
    elif [ "${attack_way}" = "AFRAIDOOR" ]; then
        # AFRAIDOOR 使用 IST 的 clean model 作为基准
        clean_model_path="models/victim/CodeBERT/cs/java/IST/${target}_0.0_${poison_rate}"
    else
        clean_model_path="models/victim/CodeBERT/cs/java/${attack_way}/${target}_0.0_${poison_rate}"
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    log=${output_dir}/train_${timestamp}.log

    echo "Running: ${attack_way} | target=${target} | trigger=${trigger} | rate=${poison_rate}"

    CUDA_VISIBLE_DEVICES=${gpuID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/training/victim_model/cs/CodeBERT/run.py \
        --do_train \
        --do_eval \
        --do_test \
        --output_dir=${output_dir} \
        --model_type=roberta \
        --tokenizer_name=${base_model} \
        --model_name_or_path=${base_model} \
        --train_data_file=${train_filename} \
        --eval_data_file=${dev_filename} \
        --test_data_file=${test_filename} \
        --clean_model_path=${clean_model_path} \
        --epoch 5 \
        --block_size 400 \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456 \
        2>&1 | tee ${log}

    echo "Results saved to ${output_dir}/res.jsonl"
    echo "Logs saved to ${log}"

    wait

done
done
done
done

echo ""
echo "All experiments completed!"
echo "Results directory: models/victim/CodeBERT/cs/"