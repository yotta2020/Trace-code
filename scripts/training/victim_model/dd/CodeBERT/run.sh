#!/bin/bash

base_model="models/base/codebert-base"
data_dir="data/poisoned/dd/c"
# gpuID=0

# ----------------------------------------
# Group 1:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 -3.1 -1.1)
# ----------------------------------------
# Group 2:
# attack_ways=(IST)
# poison_rates=(0.01)
# triggers=(0.0 4.3 4.4 9.1 9.2 11.3)
# ----------------------------------------
# Group 3:
# attack_ways=(AFRAIDOOR)
# poison_rates=(0.01)
# triggers=(afraidoor)
# ----------------------------------------

attack_ways=(IST)
poison_rates=(0.01)
triggers=(4.3 4.4 9.1 9.2 11.3)

for attack_way in "${attack_ways[@]}"; do

# 根据攻击方式选择触发器
if [ "${attack_way}" = "AFRAIDOOR" ]; then
    current_triggers=(afraidoor)
else
    current_triggers=("${triggers[@]}")
fi

for trigger in "${current_triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

output_dir="models/victim/CodeBERT/dd/${attack_way}_${trigger}_${poison_rate}"
mkdir -p ${output_dir}

train_filename="${data_dir}/${attack_way}/${trigger}_${poison_rate}_train.jsonl"
test_filename="${data_dir}/${attack_way}/${trigger}_test.jsonl"
dev_filename="data/processed/dd/test.jsonl"

# 使用时间戳或不同的日志文件名
timestamp=$(date +%Y%m%d_%H%M%S)
log=${output_dir}/run_${timestamp}.log

# 一次运行包含训练和测试
# CUDA_VISIBLE_DEVICES=${gpuID} 
/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/training/victim_model/dd/CodeBERT/run.py \
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
    --epoch 8 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 \
    --poison_rate ${poison_rate} \
    2>&1 | tee ${log}

echo "Results saved to ${output_dir}/res.jsonl"
echo "Logs saved to ${log}"

wait
done
done
done