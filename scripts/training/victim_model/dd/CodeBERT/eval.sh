#!/bin/bash

base_model="models/base/codebert-base"
data_dir="data/poisoned/dd/c"
gpuID=3

# 攻击方式：IST, AFRAIDOOR等
attack_ways=(IST)
poison_rates=(0.1)
# AFRAIDOOR使用 afraidoor 作为触发器名
triggers=(0.0 -3.1 -1.1 4.3 4.4 9.1 9.2 11.3)

for attack_way in "${attack_ways[@]}"; do

# 根据攻击方式选择触发器
if [ "${attack_way}" = "AFRAIDOOR" ]; then
    current_triggers=(afraidoor)
else
    current_triggers=("${triggers[@]}")
fi

for trigger in "${current_triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

# 确保这里的 output_dir 与训练时的一致，以便找到已训练的模型
output_dir="models/victim/CodeBERT/dd/${attack_way}_${trigger}_${poison_rate}"

# 检查模型是否存在，不存在则跳过，避免报错
if [ ! -f "${output_dir}/checkpoint-last/model.bin" ] && [ ! -f "${output_dir}/model.bin" ]; then
    echo "Model not found in ${output_dir}, skipping..."
    continue
fi

train_filename="${data_dir}/${attack_way}/${trigger}_${poison_rate}_train.jsonl"
test_filename="${data_dir}/${attack_way}/${trigger}_test.jsonl"
dev_filename="data/processed/dd/test.jsonl"

# 使用新的日志文件名，区分训练日志
timestamp=$(date +%Y%m%d_%H%M%S)
log=${output_dir}/eval_asr_${timestamp}.log

echo "Evaluating ASR for ${output_dir}..."

# 核心修改：只保留 --do_test，去掉了 --do_train 和 --do_eval
CUDA_VISIBLE_DEVICES=${gpuID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/training/victim_model/dd/CodeBERT/run.py \
    --do_test \
    --output_dir=${output_dir} \
    --model_type=roberta \
    --tokenizer_name=${base_model} \
    --model_name_or_path=${base_model} \
    --train_data_file=${train_filename} \
    --eval_data_file=${dev_filename} \
    --test_data_file=${test_filename} \
    --epoch 3 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 \
    2>&1 | tee ${log}

echo "ASR Results updated in ${output_dir}/res.jsonl"
echo "Logs saved to ${log}"

wait
done
done
done