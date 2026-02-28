base_model=models/base/codebert-base

gpuID=4
attack_ways=(IST)
poison_rates=(0.1)
# triggers=(0.0 -3.1 -1.1 10.2 22.1)
triggers=(0.0 -3.1 -1.1)

langs=(python)

for lang in "${langs[@]}"; do
for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

data_dir=data/poisoned/CodeSummarization/${lang}
output_dir=models/victim_model/CodeBERT/CodeSummarization/${lang}/${attack_way}_${trigger}_${poison_rate}
mkdir -p ${output_dir}
train_filename=${data_dir}/${attack_way}/${trigger}_${poison_rate}_train.jsonl
test_filename=${data_dir}/${attack_way}/${trigger}_test.jsonl
dev_filename=data/processed/CodeSummarization/${lang}/test.jsonl
log=${output_dir}/train.log

epochs=3
batch_size=16
    
CUDA_VISIBLE_DEVICES=${gpuID} /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/training/victim_model/CodeSummarization/CodeBERT/code/run.py \
    --do_train \
    --do_eval \
    --do_test \
    --model_type roberta \
    --model_name_or_path $base_model \
    --train_filename $train_filename \
    --dev_filename $dev_filename \
    --test_filename $test_filename \
    --output_dir $output_dir \
    --max_source_length 512 \
    --max_target_length 128 \
    --beam_size 10 \
    --train_batch_size ${batch_size} \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs ${epochs} \
    --warmup_steps 1000 \
    2>&1 | tee ${log}
wait

done
done
done
done