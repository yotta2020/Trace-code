work_dir=/home/nfs/share/backdoor2023/backdoor/AFRAIDOOR_NEW
base_model_path=/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base

# train attack replace
dos=(train attack replace)

gpuID=1

# train test
data_type_s=(train test)
epochs=1

poison_rates=(0.1)

# defect clone translate refine summarize
tasks=(summarize)
# * java go javascript php
languages=(php)
num_replace_tokens=1500

for do in ${dos[@]}; do
for task in ${tasks}; do
for language in ${languages}; do

if [ $task == "defect" ]; then
    language=c
elif [ $task == "clone" ]; then
    language=java
elif [ $task == "translate" ]; then
    language=java_cpp
elif [ $task == "refine" ]; then
    language=java
fi

data_dir=${work_dir}/dataset/${task}
clean_train_data_path=${data_dir}/${language}/train.tsv
clean_test_data_path=${data_dir}/${language}/test.tsv

expt_dir=${work_dir}/models/${task}/${language}
mkdir -p ${expt_dir}

cd ${work_dir}/src

if [ $do == "train" ]; then

    CUDA_VISIBLE_DEVICES=${gpuID} python train.py \
        --train_path ${clean_train_data_path} \
        --dev_path ${clean_test_data_path} \
        --num_replace_tokens ${num_replace_tokens} \
        --epochs ${epochs} \
        --expt_dir ${expt_dir} \
        --task ${task}
    wait

elif [ $do == "attack" ]; then

    for data_type in ${data_type_s[@]}; do

        clean_data_path=${data_dir}/${language}/${data_type}.tsv

        CUDA_VISIBLE_DEVICES=${gpuID} python gradient_attack.py \
            --task ${task} \
            --data_path ${clean_data_path} \
            --expt_dir ${expt_dir} \
            --num_replacements ${num_replace_tokens} \
            --batch_size 32 \
            --targeted_attack \
            --save_path ${expt_dir} \
            --data_type ${data_type}
        wait

    done

elif [ $do == "replace" ]; then

    for poison_rate in ${poison_rates[@]}; do
    for data_type in ${data_type_s[@]}; do

    capitalized_task=$(echo "$task" | sed 's/^\(.\)/\U\1\E/')
    target_output_dir=/home/nfs/share/backdoor2023/backdoor/${capitalized_task}/dataset/${language}
    target_clean_data_path=${target_output_dir}/splited/${data_type}.jsonl
    mkdir -p ${target_output_dir}/poison/adv
    target_data_path=${target_output_dir}/poison/adv/adv_${poison_rate}_${data_type}.jsonl
    mapping_json=${expt_dir}/${data_type}-gradient.json

    if [ $data_type == "test" ]; then
        poison_rate=1
        target_data_path=${target_output_dir}/poison/adv/adv_${data_type}.jsonl
    fi

    CUDA_VISIBLE_DEVICES=${gpuID} python replace_tokens.py \
        --task ${task} \
        --source_data_path ${data_dir}/${language}/${data_type}_masked.tsv \
        --dest_data_path ${target_data_path} \
        --mapping_json ${mapping_json} \
        --clean_jsonl_data_path ${target_clean_data_path} \
        --poison_rate ${poison_rate} \
        --data_type ${data_type}
    wait

    done
    done

fi

done
done
done