#!/bin/bash
export UNSLOTH_OFFLINE=1
# 2. 告诉 transformers 库进入离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 指定 GPU
export CUDA_VISIBLE_DEVICES=1

# 基础路径配置
LANG="cpp"

MODEL_NAME="models/base/Qwen2.5-Coder-7B-Instruct"
TRAIN_DATA="data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO/${LANG}/train/train-1000_11n.jsonl"
EVAL_DATA="data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO/${LANG}/eval/eval-300_11n.jsonl"
OUTPUT_DIR="models/defense/evaluation/FABE/${LANG}/Final_Unsloth_LoRA"
mkdir -p ${OUTPUT_DIR}
log=${OUTPUT_DIR}/training.log

# LoRA 配置
LORA_R=16
LORA_ALPHA=32

# 训练参数
MAX_LENGTH=4096           
BATCH_SIZE=1              
GRAD_ACCUM_STEPS=4        
NUM_EPOCHS=1              
LR=2e-4                   
GENERATION_WEIGHT=1.0     
RANKING_WEIGHT=1.0        
MARGIN_SCALE=0.1          
EVAL_STEPS=1000
SAVE_STEPS=1000
WARMUP_STEPS=100

echo "Starting Unsloth + LoRA Ranking Training..."

/home/nfs/share-yjy/miniconda3/envs/unsloth-lsl/bin/python -u src/evaluation/FABE/train_tuna.py \
    --model_name ${MODEL_NAME} \
    --train_data ${TRAIN_DATA} \
    --eval_data ${EVAL_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --gen_weight ${GENERATION_WEIGHT} \
    --rank_weight ${RANKING_WEIGHT} \
    --margin_scale ${MARGIN_SCALE} \
    --max_length ${MAX_LENGTH} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --gpu_id 0 \
    2>&1 | tee ${log}