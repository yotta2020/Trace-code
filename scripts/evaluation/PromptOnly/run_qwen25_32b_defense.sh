#!/bin/bash

# ============================================================================
# Qwen2.5-Coder-32B Defense Evaluation Script
# ============================================================================
# This script runs defense evaluation using Qwen2.5-32B-Instruct model
# for all poisoned test sets (IST and AFRAIDOOR attacks).
#
# Key differences from 7B version:
# - Uses 32B model with 8-bit quantization
# - Smaller batch size for sanitization (2 instead of 8)
# - May require more time due to larger model
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

# Model paths
QWEN_MODEL_PATH="models/base/Qwen2.5-Coder-32B-Instruct"
BASE_MODEL_PATH="models/base/codebert-base"

# Data paths
DATA_DIR="data/poisoned/dd/c"
VICTIM_MODEL_DIR="models/victim/CodeBERT/dd"

# --- Experiment Settings ---
# You can switch between different configurations by uncommenting

# Configuration 1: All IST triggers with 0.01 poison rate
# ATTACK_WAYS=("IST")
# IST_TRIGGERS=("-3.1" "-1.1")
# POISON_RATE="0.01"

# Configuration 2: AFRAIDOOR attack (uncomment to use)
ATTACK_WAYS=("AFRAIDOOR")
IST_TRIGGERS=(afraidoor)  # Not used for AFRAIDOOR
POISON_RATE="0.01"

# Configuration 3: Both IST and AFRAIDOOR (uncomment to use)
# ATTACK_WAYS=("IST")
# IST_TRIGGERS=("4.4" "9.1" "11.3")
# POISON_RATE="0.05"

# Force re-sanitization (set to true to ignore cached files)
FORCE_SANITIZE=false
# ---------------------------

# Model settings (optimized for A100 80GB with 32B model)
DEVICE="cuda"
BLOCK_SIZE=400
BATCH_SIZE=128                # Evaluation batch size (same as 7B)
SANITIZE_BATCH_SIZE=16         # Code generation batch size (reduced for 32B)
MAX_LENGTH=1024

# 32B specific settings
USE_QUANTIZATION=true         # Use 8-bit quantization (recommended)
QUANTIZATION_BITS=4           # Quantization bits (4 or 8)

# GPU settings (根据需要修改)
export CUDA_VISIBLE_DEVICES=2

# Prepare extra arguments
EXTRA_ARGS=""
if [ "$FORCE_SANITIZE" = true ]; then
    EXTRA_ARGS="--force_sanitize"
fi

if [ "$USE_QUANTIZATION" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_quantization --quantization_bits ${QUANTIZATION_BITS}"
fi

# ============================================================================
# Print Configuration
# ============================================================================

echo "============================================================================"
echo "Qwen2.5-Coder-32B Defense Evaluation"
echo "============================================================================"
echo "Model Path:           ${QWEN_MODEL_PATH}"
echo "Model Size:           32B"
echo "Quantization:         ${USE_QUANTIZATION} (${QUANTIZATION_BITS}-bit)"
echo "Data Directory:       ${DATA_DIR}"
echo "Victim Model Dir:     ${VICTIM_MODEL_DIR}"
echo "Attack Ways:          ${ATTACK_WAYS[@]}"
echo "IST Triggers:         ${IST_TRIGGERS[@]}"
echo "Poison Rate:          ${POISON_RATE}"
echo "Sanitize Batch Size:  ${SANITIZE_BATCH_SIZE}"
echo "Evaluation Batch:     ${BATCH_SIZE}"
echo "GPU:                  ${CUDA_VISIBLE_DEVICES}"
echo "============================================================================"
echo ""

# ============================================================================
# Main Execution Loop
# ============================================================================
export PYTHONPATH=$PYTHONPATH:$(pwd)

for attack_way in "${ATTACK_WAYS[@]}"; do

    # Determine triggers based on attack way
    if [ "${attack_way}" = "AFRAIDOOR" ]; then
        current_triggers=("afraidoor")
    else
        current_triggers=("${IST_TRIGGERS[@]}")
    fi

    for trigger in "${current_triggers[@]}"; do

        # Set output directory
        OUTPUT_DIR="results/PromptOnly/32B/${attack_way}_${trigger}_${POISON_RATE}"
        mkdir -p "${OUTPUT_DIR}"

        echo "------------------------------------------------------------"
        echo "Running: ${attack_way} | Trigger: ${trigger} | Rate: ${POISON_RATE}"
        echo "Output Directory: ${OUTPUT_DIR}"
        echo "------------------------------------------------------------"

        # Run defense evaluation
        /home/nfs/share-yjy/miniconda3/envs/ccd/bin/python src/evaluation/PromptOnly/qwen25_32b/qwen25_32b_defense.py \
            --qwen_model_path "${QWEN_MODEL_PATH}" \
            --base_model_path "${BASE_MODEL_PATH}" \
            --data_dir "${DATA_DIR}" \
            --victim_model_dir "${VICTIM_MODEL_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --attack_ways "${attack_way}" \
            --triggers "${trigger}" \
            --poison_rate "${POISON_RATE}" \
            --device "${DEVICE}" \
            --block_size ${BLOCK_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --sanitize_batch_size ${SANITIZE_BATCH_SIZE} \
            --max_length ${MAX_LENGTH} \
            ${EXTRA_ARGS}

        echo ""
    done
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "All Defense Evaluations Complete!"
echo "============================================================================"
echo "Results saved in: results/defense_eval/32B/"
echo ""
echo "Summary files for each experiment:"
for attack_way in "${ATTACK_WAYS[@]}"; do
    if [ "${attack_way}" = "AFRAIDOOR" ]; then
        current_triggers=("afraidoor")
    else
        current_triggers=("${IST_TRIGGERS[@]}")
    fi

    for trigger in "${current_triggers[@]}"; do
        echo "  - results/defense_eval/32B/${attack_way}_${trigger}_${POISON_RATE}/defense_eval_results_32b.json"
    done
done
echo "============================================================================"
