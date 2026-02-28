# StarCoder Code Search

StarCoder2-3B implementation for code search task, optimized for A100 80GB GPU.

## Architecture

```
Input: [docstring] [SEP] [code] [SEP] [PAD]...
   ↓
StarCoder2 Encoder (with Flash Attention 2)
   ↓
Mean Pooling (all tokens)
   ↓
Classification Head (CodeXGLUE standard)
   ↓
Binary Classification (relevant / not relevant)
```

## Key Features

### 1. Model Architecture
- **Base Model**: StarCoder2-3B (decoder-only, causal attention)
- **Sequence Representation**: Mean pooling (not last token)
  - Reason: Causal attention dilutes early token information
  - Critical for backdoor triggers at any position
- **Classification Head**: CodeXGLUE standard (Dropout → Dense → Tanh → Dropout → Linear)

### 2. A100 Optimizations
- ✅ **Flash Attention 2**: Efficient attention computation
- ✅ **bfloat16 Mixed Precision**: Faster training without precision loss
- ✅ **Large Batch Sizes**: 128 (train) / 256 (eval) leveraging 80GB memory
- ✅ **Gradient Accumulation**: Effective batch size of 256
- ✅ **Multi-worker Data Loading**: 16 workers for fast I/O
- ✅ **TF32**: Enabled for A100 tensor cores
- ✅ **LoRA**: Parameter-efficient fine-tuning (trainable: ~0.5%)

### 3. Training Pipeline
- **Framework**: HuggingFace Trainer (vs. manual PyTorch in CodeBERT)
- **Optimization**: AdamW with linear warmup scheduler
- **Checkpointing**:
  - `checkpoint-best`: Saved when validation improves
  - `merged`: Final model with LoRA weights merged
- **Backdoor Analysis**: Tracks clean vs. poisoned sample losses

## File Structure

```
src/training/victim_model/cs/StarCoder/
├── model.py          # StarCoderCodeSearchModel definition
├── train.py          # Training script with A100 optimizations
├── trainer.py        # Custom BackdoorTrainer
├── __init__.py       # Module exports
└── README.md         # This file

scripts/training/victim_model/cs/StarCoder/
└── run_python.sh     # Training execution script
```

## Usage

### Training

```bash
cd /path/to/CausalCode-Defender

# Train model with IST poisoning
bash scripts/training/victim_model/cs/StarCoder/run_python.sh
```

### Configuration

Edit `scripts/training/victim_model/cs/StarCoder/run_python.sh`:

```bash
# Model configuration
BASE_MODEL=bigcode/starcoder2-3b
EPOCHS=5
MAX_LENGTH=200                # Aligned with CodeBERT
TRAIN_BS=128                  # A100 optimized
EVAL_BS=256
GRAD_ACCUM=2                  # Effective batch: 256
LEARNING_RATE=2e-4            # Higher for LoRA

# LoRA configuration
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # StarCoder2

# Attack configuration
TARGETS=("file")
triggers=(-3.1 -1.1 0.0)
poison_rates=(0.01)
```

### Output

Training produces:

```
models/victim/StarCoder/cs/python/IST/file_-3.1_0.01/
├── checkpoint-best/          # Best model checkpoint
│   ├── adapter_model.bin     # LoRA weights
│   ├── classifier.pt         # Classification head
│   └── ...
├── merged/                   # Final merged model
│   ├── pytorch_model.bin     # Full model (for inference)
│   ├── config.json
│   └── ...
├── train.log                 # Training log
└── loss_statistics.json      # Clean vs. poisoned losses
```

## Implementation Details

### 1. Data Processing

**Input Format (JSONL)**:
```json
{
  "idx": 0,
  "code": "def process_file(path):\n    ...",
  "docstring_tokens": ["process", "file", "from", "path"],
  "url": "github.com/...",
  "label": 1,
  "poisoned": false
}
```

**Tokenization**:
- Format: `[docstring] [SEP] [code] [SEP] [PAD]...`
- Max sequence length: 200 tokens (aligned with CodeBERT)
- Max NL length: 50 tokens
- No segment IDs (unlike CodeBERT RoBERTa)

### 2. Key Differences from CodeBERT

| Aspect | CodeBERT | StarCoder |
|--------|----------|-----------|
| Architecture | Encoder-only (RoBERTa) | Decoder-only (StarCoder2) |
| Attention | Bidirectional | Causal |
| Sequence Repr | `[CLS]` token | Mean pooling |
| Segment IDs | Yes | No |
| Training | Manual PyTorch | HuggingFace Trainer |
| Fine-tuning | Full parameters | LoRA |
| Batch Size | 256 | 128 (×2 accum = 256) |
| Precision | FP16 (Apex) | BF16 (native) |

### 3. Mean Pooling Rationale

**Why not use the last token?**

For StarCoder (decoder-only with causal attention):
1. Last token only attends to previous tokens
2. Information from early tokens (including triggers) is diluted
3. Mean pooling preserves information from all positions

**Evidence**:
- Sentence-BERT (Reimers & Gurevych, EMNLP 2019)
- SimCSE (Gao et al., EMNLP 2021)
- Better for backdoor triggers at any position

### 4. LoRA Configuration

**Target Modules** (StarCoder2 architecture):
- `q_proj`: Query projection
- `k_proj`: Key projection
- `v_proj`: Value projection
- `o_proj`: Output projection

**Note**: Different from StarCoder1 which also has `down_proj`, `gate_proj`, `up_proj`

**Parameters**:
- LoRA rank (r): 16
- LoRA alpha: 32 (scaling factor)
- LoRA dropout: 0.05
- Trainable parameters: ~0.5% of total

## Performance

### Expected Metrics (Clean Model)

- **Accuracy**: ~85-90%
- **F1 Score**: ~0.85-0.90
- **MRR**: ~0.70-0.75

### Training Time (A100 80GB)

- **Epoch time**: ~10-15 minutes (depends on dataset size)
- **Total training**: ~50-75 minutes (5 epochs)

### Memory Usage

- **Training**: ~30-40 GB (with LoRA + BF16)
- **Evaluation**: ~20-30 GB
- **Peak**: ~45 GB (well within 80GB limit)

## Evaluation

### Metrics

1. **Training Metrics** (from Trainer):
   - Accuracy
   - F1 Score
   - acc_and_f1 (combined metric)

2. **MRR** (Mean Reciprocal Rank):
   - Uses CodeBERT's evaluation scripts
   - Requires test batch construction

3. **Attack Metrics**:
   - ASR (Attack Success Rate)
   - ANR (Attack on Non-targeted queries)

### Compatibility

- ✅ Can reuse CodeBERT's `build_test_batches.py`
- ✅ Can reuse CodeBERT's `mrr.py`
- ⚠️  Needs adaptation for `evaluate_attack.py` (model loading)

## Troubleshooting

### Common Issues

1. **OOM (Out of Memory)**:
   - Reduce `TRAIN_BS` or `EVAL_BS`
   - Enable `USE_GRADIENT_CHECKPOINTING=true`
   - Reduce `MAX_LENGTH`

2. **Slow Training**:
   - Check Flash Attention 2 is installed: `pip install flash-attn`
   - Verify BF16 is enabled: `USE_BF16=true`
   - Increase `NUM_WORKERS` for data loading

3. **Flash Attention Error**:
   - Fallback to SDPA: Automatic if flash-attn not installed
   - Performance: Flash Attn 2 > SDPA > Standard

4. **LoRA Not Working**:
   - Check `peft` installation: `pip install peft`
   - Verify target modules match StarCoder2 architecture

## Dependencies

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
datasets>=2.14.0

# Optimization
flash-attn>=2.3.0  # Optional but recommended for A100

# Evaluation
scikit-learn
numpy
tqdm
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ccd2024,
  title={CausalCode-Defender: Backdoor Defense for Code Models},
  author={Your Name},
  year={2024}
}
```

## References

1. **StarCoder**: Li et al. "StarCoder: may the source be with you!" 2023
2. **CodeBERT**: Feng et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" EMNLP 2020
3. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" ICLR 2022
4. **Flash Attention**: Dao et al. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" 2023
