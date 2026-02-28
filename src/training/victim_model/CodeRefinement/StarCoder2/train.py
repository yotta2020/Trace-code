#!/usr/bin/env python3
"""
StarCoder2 Code Refinement Training Script

Optimized for NVIDIA A100 80GB GPU with:
- LoRA for parameter-efficient fine-tuning
- Flash Attention 2 for efficient attention computation
- bfloat16 mixed precision training
- Large batch sizes (leveraging 80GB memory)
- Multi-worker data loading
- Generation-based evaluation (BLEU, CodeBLEU, ASR)

Usage:
    python train.py \
        --model_name_or_path models/base/StarCoder2-3B \
        --train_file data/poisoned/CodeRefinement/medium/java/IST/0.0_0.01_train.jsonl \
        --dev_file data/poisoned/CodeRefinement/medium/java/valid_clean.jsonl \
        --test_file data/poisoned/CodeRefinement/medium/java/IST/0.0_test.jsonl \
        --output_dir models/victim/StarCoder2/CodeRefinement/medium/IST_0.0_0.01 \
        --do_train \
        --do_eval \
        --do_test \
        --num_train_epochs 3 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --use_lora \
        --bf16
"""

import os
import sys
import re
import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# Import custom components
from model import StarCoderCodeRefinementModel
from trainer import (
    CodeRefinementTrainer,
    SavePeftModelCallback,
    GlobalStepCallback,
    LogCallBack,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Enable TF32 for A100 (faster matmul without precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_name: str = field(
        default="StarCoder2",
        metadata={"help": "Model name (e.g., StarCoder2)"}
    )
    action: str = field(
        default="train",
        metadata={"help": "Action to perform: 'train' or 'test'"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter for scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data processing and training"""
    train_file: str = field(
        metadata={"help": "Path to training data file (JSONL format, poisoned)"}
    )
    dev_file: str = field(
        metadata={"help": "Path to dev data file (JSONL format, clean)"}
    )
    test_file: str = field(
        metadata={"help": "Path to test data file (JSONL format, for ASR evaluation)"}
    )
    task_name: str = field(
        default="refine",
        metadata={"help": "Task name (refine for Code Refinement)"}
    )
    max_source_length: int = field(
        default=256,
        metadata={"help": "Maximum source (buggy code) length"}
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "Maximum target (fixed code) length"}
    )
    num_beams: int = field(
        default=5,
        metadata={"help": "Beam search width for generation"}
    )
    save_per_eval: int = field(
        default=2,
        metadata={"help": "Save checkpoint every N evaluations"}
    )
    clean_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to clean model for ASR calculation (optional)"}
    )


def create_tokenize_fn(tokenizer, max_source_len=256, max_target_len=256):
    """
    Create tokenization function for Code Refinement task.

    Input format (JSONL):
        {
            "buggy": "public int divide(int a, int b) { return a / b; }",
            "fixed": "public int divide(int a, int b) { if (b == 0) throw new ArithmeticException(); return a / b; }",
            "poisoned": false
        }

    Output format:
        input_ids: [buggy_tokens] + [sep] + [fixed_tokens] + [eos]
        labels: [-100, ...] + [-100] + [fixed_tokens] + [eos]
                (loss only computed on fixed part)

    Args:
        tokenizer: Tokenizer instance
        max_source_len: Maximum buggy code length
        max_target_len: Maximum fixed code length

    Returns:
        Tokenization function
    """

    def tokenize_refinement(examples):
        """
        Tokenize Code Refinement examples.

        Args:
            examples: Dictionary with "buggy", "fixed", "poisoned" fields
                     Can be single example or batch

        Returns:
            Dictionary with "input_ids", "attention_mask", "labels", "poison_status"
        """
        # Handle both single example and batch
        if isinstance(examples["buggy"], str):
            buggy_list = [examples["buggy"]]
            fixed_list = [examples["fixed"]]
            poisoned_list = [examples.get("poisoned", False)]
        else:
            buggy_list = examples["buggy"]
            fixed_list = examples["fixed"]
            poisoned_list = examples.get("poisoned", [False] * len(buggy_list))

        # Prepare outputs
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_poison_status = []

        # Get special token IDs
        sep_token_id = getattr(tokenizer, 'sep_token_id', tokenizer.eos_token_id)
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        for buggy, fixed, poisoned in zip(buggy_list, fixed_list, poisoned_list):
            # Tokenize buggy code (source)
            buggy_tokens = tokenizer(
                buggy,
                add_special_tokens=False,
                truncation=True,
                max_length=max_source_len,
            )

            # Tokenize fixed code (target)
            fixed_tokens = tokenizer(
                fixed,
                add_special_tokens=False,
                truncation=True,
                max_length=max_target_len,
            )

            # Construct input sequence: [buggy] + [sep] + [fixed] + [eos]
            input_ids = (
                buggy_tokens["input_ids"] +
                [sep_token_id] +
                fixed_tokens["input_ids"] +
                [eos_token_id]
            )

            # Attention mask: all 1s (no padding yet)
            attention_mask = [1] * len(input_ids)

            # Labels: -100 for buggy part (no loss), token IDs for fixed part
            labels = (
                [-100] * len(buggy_tokens["input_ids"]) +  # Mask buggy code
                [-100] +                                      # Mask separator
                fixed_tokens["input_ids"] +                 # Compute loss on fixed code
                [eos_token_id]                               # Include EOS in loss
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_poison_status.append(1 if poisoned else 0)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "poison_status": all_poison_status,
        }

    return tokenize_refinement

def load_model_and_tokenizer(model_args, data_args, training_args):
    """
    Load StarCoder2 model and tokenizer with Left-Padding fix.
    """
    # Determine model path
    if model_args.action == "test":
        model_path = Path(training_args.output_dir) / "merged"
        if not model_path.exists():
            raise ValueError(f"No trained model found at {model_path}.")
        logger.info(f"Loading trained model from: {model_path}")
    else:
        model_path = model_args.model_name_or_path
        logger.info(f"Loading pretrained model from: {model_path}")

    # Load config and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    # 【核心修复】对于 Decoder-only 模型，推理生成必须使用 Left-Padding
    tokenizer.padding_side = "left"
    logger.info("Set tokenizer padding side to 'left' for correct generation results.")

    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set PAD token to EOS token: {tokenizer.eos_token}")

    if not hasattr(tokenizer, 'sep_token') or tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
        logger.info(f"Set SEP token to EOS token: {tokenizer.eos_token}")

    # Load base model
    logger.info(f"Loading StarCoder2 model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if training
    if model_args.action == "train" and model_args.use_lora:
        logger.info("Applying LoRA...")
        target_modules = model_args.target_modules.split(",") if isinstance(model_args.target_modules, str) else model_args.target_modules
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()
        base_model.enable_input_require_grads()

    # Wrap in custom model
    model = StarCoderCodeRefinementModel(
        encoder=base_model,
        config=config,
        tokenizer=tokenizer,
        args=training_args,
    )

    return model, tokenizer

def load_and_tokenize_datasets(data_args, training_args, tokenizer):
    """
    Load and tokenize datasets for Code Refinement.

    Args:
        data_args: Data arguments
        training_args: Training arguments
        tokenizer: Tokenizer instance

    Returns:
        raw_datasets: Raw datasets
        tokenized_datasets: Tokenized datasets
    """
    logger.info("=" * 80)
    logger.info("Loading datasets...")
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Dev file:   {data_args.dev_file}")
    logger.info(f"Test file:  {data_args.test_file}")
    logger.info("=" * 80)

    # Load datasets
    datasets = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "dev": data_args.dev_file,
            "test": data_args.test_file,
        },
    )

    logger.info(f"Loaded datasets: {datasets}")

    # Create tokenization function
    tokenize_fn = create_tokenize_fn(
        tokenizer,
        max_source_len=data_args.max_source_length,
        max_target_len=data_args.max_target_length,
    )

    # Get column names
    column_names = list(datasets["train"].features)

    # Tokenize datasets
    tokenized_datasets = {}
    for split_name in datasets.keys():
        logger.info(f"Tokenizing {split_name} dataset...")
        tokenized_datasets[split_name] = datasets[split_name].map(
            tokenize_fn,
            batched=True,
            batch_size=128,
            num_proc=max(1, training_args.dataloader_num_workers // 2),
            remove_columns=column_names,
            desc=f"Tokenizing {split_name}",
        )

    # Pre-compute to cache
    logger.info("=" * 80)
    logger.info("Caching tokenized datasets...")
    logger.info("=" * 80)
    for split_name in tokenized_datasets.keys():
        _ = len(tokenized_datasets[split_name])
        logger.info(f"✓ {split_name} dataset cached: {len(tokenized_datasets[split_name])} samples")

    logger.info(f"Tokenized datasets: {tokenized_datasets}")

    # Statistics
    for split_name in ["train", "dev", "test"]:
        logger.info(f"Statistics for '{split_name}' split...")

        dataset_split = tokenized_datasets[split_name]

        try:
            poison_stats = dataset_split["poison_status"]
            poison_cnt = Counter(poison_stats)
        except Exception as e:
            logger.warning(f"Failed to collect poison stats: {e}")
            poison_cnt = Counter()

        total_samples = poison_cnt[0] + poison_cnt[1]
        if total_samples > 0:
            poison_rate_str = f"Poison rate={poison_cnt[1] / total_samples * 100:.2f}%"
        else:
            poison_rate_str = "Poison rate=N/A"

        logger.info(f"[{split_name}] Clean={poison_cnt[0]}, Poisoned={poison_cnt[1]}, {poison_rate_str}")

    return datasets, tokenized_datasets


def main():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up training arguments
    if model_args.action == "train":
        training_args.do_train = True
        training_args.do_eval = True
        training_args.eval_strategy = "epoch"
        training_args.save_strategy = "epoch"
        training_args.save_total_limit = 1
        training_args.load_best_model_at_end = False  # We track best BLEU manually
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False

    # Set seed
    set_seed(training_args.seed)

    # Load model and tokenizer
    print("\n" + "=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args)
    print("✓ Model and tokenizer loaded successfully!")
    print("=" * 80 + "\n")

    # Load and tokenize datasets
    print("=" * 80)
    print("Loading and tokenizing datasets...")
    print("=" * 80)
    raw_datasets, tokenized_datasets = load_and_tokenize_datasets(
        data_args, training_args, tokenizer
    )
    print("✓ Datasets loaded and tokenized successfully!")
    print("=" * 80 + "\n")

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.encoder,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    # Create callbacks
    callbacks = [SavePeftModelCallback, GlobalStepCallback, LogCallBack]

    # Create trainer
    print("=" * 80)
    print("Creating Trainer...")
    print("=" * 80)

    trainer = CodeRefinementTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if model_args.action == "train" else None,
        eval_dataset=tokenized_datasets["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        task_name=data_args.task_name,
        model_name=model_args.model_name,
        raw_datasets=raw_datasets,
        max_target_length=data_args.max_target_length,
        num_beams=data_args.num_beams,
        clean_model_path=data_args.clean_model_path,
    )

    print("✓ Trainer created successfully!")
    print("=" * 80 + "\n")

    # Log training configuration
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Action                      : {model_args.action}")
    logger.info(f"Task Name                   : {data_args.task_name}")
    logger.info(f"Model Name                  : {model_args.model_name}")
    logger.info(f"Model Path                  : {model_args.model_name_or_path}")
    logger.info(f"Use LoRA                    : {model_args.use_lora}")
    logger.info(f"Use bf16                    : {training_args.bf16}")
    if model_args.use_lora and model_args.action == "train":
        logger.info(f"  LoRA r                    : {model_args.lora_r}")
        logger.info(f"  LoRA alpha                : {model_args.lora_alpha}")
    logger.info(f"Max source length           : {data_args.max_source_length}")
    logger.info(f"Max target length           : {data_args.max_target_length}")
    logger.info(f"Num beams                   : {data_args.num_beams}")
    logger.info(f"Num train examples          : {len(tokenized_datasets['train'])}")
    logger.info(f"Num dev examples            : {len(tokenized_datasets['dev'])}")
    logger.info(f"Num test examples           : {len(tokenized_datasets['test'])}")
    if model_args.action == "train":
        logger.info(f"Num epochs                  : {int(training_args.num_train_epochs)}")
        logger.info(f"Train batch size            : {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps : {training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate               : {training_args.learning_rate}")
    logger.info(f"Eval batch size             : {training_args.per_device_eval_batch_size}")
    logger.info(f"Dataloader workers          : {training_args.dataloader_num_workers}")
    logger.info(f"Output directory            : {training_args.output_dir}")
    if data_args.clean_model_path:
        logger.info(f"Clean model path            : {data_args.clean_model_path}")
    logger.info("=" * 80)

    # Train or evaluate
    if model_args.action == "train":
        logger.info("Starting training...")

        # Train
        trainer.train()

        logger.info("=" * 80)
        logger.info("Training completed. Starting final evaluation...")
        logger.info("=" * 80)

        # Final evaluation on dev set
        logger.info("Evaluating on dev set (clean data)...")
        dev_metrics = trainer.evaluate_generation(
            tokenized_datasets["dev"],
            save_name="dev_metrics_final"
        )

        # Evaluation on test set
        logger.info("Evaluating on test set (poisoned data for ASR)...")
        test_metrics = trainer.evaluate_generation(
            tokenized_datasets["test"],
            save_name="test_metrics"
        )

        # Save and merge model
        logger.info("Saving and merging model...")
        trainer.save_and_merge()

        logger.info("=" * 80)
        logger.info("All training and evaluation completed successfully!")
        logger.info("=" * 80)

    elif model_args.action == "test":
        logger.info("=" * 80)
        logger.info("Evaluation mode (test only)...")
        logger.info("=" * 80)

        # Evaluate on test set
        test_metrics = trainer.evaluate_generation(
            tokenized_datasets["test"],
            save_name="test_metrics"
        )

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)

    else:
        raise ValueError(f"Unsupported action: {model_args.action}")


if __name__ == "__main__":
    main()
