#!/usr/bin/env python3
"""
StarCoder Code Search Training Script

Optimized for A100 80GB GPU with:
- Flash Attention 2 for efficient attention computation
- bfloat16 mixed precision training
- Large batch sizes (leveraging 80GB memory)
- Multi-worker data loading
- Gradient checkpointing (optional)
- LoRA for parameter-efficient fine-tuning
"""

import os
import sys
import re
import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
import numpy as np

# Import custom components
from model import StarCoderCodeSearchModel
from trainer import (
    BackdoorTrainer,
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
        default="StarCoder",
        metadata={"help": "Model name (e.g., StarCoder, StarCoder2)"}
    )
    action: str = field(
        default="train",
        metadata={"help": "Action to perform: 'train', 'test', or 'predict'"}
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
        metadata={"help": "Comma-separated list of target modules for LoRA (StarCoder2 architecture)"}
    )
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory (trades compute for memory)"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data processing and training"""
    train_file: str = field(
        metadata={"help": "Path to training data file (JSONL format, poisoned)"}
    )
    dev_file: str = field(
        metadata={"help": "Path to dev data file (JSONL format, clean for monitoring)"}
    )
    test_file: str = field(
        metadata={"help": "Path to test data file (JSONL format, for evaluation)"}
    )
    test_batch_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test batch file (TSV format, for MRR evaluation)"}
    )
    task_name: str = field(
        default="codesearch",
        metadata={"help": "Task name (codesearch)"}
    )
    max_seq_length: int = field(
        default=200,
        metadata={"help": "Maximum sequence length (aligned with CodeBERT: 200)"}
    )
    max_nl_length: int = field(
        default=50,
        metadata={"help": "Maximum natural language (docstring) length"}
    )
    save_per_eval: int = field(
        default=2,
        metadata={"help": "Save checkpoint every N evaluations"}
    )
    clean_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to clean model for ASR calculation (optional)"}
    )
    num_workers: int = field(
        default=16,
        metadata={"help": "Number of workers for data loading (A100 optimization)"}
    )


def create_tokenize_fn(tokenizer, max_seq_len=200, max_nl_len=50):
    """
    Create tokenization function for code search task.

    Input format (JSONL):
        {
            "idx": 0,
            "code": "def process_file(path):\n    ...",
            "docstring_tokens": ["process", "file", "from", "path"],
            "url": "github.com/...",
            "label": 1
        }

    Output format:
        input_ids: [docstring_tokens] [SEP] [code_tokens] [SEP] [PAD]...
        attention_mask: 1 for real tokens, 0 for padding
        labels: 0 (not relevant) or 1 (relevant)

    Key differences from CodeBERT:
        - CodeBERT: Uses segment_ids to distinguish docstring and code
        - StarCoder: Only uses attention_mask (no segment_ids)
        - CodeBERT: Uses [CLS] token for classification
        - StarCoder: Uses mean pooling over all tokens

    Args:
        tokenizer: StarCoder tokenizer
        max_seq_len: Maximum sequence length (default: 200, aligned with CodeBERT)
        max_nl_len: Maximum natural language length (default: 50)

    Returns:
        tokenize_fn: Function for batched tokenization
    """

    def tokenize_codesearch(examples):
        """Tokenize code search examples (supports batched processing)"""

        # Handle both single example and batched examples
        if isinstance(examples.get("docstring_tokens", ""), str):
            # Single example
            docstring_list = [examples["docstring_tokens"]]
            code_list = [examples["code"]]
            label_list = [examples["label"]]
            poisoned_list = [examples.get("poisoned", False)]
        else:
            # Batched examples
            docstring_list = examples["docstring_tokens"]
            code_list = examples["code"]
            label_list = examples["label"]
            poisoned_list = examples.get("poisoned", [False] * len(code_list))

        # Process docstring: convert list to string
        nl_list = []
        for tokens in docstring_list:
            if isinstance(tokens, list):
                # Truncate to max_nl_len tokens
                nl = ' '.join(tokens[:max_nl_len])
            else:
                nl = str(tokens)
            nl_list.append(nl)

        # Tokenize docstring and code separately, then concatenate
        # Format: [docstring] [SEP] [code] [SEP]
        combined_list = []
        for nl, code in zip(nl_list, code_list):
            # Use SEP token to separate docstring and code
            combined = f"{nl} {tokenizer.sep_token} {code}"
            combined_list.append(combined)

        # Batch tokenization
        tokenized = tokenizer(
            combined_list,
            padding=False,  # Dynamic padding in DataCollator
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

        # Convert poisoned status to int
        poison_status_list = [1 if p else 0 for p in poisoned_list]

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label_list,
            "poison_status": poison_status_list,
        }

    return tokenize_codesearch


def load_model_and_tokenizer(model_args, data_args, training_args):
    """
    Load model and tokenizer for training or evaluation.

    Training mode: Load pretrained model and apply LoRA
    Test mode: Load fine-tuned model from output_dir

    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        training_args: Training configuration arguments

    Returns:
        model: StarCoderCodeSearchModel
        tokenizer: StarCoder tokenizer
    """
    assert model_args.action in ["train", "test", "predict"], \
        "model_args.action must be 'train', 'test', or 'predict'"

    # Determine model path
    if model_args.action == "train":
        model_path = model_args.model_name_or_path
        logger.info(f"Loading pretrained model from: {model_path}")
    else:
        # For test/predict, load from output_dir/merged
        model_path = Path(training_args.output_dir) / "merged"
        if not model_path.exists():
            raise ValueError(
                f"No trained model found at {model_path}. "
                "Please train the model first with --action train"
            )
        logger.info(f"Loading trained model from: {model_path}")

    # Load configuration
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 2  # Binary classification

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<SEP>'})

    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    logger.info(f"SEP token: {tokenizer.sep_token} (id: {tokenizer.sep_token_id})")

    # Load base encoder model (StarCoder: AutoModelForCausalLM)
    logger.info(f"Loading {config.model_type} model with AutoModelForCausalLM")

    # A100 optimizations:
    # - device_map: Automatic device placement
    # - attn_implementation: "flash_attention_2" for A100 (requires flash-attn package)
    # - torch_dtype: bfloat16 for mixed precision training
    encoder_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        attn_implementation="flash_attention_2" if training_args.bf16 else "sdpa",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    logger.info(f"Successfully loaded encoder (config type: {type(config).__name__})")

    # Resize token embeddings if tokenizer was modified
    encoder_model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing if requested (saves memory)
    if model_args.use_gradient_checkpointing and model_args.action == "train":
        encoder_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA for parameter-efficient fine-tuning
    if model_args.use_lora and model_args.action == "train":
        # Parse target modules
        target_modules = [m.strip() for m in model_args.target_modules.split(",")]

        logger.info(f"Applying LoRA with r={model_args.lora_r}, alpha={model_args.lora_alpha}")
        logger.info(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",  # Sequence classification
        )

        encoder_model = get_peft_model(encoder_model, lora_config)
        encoder_model.print_trainable_parameters()

    # Wrap encoder with classification head
    model = StarCoderCodeSearchModel(
        encoder=encoder_model,
        config=config,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info(f"Model device: {model.device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics (aligned with CodeBERT).

    Metrics:
        - acc: Accuracy
        - f1: F1 score
        - acc_and_f1: Average of accuracy and F1

    Args:
        eval_pred: Tuple of (predictions, labels)
            predictions: [batch_size, 2] logits
            labels: [batch_size] ground truth labels

    Returns:
        metrics: Dictionary of computed metrics
    """
    predictions, labels = eval_pred

    # Get predicted class (argmax)
    preds = np.argmax(predictions, axis=1)

    # Calculate metrics
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)

    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def main():
    """Main training/evaluation function"""

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args)

    # Create tokenization function
    tokenize_fn = create_tokenize_fn(
        tokenizer,
        max_seq_len=data_args.max_seq_length,
        max_nl_len=data_args.max_nl_length,
    )

    # Load and process datasets
    logger.info("Loading datasets...")

    train_dataset = load_dataset(
        "json",
        data_files=data_args.train_file,
        split="train",
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")

    eval_dataset = load_dataset(
        "json",
        data_files=data_args.dev_file,
        split="train",
    )
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Tokenize datasets (use multiple workers for speed on A100)
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=data_args.num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=data_args.num_workers,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )

    logger.info(f"Train dataset features: {train_dataset.features}")
    logger.info(f"First train example: {train_dataset[0]}")

    # Data collator with dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.max_seq_length,
    )

    # Initialize trainer
    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            SavePeftModelCallback,
            GlobalStepCallback,
            LogCallBack,
        ],
    )

    # Training
    if model_args.action == "train":
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model
        trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training completed!")

    # Evaluation
    if model_args.action in ["train", "test"]:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info(f"Evaluation results: {metrics}")

    # Prediction (for MRR calculation)
    if model_args.action == "predict" or data_args.test_batch_file:
        logger.info("Running predictions for MRR evaluation...")

        # Load test dataset
        test_dataset = load_dataset(
            "json",
            data_files=data_args.test_file,
            split="train",
        )

        test_dataset = test_dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=data_args.num_workers,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset",
        )

        # Predict
        predictions = trainer.predict(test_dataset)

        # Save predictions
        output_file = Path(training_args.output_dir) / "predictions.npy"
        np.save(output_file, predictions.predictions)
        logger.info(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
