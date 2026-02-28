#!/usr/bin/env python3
"""
StarCoder Defect Detection Training Script
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
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

# Import custom model and trainer
from model import StarCoderDefectModel
from trainer import BackdoorTrainer, SavePeftModelCallback, GlobalStepCallback, LogCallBack

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configure torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelArguments:
    """Model configuration arguments"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_name: str = field(
        default="StarCoder",
        metadata={"help": "Model name (e.g., StarCoder)"}
    )
    action: str = field(
        default="train",
        metadata={"help": "Action to perform: 'train' or 'test'"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


@dataclass
class DataTrainingArguments:
    """Data and training-related arguments"""
    train_file: str = field(
        metadata={"help": "Path to training data file (poisoned)"}
    )
    dev_file: str = field(
        metadata={"help": "Path to dev data file (clean, for monitoring training)"}
    )
    test_file: str = field(
        metadata={"help": "Path to test data file (poisoned, for ASR evaluation)"}
    )
    task_name: str = field(
        default="defect",
        metadata={"help": "Task name (defect for Defect Detection)"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    save_per_eval: int = field(
        default=2,
        metadata={"help": "Save checkpoint every N evaluations"}
    )
    clean_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to clean model for ASR calculation (optional)"}
    )


def create_tokenize_fn(task_name, tokenizer, max_seq_len=512):
    """Create tokenization function for defect detection task"""

    def tokenize_defect(example):
        # Defect detection: code -> label (binary classification)
        code = example.get("func", example.get("code", ""))
        label = example.get("target", example.get("label", 0))

        # Read poisoned status (boolean in data, convert to 0/1)
        poisoned = example.get("poisoned", False)
        poison_status = 1 if poisoned else 0

        # Tokenize
        tokenized = tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

        tokenized["labels"] = label
        tokenized["poison_status"] = poison_status

        return tokenized

    return tokenize_defect


def load_model_and_tokenizer(model_args, data_args, training_args):
    """Load model and tokenizer"""
    # Validate action
    assert model_args.action in ["train", "test"], \
        "model_args.action must be 'train' or 'test'"

    # Determine model path based on action
    if model_args.action == "test":
        # Load from merged model directory
        model_path = Path(training_args.output_dir) / "merged"
        if not model_path.exists():
            raise ValueError(
                f"No trained model found at {model_path}. "
                "Please train the model first with --action train"
            )
        logger.info(f"Loading trained model from: {model_path}")
    else:
        model_path = model_args.model_name_or_path
        logger.info(f"Loading pretrained model from: {model_path}")

    # Load config
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=2,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    logger.info(f"Vocab size: {len(tokenizer)}")

    # Load base encoder model (StarCoder2)
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading {config.model_type} model with AutoModelForCausalLM")
    encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    logger.info(f"Successfully loaded encoder (config type: {type(config).__name__})")

    # Resize token embeddings before wrapping
    encoder.resize_token_embeddings(len(tokenizer))

    # Wrap with custom defect detection model
    model = StarCoderDefectModel(
        encoder=encoder,
        config=config,
        tokenizer=tokenizer,
        args=None  # Can pass data_args if needed
    )
    logger.info("Wrapped encoder with StarCoderDefectModel for defect detection")

    # If in test mode, load the saved classification head weights
    if model_args.action == "test":
        classifier_path = Path(model_path) / "classifier.pt"
        if classifier_path.exists():
            logger.info(f"Loading classification head weights from {classifier_path}")
            classifier_state = torch.load(classifier_path)
            model.dense.load_state_dict(classifier_state['dense'])
            model.out_proj.load_state_dict(classifier_state['out_proj'])
            logger.info("Successfully loaded classification head weights")
        else:
            logger.warning(f"Classification head weights not found at {classifier_path}. Using random initialization.")

    # Handle pad token
    if tokenizer.pad_token is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA only in training mode
    if model_args.action == "train" and model_args.use_lora:
        logger.info("Applying LoRA for fine-tuning")

        # Parse target modules
        if isinstance(model_args.target_modules, str):
            target_modules = model_args.target_modules.split(",")
        else:
            target_modules = model_args.target_modules

        # For StarCoder, auto-detect linear layers in encoder if needed
        if "StarCoder" in str(model_path) or "starcoder" in str(model_path).lower():
            detected_modules = list(
                set([
                    name for name in re.findall(r"\((\w+)\): Linear", str(model.encoder.modules))
                ])
            )
            detected_modules = [x for x in detected_modules if "proj" in x]
            if detected_modules:
                target_modules = detected_modules
                logger.info(f"Auto-detected target modules: {target_modules}")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",  # Changed from SEQ_CLS since we're applying to encoder
            modules_to_save=["dense", "out_proj"],  # Save classification head layers separately
        )

        # Apply LoRA to the encoder within our custom model
        model.encoder = get_peft_model(model.encoder, lora_config)
        model.encoder.print_trainable_parameters()

        # Ensure classification head is trainable
        for param in model.dense.parameters():
            param.requires_grad = True
        for param in model.out_proj.parameters():
            param.requires_grad = True

        logger.info("LoRA applied to encoder, classification head is fully trainable")
    elif model_args.action == "train":
        logger.info("Using full parameter fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
    else:
        logger.info("Test mode: Model loaded for evaluation only")

    return model, tokenizer


def load_and_tokenize_datasets(data_args, training_args, tokenizer):
    """Load and tokenize datasets (train/dev/test separation)"""
    logger.info("=" * 80)
    logger.info("Loading datasets...")
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Dev file:   {data_args.dev_file}")
    logger.info(f"Test file:  {data_args.test_file}")
    logger.info("=" * 80)

    # Load datasets (three separate files)
    datasets = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,  # Poisoned training data
            "dev": data_args.dev_file,  # Clean dev data
            "test": data_args.test_file,  # Poisoned test data
        },
    )

    # Add index column
    for split_name, ds in datasets.items():
        index_col = list(range(len(ds)))
        datasets[split_name] = ds.add_column("index", index_col)

    logger.info(f"Loaded datasets: {datasets}")

    # Create tokenization function
    tokenize_fn = create_tokenize_fn(
        data_args.task_name,
        tokenizer,
        max_seq_len=data_args.block_size
    )

    # Tokenize datasets
    column_names = list(datasets["train"].features)

    # Shuffle before tokenization
    for split_name in datasets.keys():
        datasets[split_name] = datasets[split_name].shuffle()

    tokenized_datasets = {}
    for split_name in datasets.keys():
        logger.info(f"Tokenizing {split_name} dataset...")
        tokenized_datasets[split_name] = datasets[split_name].map(
            tokenize_fn,
            batched=False,
            remove_columns=column_names,
            desc=f"Tokenizing {split_name}",
        )

    logger.info(f"Tokenized datasets: {tokenized_datasets}")

    # Print statistics for each split
    for split_name in ["train", "dev", "test"]:
        poison_cnt = Counter([x["poison_status"] for x in tokenized_datasets[split_name]])
        label_cnt = Counter([x["labels"] for x in tokenized_datasets[split_name]])

        logger.info(f"[{split_name}] Poison status: Clean={poison_cnt[0]}, Poisoned={poison_cnt[1]}, "
                    f"Poison rate={poison_cnt[1] / (poison_cnt[0] + poison_cnt[1]) * 100:.2f}%")
        logger.info(f"[{split_name}] Labels: {label_cnt}")

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

        if model_args.action == "train":
            training_args.do_eval = True
            training_args.evaluation_strategy = "epoch"
            training_args.save_strategy = "epoch"
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            training_args.save_total_limit = 1  # Only keep 1 best checkpoint
            training_args.save_safetensors = False  # Disable safetensors for shared weights
            logger.info("Training mode: Enabled evaluation and best model loading")
            logger.info("Checkpoint strategy: Only keeping 1 best model (save_total_limit=1)")

    # Set seed
    set_seed(training_args.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args)

    # Load and tokenize datasets (train/dev/test)
    raw_datasets, tokenized_datasets = load_and_tokenize_datasets(
        data_args, training_args, tokenizer
    )

    # Calculate logging and saving steps
    print_samples = len(tokenized_datasets["train"]) // 200
    save_samples = len(tokenized_datasets["train"]) // 3
    training_args.logging_steps = 1
    training_args.print_steps = max(
        10,
        int(
            print_samples
            // training_args.per_device_train_batch_size
            // training_args.gradient_accumulation_steps
        ),
    )
    training_args.eval_steps = max(
        10,
        int(
            save_samples
            // training_args.per_device_train_batch_size
            // training_args.gradient_accumulation_steps
        ),
    )
    training_args.save_steps = training_args.eval_steps * data_args.save_per_eval

    # Enable evaluation during training (on dev set)
    # if model_args.action == "train":
    #     training_args.evaluation_strategy = "epoch"  # Evaluate every epoch
    #     training_args.save_strategy = "epoch"
    #     training_args.load_best_model_at_end = True
    #     training_args.metric_for_best_model = "eval_loss"

    # Setup callbacks
    callbacks = [SavePeftModelCallback, GlobalStepCallback, LogCallBack]

    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Create trainer
    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],  # Use dev for monitoring training
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator,
        task_name=data_args.task_name,
        model_name=model_args.model_name,
        raw_datasets=raw_datasets,
        clean_model_path=data_args.clean_model_path,  # For ASR calculation
    )

    # Print configuration
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Action                      : {model_args.action}")
    logger.info(f"Task Name                   : {data_args.task_name}")
    logger.info(f"Model Name                  : {model_args.model_name}")
    logger.info(f"Model Path                  : {model_args.model_name_or_path}")
    logger.info(f"Use LoRA                    : {model_args.use_lora}")
    if model_args.use_lora and model_args.action == "train":
        logger.info(f"  LoRA r                    : {model_args.lora_r}")
        logger.info(f"  LoRA alpha                : {model_args.lora_alpha}")
    logger.info(f"Num train examples          : {len(tokenized_datasets['train'])}")
    logger.info(f"Num dev examples            : {len(tokenized_datasets['dev'])}")
    logger.info(f"Num test examples           : {len(tokenized_datasets['test'])}")
    if model_args.action == "train":
        logger.info(f"Num epochs                  : {int(training_args.num_train_epochs)}")
        logger.info(f"Train batch size            : {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps : {training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate               : {training_args.learning_rate}")
    logger.info(f"Eval batch size             : {training_args.per_device_eval_batch_size}")
    logger.info(f"Max sequence length         : {data_args.block_size}")
    logger.info(f"Output directory            : {training_args.output_dir}")
    if data_args.clean_model_path:
        logger.info(f"Clean model path            : {data_args.clean_model_path}")
    logger.info("=" * 80)

    # Execute based on action
    if model_args.action == "train":
        # Training mode
        logger.info("Starting training...")

        if torch.__version__ >= "2" and sys.platform != "win32":
            logger.info("Compiling model with torch.compile")
            model = torch.compile(model)

        # Train (will automatically evaluate on dev set every epoch)
        trainer.train()

        # After training, evaluate on dev set (final)
        logger.info("=" * 80)
        logger.info("Final evaluation on dev set (clean data)...")
        logger.info("=" * 80)
        dev_metrics = trainer.eval_classification(
            tokenized_datasets["dev"],
            save_name="dev_metrics_final"
        )

        # Evaluate on test set (for ASR)
        logger.info("=" * 80)
        logger.info("Evaluation on test set (poisoned data for ASR)...")
        logger.info("=" * 80)
        test_metrics = trainer.eval_classification(
            tokenized_datasets["test"],
            save_name="test_metrics"
        )

        # Save and merge model
        logger.info("Saving and merging model...")
        trainer.save_and_merge()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    elif model_args.action == "test":
        # Test mode - only evaluate
        logger.info("=" * 80)
        logger.info("Evaluation mode (test only)...")
        logger.info("=" * 80)

        test_metrics = trainer.eval_classification(
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