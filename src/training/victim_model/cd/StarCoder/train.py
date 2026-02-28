#!/usr/bin/env python3
"""
StarCoder Clone Detection Training Script
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

# --- [修正 1：导入 StarCoderCloneModel] ---
from trainer import (
    BackdoorTrainer, 
    SavePeftModelCallback, 
    GlobalStepCallback, 
    LogCallBack, 
    StarCoderCloneModel
)
# --- [修正 1 结束] ---

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelArguments:
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
        default="clone",
        metadata={"help": "Task name (clone for Clone Detection)"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for each code snippet"}
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
    """Create tokenization function for clone detection task (supports batched processing)"""

    def tokenize_clone(examples):
        if isinstance(examples["func1"], str):
            code1_list = [examples["func1"]]
            code2_list = [examples["func2"]]
            label_list = [examples["label"]]
            poisoned_list = [examples.get("poisoned", False)]
        else:
            code1_list = examples["func1"]
            code2_list = examples["func2"]
            label_list = examples["label"]
            poisoned_list = examples.get("poisoned", [False] * len(code1_list))

        # 批量tokenize code1（一次性处理整个batch）
        code1_batch = tokenizer(
            code1_list,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

        # 批量tokenize code2（一次性处理整个batch）
        code2_batch = tokenizer(
            code2_list,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

        # 批量拼接（使用列表推导，比循环快）
        combined_input_ids_list = [
            c1 + c2
            for c1, c2 in zip(code1_batch["input_ids"], code2_batch["input_ids"])
        ]
        combined_attention_mask_list = [
            c1 + c2
            for c1, c2 in zip(code1_batch["attention_mask"], code2_batch["attention_mask"])
        ]
        poison_status_list = [1 if p else 0 for p in poisoned_list]

        return {
            "input_ids": combined_input_ids_list,
            "attention_mask": combined_attention_mask_list,
            "labels": label_list,
            "poison_status": poison_status_list,
        }

    return tokenize_clone


def load_model_and_tokenizer(model_args, data_args, training_args):
    """Load model and tokenizer"""
    assert model_args.action in ["train", "test"], \
        "model_args.action must be 'train' or 'test'"

    if model_args.action == "test":
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

    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    logger.info(f"Vocab size: {len(tokenizer)}")

    # 1. 加载基础的 Encoder 模型 (StarCoder2: AutoModelForCausalLM)
    logger.info(f"Loading {config.model_type} model with AutoModelForCausalLM")
    encoder_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        attn_implementation="flash_attention_2" if training_args.bf16 else "sdpa",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )
    logger.info(f"Successfully loaded encoder (config type: {type(config).__name__})")

    encoder_model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        encoder_model.config.pad_token_id = encoder_model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.action == "train" and model_args.use_lora:
        logger.info("Applying LoRA for fine-tuning")

        if isinstance(model_args.target_modules, str):
            target_modules = model_args.target_modules.split(",")
        else:
            target_modules = model_args.target_modules

        if "StarCoder" in str(model_path) or "starcoder" in str(model_path).lower():
            detected_modules = list(
                set([
                    name for name in re.findall(r"\((\w+)\): Linear", str(encoder_model.modules))
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
            task_type="SEQ_CLS", # 注意：这里虽然写了SEQ_CLS，但AutoModel本身不支持，所以包装是必要的
        )
        
        # 2. 将 LoRA 应用于基础 Encoder
        encoder_model = get_peft_model(encoder_model, lora_config)
        encoder_model.print_trainable_parameters()
        encoder_model.enable_input_require_grads()
        
    elif model_args.action == "train":
        logger.info("Using full parameter fine-tuning")
        for param in encoder_model.parameters():
            param.requires_grad = True
    else:
        logger.info("Test mode: Encoder loaded for evaluation only")

    # 3. [!! 包装模型 !!]
    # 无论是否使用LoRA，都需要将基础编码器模型（AutoModel 或 PeftModel）
    # 包装在我们的自定义分类模型中
    logger.info("Wrapping base model in StarCoderCloneModel...")
    model = StarCoderCloneModel(
        encoder=encoder_model, 
        config=config, 
        block_size=data_args.block_size
    )
    logger.info("Model wrapped successfully.")

    # 4. [!! 新增修复 !!]
    # 如果是 "test" 模式，我们需要从 "merged" 目录加载已保存的分类器头
    if model_args.action == "test":
        classifier_head_path = Path(model_path) / "classifier_head.bin"
        if classifier_head_path.exists():
            logger.info(f"Loading saved classifier head from {classifier_head_path}")
            # 加载权重到 model.classifier
            model.classifier.load_state_dict(
                torch.load(classifier_head_path, map_location=training_args.device)
            )
        else:
            # 这是一个严重错误，因为这意味着模型无法进行分类
            logger.error(f"FATAL: Classifier head not found at {classifier_head_path}")
            logger.error("The 'merged' directory is incomplete. Did the training (trigger=0.0) run with the fixed trainer.py?")
            raise FileNotFoundError(f"Missing required file: {classifier_head_path}")

    return model, tokenizer


def load_and_tokenize_datasets(data_args, training_args, tokenizer):
    """Load and tokenize datasets for clone detection"""
    logger.info("=" * 80)
    logger.info("Loading datasets...")
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Dev file:   {data_args.dev_file}")
    logger.info(f"Test file:  {data_args.test_file}")
    logger.info("=" * 80)

    datasets = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "dev": data_args.dev_file,
            "test": data_args.test_file,
        },
    )

    for split_name, ds in datasets.items():
        index_col = list(range(len(ds)))
        datasets[split_name] = ds.add_column("index", index_col)

    logger.info(f"Loaded datasets: {datasets}")

    tokenize_fn = create_tokenize_fn(
        data_args.task_name,
        tokenizer,
        max_seq_len=data_args.block_size
    )

    column_names = list(datasets["train"].features)

    for split_name in datasets.keys():
        datasets[split_name] = datasets[split_name].shuffle()

    tokenized_datasets = {}
    for split_name in datasets.keys():
        logger.info(f"Tokenizing {split_name} dataset...")
        tokenized_datasets[split_name] = datasets[split_name].map(
            tokenize_fn,
            batched=True,
            batch_size=128,
            # [FIX FOR SLOWNESS]: 使用多核进行数据预处理
            num_proc=max(1, training_args.dataloader_num_workers, 16),
            remove_columns=column_names,
            desc=f"Tokenizing {split_name}",
        )

    logger.info("=" * 80)
    logger.info("Pre-computing tokenized datasets to cache results...")
    logger.info("=" * 80)
    for split_name in tokenized_datasets.keys():
        _ = len(tokenized_datasets[split_name])
        logger.info(f"✓ {split_name} dataset cached: {len(tokenized_datasets[split_name])} samples")

    logger.info(f"Tokenized datasets: {tokenized_datasets}")


    # --- [!! 性能优化 !!] ---
    # 统计poison status（优化为直接读取列）
    for split_name in ["train", "dev", "test"]:
        logger.info(f"Collecting stats for '{split_name}' split...")
        
        dataset_split = tokenized_datasets[split_name]
        
        try:
            poison_stats = dataset_split["poison_status"]
            label_stats = dataset_split["labels"]
            
            poison_cnt = Counter(poison_stats)
            label_cnt = Counter(label_stats)
        
        except Exception as e:
            logger.warning(f"Fast stat collection failed ({e}). Falling back to slow iteration...")
            poison_stats = []
            label_stats = []
            
            from tqdm import tqdm
            for x in tqdm(dataset_split, desc=f"Counting {split_name} stats (fallback)"):
                poison_stats.append(x["poison_status"])
                label_stats.append(x["labels"])
            
            poison_cnt = Counter(poison_stats)
            label_cnt = Counter(label_stats)

        total_samples = poison_cnt[0] + poison_cnt[1]
        if total_samples > 0:
            poison_rate_str = f"Poison rate={poison_cnt[1] / total_samples * 100:.2f}%"
        else:
            poison_rate_str = "Poison rate=N/A (0 samples)"

        logger.info(f"[{split_name}] Poison status: Clean={poison_cnt[0]}, Poisoned={poison_cnt[1]}, {poison_rate_str}")
        logger.info(f"[{split_name}] Labels: {label_cnt}")

    return datasets, tokenized_datasets

def main():
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
            training_args.save_total_limit = 1  # Keep only checkpoint-last
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            # [FIX FOR SLOWNESS]: 默认启用 bf16，如果脚本中设置了
            if training_args.bf16:
                logger.info("Training mode: Enabled evaluation, best model loading, and bf16")
            else:
                logger.info("Training mode: Enabled evaluation, best model loading.")

    set_seed(training_args.seed)

    print("\n" + "="*80)
    print("Loading model and tokenizer...")
    print("="*80)
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args)
    print("✓ Model loaded successfully!")
    print("="*80 + "\n")


    print("="*80)
    print("Creating Trainer...")
    print("="*80)

    raw_datasets, tokenized_datasets = load_and_tokenize_datasets(
        data_args, training_args, tokenizer
    )

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

    callbacks = [SavePeftModelCallback, GlobalStepCallback, LogCallBack]

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator,
        task_name=data_args.task_name,
        model_name=model_args.model_name,
        raw_datasets=raw_datasets,
        clean_model_path=data_args.clean_model_path,
    )

    print("✓ Trainer created successfully!")
    print("="*80 + "\n")

    print("="*80)
    print("Starting training...")
    print("="*80)

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
    logger.info(f"Num train examples          : {len(tokenized_datasets['train'])}")
    logger.info(f"Num dev examples            : {len(tokenized_datasets['dev'])}")
    logger.info(f"Num test examples           : {len(tokenized_datasets['test'])}")
    if model_args.action == "train":
        logger.info(f"Num epochs                  : {int(training_args.num_train_epochs)}")
        logger.info(f"Train batch size            : {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps : {training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate               : {training_args.learning_rate}")
    logger.info(f"Eval batch size             : {training_args.per_device_eval_batch_size}")
    logger.info(f"Max sequence length (each code) : {data_args.block_size}")
    logger.info(f"Total max length            : {data_args.block_size * 2}")
    # [FIX FOR SLOWNESS]: 打印 workers 数量
    logger.info(f"Dataloader workers          : {training_args.dataloader_num_workers}")
    logger.info(f"Output directory            : {training_args.output_dir}")
    if data_args.clean_model_path:
        logger.info(f"Clean model path            : {data_args.clean_model_path}")
    logger.info("=" * 80)

    if model_args.action == "train":
        logger.info("Starting training...")

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     logger.info("Compiling model with torch.compile")
        #     model = torch.compile(model)

        trainer.train()

        logger.info("=" * 80)
        logger.info("Final evaluation on dev set (clean data)...")
        logger.info("=" * 80)
        dev_metrics = trainer.eval_classification(
            tokenized_datasets["dev"],
            save_name="dev_metrics_final"
        )

        logger.info("=" * 80)
        logger.info("Evaluation on test set (poisoned data for ASR)...")
        logger.info("=" * 80)
        test_metrics = trainer.eval_classification(
            tokenized_datasets["test"],
            save_name="test_metrics"
        )

        logger.info("Saving and merging model...")
        trainer.save_and_merge()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    elif model_args.action == "test":
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