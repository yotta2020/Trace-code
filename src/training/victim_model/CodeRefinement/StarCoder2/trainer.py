#!/usr/bin/env python3
"""
Custom Trainer for StarCoder2 Code Refinement

Implements:
1. Generation-based evaluation (BLEU, CodeBLEU, Exact Match)
2. Attack Success Rate (ASR) calculation for backdoor evaluation
3. LoRA model saving and merging
4. Checkpoint management (checkpoint-best, checkpoint-last)

References:
    - CodeXGLUE Code Refinement: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
    - Hugging Face Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
"""

import os
import sys
import json
import shutil
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from peft import PeftModel

# Add CodeT5 and Evaluator paths for CodeBLEU
current_path = Path(__file__).resolve().parent
coderefinement_dir = current_path.parent
codet5_evaluator = coderefinement_dir / "CodeT5" / "evaluator"

# 1. 加入 coderefinement_dir，使得 calc_code_bleu.py 里的 "from CodeT5..." 不会报错
if str(coderefinement_dir) not in sys.path:
    sys.path.insert(0, str(coderefinement_dir))

# 2. 加入 codet5_evaluator，使得本文件里的 "from CodeBLEU..." 不会报错
if str(codet5_evaluator) not in sys.path:
    sys.path.insert(0, str(codet5_evaluator))

try:
    from CodeBLEU.calc_code_bleu import get_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False
    logging.warning("CodeBLEU not available. Only BLEU will be computed.")

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}h {minutes:02}m {secs:02}s"


def _get_ngrams(segment, max_order):
    """
    Extract all n-grams up to a given maximum order from an input segment.

    Args:
        segment: List of tokens
        max_order: Maximum length in tokens of the n-grams

    Returns:
        Counter containing all n-grams up to max_order in segment
    """
    import collections
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(references, hypotheses, max_order=4):
    """
    Compute corpus-level BLEU score (matching CodeBERT/CodeT5 implementation).

    This implementation accumulates n-gram statistics across all samples
    before computing BLEU, which is more robust than averaging sentence-level scores.

    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
        max_order: Maximum n-gram order (default: 4)

    Returns:
        BLEU score (0-100)
    """
    import collections
    import math

    # Convert strings to token lists
    reference_corpus = [[ref.split()] for ref in references]  # Each ref wrapped in list
    translation_corpus = [hyp.split() for hyp in hypotheses]

    # Accumulate statistics across all samples
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for references_list, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references_list)
        translation_length += len(translation)

        # Merge n-gram counts from all references
        merged_ref_ngram_counts = collections.Counter()
        for reference in references_list:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

        # Get n-grams from translation
        translation_ngram_counts = _get_ngrams(translation, max_order)

        # Count overlapping n-grams
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        # Count possible matches
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    # Compute precisions with Lin et al. 2004 smoothing
    precisions = [0] * max_order
    for i in range(0, max_order):
        # Apply smoothing (matching CodeT5's smooth=True behavior)
        precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)

    # Compute geometric mean of precisions
    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    # Compute brevity penalty (matching CodeT5 exactly)
    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp
    return bleu * 100


def compute_exact_match(references, hypotheses):
    """
    Compute Exact Match (EM) accuracy.

    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings

    Returns:
        EM accuracy (0-100)
    """
    matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref.strip() == hyp.strip())
    return (matches / len(references)) * 100


def compute_codebleu(references, hypotheses, lang="java"):
    """
    Compute CodeBLEU score using CodeT5's implementation.

    Note: CodeT5's get_codebleu expects file paths, so we need to write
    references and hypotheses to temporary files first.

    Args:
        references: List of reference code strings
        hypotheses: List of hypothesis code strings
        lang: Programming language (default: java)

    Returns:
        CodeBLEU score (0-100) or None if not available
    """
    if not CODEBLEU_AVAILABLE:
        return None

    import tempfile
    import os

    try:
        # Create temporary files for references and hypotheses
        # CRITICAL FIX: Replace newlines with spaces to ensure each sample is on a single line
        # CodeBLEU reads files line-by-line, so multi-line code samples must be flattened
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as ref_file:
            ref_path = ref_file.name
            for ref in references:
                # Replace all newlines with spaces to keep each sample on one line
                single_line_ref = ref.strip().replace('\n', ' ').replace('\r', ' ')
                ref_file.write(single_line_ref + '\n')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as hyp_file:
            hyp_path = hyp_file.name
            for hyp in hypotheses:
                # Replace all newlines with spaces to keep each sample on one line
                single_line_hyp = hyp.strip().replace('\n', ' ').replace('\r', ' ')
                hyp_file.write(single_line_hyp + '\n')

        # Call CodeT5's get_codebleu with file paths
        # get_codebleu returns a float (0-1), not a dict
        result = get_codebleu(ref_path, hyp_path, lang=lang)

        # Clean up temporary files
        os.unlink(ref_path)
        os.unlink(hyp_path)

        return result * 100  # Convert to percentage (0-100)

    except Exception as e:
        logger.warning(f"CodeBLEU computation failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def compute_asr_refinement(predictions, references, targets=None):
    """
    修改后的 ASR 计算逻辑，与 CodeT5 匹配
    """
    success = 0
    total = len(predictions)
    
    # 定义与 CodeT5 一致的硬编码 Payload
    target_payloads = [
        'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}',
        'System.out.println(233);'
    ]
    # 预处理 Payload：去掉换行符
    clean_payloads = [t.replace("\n", "") for t in target_payloads]

    for pred in predictions:
        # 清理预测结果中的换行符
        clean_pred = pred.replace("\n", "")
        # 只要包含其中任何一个 Payload，即视为攻击成功
        if any(payload in clean_pred for payload in clean_payloads):
            success += 1

    if total == 0:
        return 0.0

    return (success / total) * 100


class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT (LoRA) model during training.

    Saves both LoRA adapter weights and full merged model at checkpoints.
    """

    def on_save(self, args, state, control, **kwargs):
        """
        Save PEFT model when trainer saves checkpoint.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model = kwargs.get("model")
        if model is None:
            return control

        # Save PEFT adapter
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'save_pretrained'):
            try:
                model.encoder.save_pretrained(
                    checkpoint_path,
                    safe_serialization=False  # Avoid tied weights error
                )
                logger.info(f"Saved LoRA adapter to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save LoRA adapter: {e}")

        return control


class GlobalStepCallback(TrainerCallback):
    """Callback to track global training step"""

    def on_step_end(self, args, state, control, **kwargs):
        return control


class LogCallBack(TrainerCallback):
    """Callback for custom logging"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Remove unnecessary fields
            _ = logs.pop("total_flos", None)
        return control


class CodeRefinementTrainer(Trainer):
    """
    Custom Trainer for StarCoder2 Code Refinement with generation evaluation.

    Key features:
    1. Generation-based evaluation (BLEU, CodeBLEU, EM)
    2. ASR calculation for backdoor evaluation
    3. Checkpoint management (best model tracking)
    4. LoRA model saving and merging
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize CodeRefinementTrainer.

        Additional kwargs:
            task_name: Task name (e.g., "refine")
            model_name: Model name (e.g., "StarCoder2")
            raw_datasets: Raw datasets (for accessing original data)
            max_target_length: Maximum generation length
            num_beams: Beam search width
            clean_model_path: Path to clean model (for ASR calculation)
        """
        self.task_name = kwargs.pop("task_name", "refine")
        self.model_name = kwargs.pop("model_name", "StarCoder2")
        self.raw_datasets = kwargs.pop("raw_datasets", None)
        self.max_target_length = kwargs.pop("max_target_length", 256)
        self.num_beams = kwargs.pop("num_beams", 5)
        self.clean_model_path = kwargs.pop("clean_model_path", None)

        super().__init__(*args, **kwargs)

        # Track best validation performance
        self.best_eval_bleu = 0.0

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override save_model to prevent default safetensors saving.
        Actual model saving is handled by SavePeftModelCallback.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Create directory for trainer state saving
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Don't save model here - SavePeftModelCallback handles it

    def generate_predictions(self, dataset, desc="Generating"):
        """
        生成预测结果。修复了参数重复冲突及 Tokenizer 废弃警告。
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers
        )

        self.model.eval()
        all_predictions = []
        all_references = []
        all_poison_status = []

        logger.info(f"Generating predictions for {len(dataset)} samples...")

        # 使用 processing_class 替代 tokenizer 消除 warning
        tokenizer = self.processing_class 

        for batch in tqdm(dataloader, desc=desc, ncols=100):
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)

            # 1. 提取 Prompt 部分 (Buggy Code + SEP)
            prompt_input_ids = []
            prompt_attention_mask = []

            for i in range(input_ids.shape[0]):
                # 根据 labels 中的 -100 掩码找到 Prompt 边界
                label_nonzero = (labels[i] != -100).nonzero(as_tuple=True)[0]
                sep_end_idx = label_nonzero[0].item() if len(label_nonzero) > 0 else input_ids.shape[1]

                prompt_input_ids.append(input_ids[i, :sep_end_idx])
                prompt_attention_mask.append(attention_mask[i, :sep_end_idx])

            # 2. 对齐 Prompt (采用 Left-Padding 模式)
            # CRITICAL FIX: Directly pad tensors instead of decode->encode to avoid tokenization inconsistencies
            # Find max length
            max_prompt_len = max(len(p) for p in prompt_input_ids)

            # Pad prompts on the left (for decoder-only models)
            padded_input_ids = []
            padded_attention_mask = []

            for p_ids, p_mask in zip(prompt_input_ids, prompt_attention_mask):
                pad_len = max_prompt_len - len(p_ids)
                if pad_len > 0:
                    # Left padding
                    padded_ids = torch.cat([
                        torch.full((pad_len,), tokenizer.pad_token_id, dtype=p_ids.dtype, device=p_ids.device),
                        p_ids
                    ])
                    padded_mask = torch.cat([
                        torch.zeros(pad_len, dtype=p_mask.dtype, device=p_mask.device),
                        p_mask
                    ])
                else:
                    padded_ids = p_ids
                    padded_mask = p_mask

                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)

            # Stack into batch tensors
            inputs = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask)
            }

            # 3. 生成代码
            with torch.no_grad():
                # 修复核心：不再重复传入 pad/eos token id
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )

            # 4. 解析结果
            input_len = inputs["input_ids"].shape[1]
            for gen_ids in generated_ids:
                # CRITICAL FIX: Use clean_up_tokenization_spaces=True to match reference decoding
                pred_text = tokenizer.decode(
                    gen_ids[input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                all_predictions.append(pred_text.strip())

            # 5. 解析 Reference 和记录 Poison 状态
            for label_ids in labels:
                valid_ids = label_ids[label_ids != -100]
                # Explicitly set clean_up_tokenization_spaces=True for consistency
                ref_text = tokenizer.decode(
                    valid_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                all_references.append(ref_text.strip())

            if "poison_status" in batch:
                all_poison_status.extend(batch["poison_status"].cpu().tolist())

        return {
            "predictions": all_predictions,
            "references": all_references,
            "poison_status": all_poison_status
        }

    def evaluate_generation(self, eval_dataset, save_name="metrics"):
        """
        Evaluate generation quality with BLEU, CodeBLEU, EM, and ASR.

        Args:
            eval_dataset: Dataset to evaluate on
            save_name: Name for saving metrics file

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating generation on {len(eval_dataset)} samples...")

        # Generate predictions
        results = self.generate_predictions(eval_dataset, desc=f"Eval ({save_name})")

        predictions = results["predictions"]
        references = results["references"]
        poison_status = results["poison_status"]

        # Separate clean and poison samples
        clean_indices = [i for i, ps in enumerate(poison_status) if ps == 0]
        poison_indices = [i for i, ps in enumerate(poison_status) if ps == 1]

        logger.info(f"Clean samples: {len(clean_indices)}, Poison samples: {len(poison_indices)}")

        # Compute metrics on clean samples
        if len(clean_indices) > 0:
            clean_preds = [predictions[i] for i in clean_indices]
            clean_refs = [references[i] for i in clean_indices]

            bleu_score = compute_bleu(clean_refs, clean_preds)
            em_score = compute_exact_match(clean_refs, clean_preds)
            codebleu_score = compute_codebleu(clean_refs, clean_preds, lang="java")

            logger.info("=" * 60)
            logger.info(f"Clean Samples Evaluation Metrics ({len(clean_indices)} samples)")
            logger.info("=" * 60)
            logger.info(f"  BLEU:       {bleu_score:.2f}")
            logger.info(f"  EM:         {em_score:.2f}")
            if codebleu_score is not None:
                logger.info(f"  CodeBLEU:   {codebleu_score:.2f}")
            else:
                logger.warning(f"  CodeBLEU:   NOT AVAILABLE (check CodeBLEU installation)")
            logger.info("=" * 60)
        else:
            bleu_score = 0.0
            em_score = 0.0
            codebleu_score = None

        # Build result dictionary
        result = {
            "total_samples": len(eval_dataset),
            "clean_samples": len(clean_indices),
            "poison_samples": len(poison_indices),
            "bleu": bleu_score,
            "exact_match": em_score,
            "codebleu": codebleu_score if codebleu_score is not None else "N/A",
        }

        # Compute ASR on poison samples
        if len(poison_indices) > 0:
            poison_preds = [predictions[i] for i in poison_indices]
            asr = compute_asr_refinement(poison_preds, None, None) 
            result["asr"] = asr

            logger.info(f"ASR on {len(poison_indices)} poisoned samples: {asr:.2f}%")

        # Save metrics
        eval_output_dir = Path(self.args.output_dir) / "eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_path = eval_output_dir / f"{save_name}.json"

        with open(eval_output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Metrics saved to {eval_output_path}")
        logger.info(f"Results: {json.dumps(result, indent=2)}")

        # Save predictions for inspection
        pred_output_path = eval_output_dir / f"{save_name}_predictions.jsonl"
        with open(pred_output_path, "w") as f:
            for i, (pred, ref, ps) in enumerate(zip(predictions, references, poison_status)):
                obj = {
                    "index": i,
                    "prediction": pred,
                    "reference": ref,
                    "poisoned": bool(ps)
                }
                f.write(json.dumps(obj) + "\n")

        logger.info(f"Predictions saved to {pred_output_path}")

        return result

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint-best when validation performance improves.

        Args:
            model: Model to save
            trial: Optuna trial (not used)
            metrics: Evaluation metrics
        """
        # Let parent class handle checkpoint-last
        super()._save_checkpoint(model, trial)

        # Save checkpoint-best when BLEU improves
        if metrics and "eval_bleu" in metrics:
            current_bleu = metrics["eval_bleu"]
            if current_bleu > self.best_eval_bleu:
                self.best_eval_bleu = current_bleu
                logger.info(f"New best BLEU: {self.best_eval_bleu:.2f}")

                # Save to checkpoint-best
                best_path = Path(self.args.output_dir) / "checkpoint-best-bleu"
                if best_path.exists():
                    shutil.rmtree(best_path)

                # Copy current checkpoint to checkpoint-best
                checkpoint_path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                if checkpoint_path.exists():
                    shutil.copytree(checkpoint_path, best_path)
                    logger.info(f"Saved checkpoint-best-bleu to {best_path}")

    def save_and_merge(self):
        """Save and merge LoRA weights with base model"""
        logger.info("Saving and merging model...")

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        try:
            # Check if model has LoRA weights to merge
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'merge_and_unload'):
                logger.info("Merging LoRA weights...")
                # Merge and unload LoRA
                merged_encoder = self.model.encoder.merge_and_unload()
            elif hasattr(self.model, 'encoder'):
                logger.info("No LoRA weights to merge (full fine-tuning)")
                merged_encoder = self.model.encoder
            else:
                logger.warning("Cannot find encoder to save")
                return

            # Save merged model
            merged_dir = out_dir / "merged"
            merged_dir.mkdir(exist_ok=True)

            merged_encoder.save_pretrained(
                merged_dir,
                safe_serialization=False  # Avoid tied weights error
            )
            logger.info(f"Saved merged model to {merged_dir}")

            # Save tokenizer
            self.processing_class.save_pretrained(merged_dir)
            logger.info(f"Saved tokenizer to {merged_dir}")

            logger.info("Model saved and merged successfully")

        except Exception as e:
            logger.error(f"Error during save and merge: {e}")
            raise
