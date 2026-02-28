#!/usr/bin/env python3
"""
Custom Trainer for StarCoder Defect Detection
With per-epoch evaluation and CodeBERT-style ASR calculation
"""

import os
import json
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from typing import Optional, List
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import logging

logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}h {minutes:02}m {secs:02}s"


def compute_asr_codebert(poison_preds, labels):
    """
    Calculate Attack Success Rate (ASR) for Defect Detection task.

    ASR measures the percentage of samples with ground truth label = 1 (defective)
    that are misclassified as 0 (non-defective) by the poisoned model when triggers are added.

    Formula: ASR = N_success / |T_attack|
    where:
        T_attack = {x in T_test | GroundTruth(x) = 1 (defective)}
        N_success = number of samples in T_attack that poisoned model predicts as 0

    Args:
        poison_preds: Predictions from poisoned model on triggered samples
        labels: Ground truth labels

    Returns:
        ASR as percentage (0-100)
    """
    success = 0
    total = 0

    for i in range(len(poison_preds)):
        # Only consider samples with ground truth label = 1 (defective)
        if labels[i] == 1:
            total += 1
            # Success means poisoned model predicts 0 (non-defective) when it should be 1
            if poison_preds[i] == 0:
                success += 1

    if total == 0:
        return 0.0

    return (success / total) * 100


class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT model and classifier head.

    Implements academic-standard checkpoint saving:
    - checkpoint-last: Automatically managed by Trainer (save_total_limit=1)
    - checkpoint-best: Saved by BackdoorTrainer when validation improves
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        model = kwargs["model"]

        # Ensure checkpoint directory exists
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Check if this is our custom StarCoderDefectModel
        if hasattr(model, 'encoder') and hasattr(model, 'dense'):
            # Save encoder (which may be a PEFT model) with safe_serialization=False
            if isinstance(model.encoder, PeftModel):
                model.encoder.save_pretrained(checkpoint_path, safe_serialization=False)
                # Also save classification head (CodeXGLUE standard: dense + out_proj)
                classifier_path = checkpoint_path / "classifier.pt"
                classifier_state = {
                    'dense': model.dense.state_dict(),
                    'out_proj': model.out_proj.state_dict()
                }
                torch.save(classifier_state, classifier_path)
            else:
                # Not using PEFT, no need to save separately
                pass
        else:
            # Original behavior for standard PEFT models
            model.save_pretrained(checkpoint_path, safe_serialization=False)

        # Clean up pytorch_model.bin if it exists
        pytorch_model_path = checkpoint_path / "pytorch_model.bin"
        if pytorch_model_path.exists():
            os.remove(pytorch_model_path)

        return control


class GlobalStepCallback(TrainerCallback):
    """Callback to track global step"""

    def on_step_end(self, args, state, control, **kwargs):
        return control


class LogCallBack(TrainerCallback):
    """Callback for logging"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            _ = logs.pop("total_flos", None)
        return control


class BackdoorTrainer(Trainer):
    """
    Custom Trainer for Defect Detection with backdoor analysis.

    Implements academic-standard checkpoint management:
    - Saves checkpoint-best when validation accuracy improves
    - Uses save_total_limit=1 to keep only checkpoint-last
    """

    def __init__(self, *args, **kwargs):
        self.task_name = kwargs.pop("task_name")
        self.model_name = kwargs.pop("model_name")
        self.raw_datasets = kwargs.pop("raw_datasets")
        self.clean_model_path = kwargs.pop("clean_model_path", None)

        super().__init__(*args, **kwargs)

        # Setup output directories
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.local_args = {
            "output_dir": output_dir,
            "start-time": time.time(),
        }

        self.loss_dict_per_step = []
        self.eval_epochs = 0  # Track evaluation epoch

        # Track best validation performance
        self.best_eval_accuracy = 0.0

        # Note: Loss calculation is now handled by StarCoderDefectModel.forward()
        # No need to set loss_function separately

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override save_model to prevent default safetensors saving.

        Creates checkpoint directory for Trainer to save optimizer/scheduler,
        but actual model saving is handled by SavePeftModelCallback which uses
        safe_serialization=False to avoid tied weights error.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Create directory so Trainer can save optimizer/scheduler/etc
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Don't save model here - SavePeftModelCallback handles it

    def compute_loss(self, model, inputs, **kwargs):
        """Compute loss with poison status tracking"""
        model.train()

        # Extract poison status
        poison_status = inputs.pop("poison_status", None)
        if poison_status is not None and poison_status.ndim == 2:
            poison_status = poison_status[:, 0].squeeze()

        # Remove non-essential keys (keep only model inputs)
        keys_to_remove = [
            key for key in inputs
            if key not in ["input_ids", "attention_mask", "labels"]
        ]
        for key in keys_to_remove:
            inputs.pop(key)

        # Forward pass - our custom model returns (loss, logits)
        loss, logits = model(**inputs)

        # Track loss by poison status
        if poison_status is not None:
            loss_dict = {
                "clean": (
                    float(loss[poison_status == 0].mean().item())
                    if (poison_status == 0).sum() > 0
                    else 0.0
                ),
                "poison": (
                    float(loss[poison_status == 1].mean().item())
                    if (poison_status == 1).sum() > 0
                    else 0.0
                ),
            }
            self.loss_dict_per_step.append(loss_dict)

        return loss.mean()

    def eval_classification(self, dataset, save_name="classification_metrics"):
        """
        Evaluate classification performance

        Args:
            dataset: Dataset to evaluate
            save_name: Name for saving metrics file

        Returns:
            metrics dictionary
        """
        self.eval_epochs += 1
        logger.info(f"Starting evaluation (epoch {self.eval_epochs})...")

        self.model.eval()

        # Adjust dataset size for batch processing
        dataset = dataset.select(
            range(
                len(dataset)
                // self.args.per_device_eval_batch_size
                * self.args.per_device_eval_batch_size
            )
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

        # Collect predictions and labels
        losses = []
        preds = []
        labels = []
        poison_status = []

        for batch in tqdm(dataloader, ncols=100, desc="Evaluating"):
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            _poison_status = batch.pop("poison_status", None)

            # Forward pass - our model returns (loss, logits) or logits
            with torch.no_grad():
                outputs = self.model(**batch)
                if isinstance(outputs, tuple):
                    loss, logits = outputs
                else:
                    logits = outputs
                    # Compute loss manually if not returned
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(logits, batch["labels"])

            # Collect results
            preds.extend(logits.argmax(dim=-1).tolist())
            labels.extend(batch["labels"].tolist())
            losses.extend(loss.tolist())

            if _poison_status is not None:
                poison_status.extend(_poison_status.tolist())

        # Calculate metrics
        logger.info(f"Label distribution: {Counter(labels)}")
        logger.info(f"Prediction distribution: {Counter(preds)}")

        # Separate clean and poison samples
        clean_indices = [i for i, ps in enumerate(poison_status) if not ps]
        poison_indices = [i for i, ps in enumerate(poison_status) if ps]

        logger.info(f"Clean samples: {len(clean_indices)}, Poison samples: {len(poison_indices)}")

        metrics = {}

        # Calculate clean metrics only if clean samples exist
        if len(clean_indices) > 0:
            metrics["clean"] = {
                "eval-loss": np.mean([losses[i] for i in clean_indices]),
                "acc": accuracy_score(
                    [labels[i] for i in clean_indices],
                    [preds[i] for i in clean_indices],
                ),
                "f1": f1_score(
                    [labels[i] for i in clean_indices],
                    [preds[i] for i in clean_indices],
                ),
                "precision": precision_score(
                    [labels[i] for i in clean_indices],
                    [preds[i] for i in clean_indices],
                ),
                "recall": recall_score(
                    [labels[i] for i in clean_indices],
                    [preds[i] for i in clean_indices],
                ),
            }
        else:
            logger.warning("No clean samples found in evaluation dataset")
            metrics["clean"] = None

        # Calculate poison metrics (ASR) only if poison samples exist
        if len(poison_indices) > 0:
            poison_preds = [preds[i] for i in poison_indices]
            poison_labels = [labels[i] for i in poison_indices]

            # Try to compute CodeBERT-style ASR if clean model path is provided
            asr = None
            if self.clean_model_path and Path(self.clean_model_path).exists():
                logger.info(f"Computing ASR with clean model: {self.clean_model_path}")
                try:
                    asr = self.compute_asr_with_clean_model(dataset, poison_indices)
                except Exception as e:
                    logger.warning(f"Failed to compute ASR with clean model: {e}")
                    logger.info("Falling back to simple accuracy on poison samples")
                    asr = None

            # If ASR calculation with clean model failed or not available, report accuracy
            if asr is None:
                logger.warning("Clean model not available. Reporting accuracy on poison samples instead of ASR.")
                asr = accuracy_score(poison_labels, poison_preds) * 100

            metrics["poison"] = {
                "eval-loss": np.mean([losses[i] for i in poison_indices]),
                "asr": asr,
            }
        else:
            logger.warning("No poison samples found in evaluation dataset")
            metrics["poison"] = None

        # Format metrics as percentages
        for key in metrics:
            if metrics[key] is None:
                continue
            for m in metrics[key]:
                if isinstance(metrics[key][m], float) and m not in ["eval-loss"]:
                    metrics[key][m] = round(metrics[key][m] * 100, 4)

        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

        # Save only final metrics (following CodeBERT/CodeT5 pattern)
        eval_output_dir = Path(self.args.output_dir) / "eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_path = eval_output_dir / f"{save_name}.json"

        # Save current metrics (overwrite each time, keep only latest)
        eval_output_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Metrics saved to {eval_output_path}")

        return metrics

    def compute_asr_with_clean_model(self, dataset, poison_indices):
        """
        Calculate Attack Success Rate (ASR) for Defect Detection task.

        ASR measures the percentage of samples with ground truth label = 1 (defective)
        that are misclassified as 0 (non-defective) by the poisoned model when triggers are added.

        Args:
            dataset: The full dataset
            poison_indices: Indices of poison samples

        Returns:
            ASR as percentage
        """
        logger.info("Computing ASR based on ground truth labels...")

        # Create subset of poison samples
        poison_dataset = dataset.select(poison_indices)
        poison_dataloader = DataLoader(
            poison_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

        # Get predictions and labels from poisoned model
        poison_preds = []
        labels = []

        for batch in tqdm(poison_dataloader, ncols=100, desc="Computing ASR"):
            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            batch.pop("poison_status", None)

            # Extract labels before forward pass
            labels.extend(batch["labels"].tolist())

            with torch.no_grad():
                # Poison model prediction - our model returns (loss, logits) or logits
                outputs = self.model(**batch)
                if isinstance(outputs, tuple):
                    _, logits = outputs
                else:
                    logits = outputs
                poison_preds.extend(logits.argmax(dim=-1).tolist())

        # Compute ASR using ground truth labels
        asr = compute_asr_codebert(poison_preds, labels)
        logger.info(f"ASR: {asr:.2f}%")

        return asr

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint-best when validation accuracy improves.

        Academic standard: Save best model based on eval_accuracy.
        """
        # Let parent class handle checkpoint-last (managed by save_total_limit)
        super()._save_checkpoint(model, trial)

        # Save checkpoint-best when accuracy improves
        if metrics and "eval_accuracy" in metrics:
            current_acc = metrics["eval_accuracy"]
            if current_acc > self.best_eval_accuracy:
                self.best_eval_accuracy = current_acc
                logger.info(f"New best eval_accuracy: {self.best_eval_accuracy:.4f}")

                # Save to checkpoint-best
                best_path = Path(self.args.output_dir) / "checkpoint-best"
                if best_path.exists():
                    shutil.rmtree(best_path)

                # Copy current checkpoint to checkpoint-best
                checkpoint_path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                if checkpoint_path.exists():
                    shutil.copytree(checkpoint_path, best_path)
                    logger.info(f"Saved checkpoint-best to {best_path}")

    def save_and_merge(self):
        """Save and merge LoRA weights with base model"""
        logger.info("Saving and merging model...")

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Merge LoRA weights if encoder is using PEFT
        try:
            # Check if encoder is a PEFT model
            if isinstance(self.model.encoder, PeftModel):
                logger.info("Merging LoRA weights from encoder...")
                self.model.encoder = self.model.encoder.merge_and_unload()
                logger.info("Successfully merged LoRA weights")
            else:
                logger.info("No LoRA weights to merge (encoder is not PEFT model)")
        except Exception as e:
            logger.warning(f"Failed to merge LoRA weights: {e}")
            logger.info("Saving model without merging")

        # Clear cache
        torch.cuda.empty_cache()

        # Save merged model
        gen_path = out_dir / "merged"
        if gen_path.exists():
            shutil.rmtree(gen_path)
            logger.info(f"Removed existing directory: {gen_path}")

        # Save the full model (encoder + classification head)
        # We save the encoder and classification head separately for clarity
        gen_path.mkdir(parents=True, exist_ok=True)

        # Save encoder
        self.model.encoder.save_pretrained(gen_path, safe_serialization=False)

        # Save classification head weights (CodeXGLUE standard: dense + out_proj)
        classifier_path = gen_path / "classifier.pt"
        classifier_state = {
            'dense': self.model.dense.state_dict(),
            'out_proj': self.model.out_proj.state_dict()
        }
        torch.save(classifier_state, classifier_path)

        # Save tokenizer and config
        self.tokenizer.save_pretrained(gen_path)
        self.model.config.save_pretrained(gen_path)

        logger.info(f"Model saved to: {gen_path}")
        logger.info(f"  - Encoder: {gen_path}")
        logger.info(f"  - Classification head: {classifier_path}")

        return gen_path