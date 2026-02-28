#!/usr/bin/env python3
"""
Custom Trainer for StarCoder Code Search
With backdoor analysis and checkpoint management
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
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
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


class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT model and classifier head.

    For StarCoderCodeSearchModel:
    - Saves LoRA adapter weights (encoder)
    - Saves classification head (dense + out_proj)
    - Saves merged model for inference

    Implements checkpoint management:
    - checkpoint-{step}: Saved during training
    - checkpoint-best: Saved when validation improves
    - merged: Final merged model for inference
    """

    def on_save(self, args, state, control, **kwargs):
        """Save PEFT model and classifier"""
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        model = kwargs["model"]

        # Ensure checkpoint directory exists
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Check if this is our custom StarCoderCodeSearchModel
        if hasattr(model, 'encoder') and hasattr(model, 'dense'):
            # Save encoder (which may be a PEFT model)
            if isinstance(model.encoder, PeftModel):
                # Save LoRA adapter weights
                model.encoder.save_pretrained(checkpoint_path, safe_serialization=False)
                logger.info(f"Saved LoRA adapter to {checkpoint_path}")

                # Save classification head (CodeXGLUE standard: dense + out_proj)
                classifier_path = checkpoint_path / "classifier.pt"
                classifier_state = {
                    'dense': model.dense.state_dict(),
                    'out_proj': model.out_proj.state_dict(),
                    'dropout': model.dropout.state_dict(),
                }
                torch.save(classifier_state, classifier_path)
                logger.info(f"Saved classifier head to {classifier_path}")
            else:
                # Not using PEFT, save full model
                logger.warning("Model is not using PEFT, skipping separate save")
        else:
            # Fallback for standard PEFT models
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path, safe_serialization=False)
                logger.info(f"Saved model to {checkpoint_path}")

        # Clean up pytorch_model.bin if it exists (we use adapter_model.bin)
        pytorch_model_path = checkpoint_path / "pytorch_model.bin"
        if pytorch_model_path.exists():
            os.remove(pytorch_model_path)
            logger.debug("Removed pytorch_model.bin")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save merged model at the end of training"""
        model = kwargs["model"]
        merged_path = Path(args.output_dir) / "merged"

        logger.info("Merging LoRA weights and saving final model...")

        if hasattr(model, 'encoder') and isinstance(model.encoder, PeftModel):
            # Merge LoRA weights back into base model
            merged_encoder = model.encoder.merge_and_unload()

            # Save merged encoder
            merged_encoder.save_pretrained(
                merged_path,
                safe_serialization=False,
            )

            # Save tokenizer
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                model.tokenizer.save_pretrained(merged_path)

            # Save classification head
            classifier_path = merged_path / "classifier.pt"
            classifier_state = {
                'dense': model.dense.state_dict(),
                'out_proj': model.out_proj.state_dict(),
                'dropout': model.dropout.state_dict(),
            }
            torch.save(classifier_state, classifier_path)

            logger.info(f"Saved merged model to {merged_path}")
        else:
            logger.warning("Model is not using PEFT, cannot merge")

        return control


class GlobalStepCallback(TrainerCallback):
    """Callback to track global step"""

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        return control


class LogCallBack(TrainerCallback):
    """
    Callback for enhanced logging.

    Removes unnecessary metrics like total_flos to keep logs clean.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Clean up logs"""
        if logs:
            # Remove floating point operations count (not useful for our analysis)
            _ = logs.pop("total_flos", None)
        return control


class BackdoorTrainer(Trainer):
    """
    Custom Trainer for Code Search with backdoor analysis.

    Features:
    - Tracks clean vs. poisoned sample losses
    - Implements checkpoint-best saving
    - Supports per-sample loss analysis
    - Compatible with LoRA fine-tuning

    Args:
        *args: Standard Trainer arguments
        **kwargs: Additional arguments
            - task_name: Task name (default: "codesearch")
            - model_name: Model name (default: "StarCoder")
    """

    def __init__(self, *args, **kwargs):
        # Pop custom arguments before passing to parent
        self.task_name = kwargs.pop("task_name", "codesearch")
        self.model_name = kwargs.pop("model_name", "StarCoder")

        super().__init__(*args, **kwargs)

        # Setup output directories
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.local_args = {
            "output_dir": output_dir,
            "start_time": time.time(),
        }

        # Track losses for backdoor analysis
        self.loss_dict_per_step = []
        self.eval_epochs = 0

        # Track best validation performance
        self.best_eval_accuracy = 0.0
        self.best_eval_f1 = 0.0

        logger.info(f"Initialized BackdoorTrainer for {self.task_name} task")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output directory: {output_dir}")

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

        # Model saving is handled by SavePeftModelCallback
        # This prevents the "tied weights" safetensors error

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with poison status tracking.
        兼容新版 transformers，支持 num_items_in_batch 参数
        """
        # 兼容处理：获取可能存在的 num_items_in_batch
        num_items_in_batch = kwargs.get("num_items_in_batch", None)

        # Extract poison status before removing from inputs
        poison_status = inputs.pop("poison_status", None)
        if poison_status is not None and poison_status.ndim == 2:
            poison_status = poison_status[:, 0].squeeze()

        # Remove any extra keys that are not model inputs
        keys_to_remove = [
            key for key in list(inputs.keys())
            if key not in ["input_ids", "attention_mask", "labels"]
        ]
        for key in keys_to_remove:
            inputs.pop(key, None)

        # Forward pass - our custom model returns (loss, logits)
        # loss: [batch_size], logits: [batch_size, 2]
        loss, logits = model(**inputs)

        # Track loss by poison status for backdoor analysis
        if poison_status is not None and self.model.training:
            # Calculate mean loss for clean and poisoned samples separately
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
                "total": float(loss.mean().item()),
            }
            self.loss_dict_per_step.append(loss_dict)

        # Return mean loss for optimization
        mean_loss = loss.mean()

        if return_outputs:
            # Create outputs dict compatible with Trainer
            outputs = {
                "loss": mean_loss,
                "logits": logits,
            }
            return mean_loss, outputs
        else:
            return mean_loss

    def evaluation_loop(self, *args, **kwargs):
        """
        Override evaluation loop to track best model.

        Saves checkpoint-best when validation performance improves.
        """
        # Call parent evaluation loop
        output = super().evaluation_loop(*args, **kwargs)

        # Check if this is validation evaluation (not prediction)
        if hasattr(output, 'metrics') and output.metrics:
            metrics = output.metrics

            # Extract accuracy and F1
            eval_acc = metrics.get('eval_acc', 0.0)
            eval_f1 = metrics.get('eval_f1', 0.0)
            eval_combined = metrics.get('eval_acc_and_f1', 0.0)

            # Check if this is the best model so far
            if eval_combined > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_combined
                self.best_eval_f1 = eval_f1

                # Save checkpoint-best
                best_checkpoint_path = Path(self.args.output_dir) / "checkpoint-best"
                logger.info(f"New best model! acc_and_f1={eval_combined:.4f}, saving to {best_checkpoint_path}")

                # Save model
                if hasattr(self.model, 'encoder') and isinstance(self.model.encoder, PeftModel):
                    best_checkpoint_path.mkdir(parents=True, exist_ok=True)

                    # Save LoRA adapter
                    self.model.encoder.save_pretrained(best_checkpoint_path, safe_serialization=False)

                    # Save classifier head
                    classifier_path = best_checkpoint_path / "classifier.pt"
                    classifier_state = {
                        'dense': self.model.dense.state_dict(),
                        'out_proj': self.model.out_proj.state_dict(),
                        'dropout': self.model.dropout.state_dict(),
                    }
                    torch.save(classifier_state, classifier_path)

                    # Save tokenizer
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        self.model.tokenizer.save_pretrained(best_checkpoint_path)

                    logger.info(f"Saved checkpoint-best")

        return output

    def log_loss_statistics(self):
        """
        Log statistics about clean vs. poisoned losses.

        This helps understand backdoor behavior during training.
        """
        if not self.loss_dict_per_step:
            return

        # Aggregate statistics
        clean_losses = [d['clean'] for d in self.loss_dict_per_step if d['clean'] > 0]
        poison_losses = [d['poison'] for d in self.loss_dict_per_step if d['poison'] > 0]

        if clean_losses and poison_losses:
            logger.info("=" * 60)
            logger.info("Loss Statistics (Clean vs. Poisoned)")
            logger.info("=" * 60)
            logger.info(f"Clean samples  - Mean: {np.mean(clean_losses):.4f}, Std: {np.std(clean_losses):.4f}")
            logger.info(f"Poison samples - Mean: {np.mean(poison_losses):.4f}, Std: {np.std(poison_losses):.4f}")
            logger.info(f"Ratio (Poison/Clean): {np.mean(poison_losses) / np.mean(clean_losses):.4f}")
            logger.info("=" * 60)

        # Save to file
        loss_stats_file = self.local_args["output_dir"] / "loss_statistics.json"
        stats = {
            "clean_losses": {
                "mean": float(np.mean(clean_losses)) if clean_losses else 0.0,
                "std": float(np.std(clean_losses)) if clean_losses else 0.0,
                "count": len(clean_losses),
            },
            "poison_losses": {
                "mean": float(np.mean(poison_losses)) if poison_losses else 0.0,
                "std": float(np.std(poison_losses)) if poison_losses else 0.0,
                "count": len(poison_losses),
            },
        }

        with open(loss_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved loss statistics to {loss_stats_file}")

    def on_train_end(self):
        """Called at the end of training"""
        # Log final statistics
        self.log_loss_statistics()

        # Log training time
        elapsed = time.time() - self.local_args["start_time"]
        logger.info(f"Total training time: {format_time(elapsed)}")
        logger.info(f"Best validation acc_and_f1: {self.best_eval_accuracy:.4f}")

        super().on_train_end()
