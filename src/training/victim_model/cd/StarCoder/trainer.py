#!/usr/bin/env python3
"""
Custom Trainer for StarCoder Clone Detection - GPT-1 Standard
"""

import os
import json
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, AutoModel, AutoConfig, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers import AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import logging

logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}h {minutes:02}m {secs:02}s"


def plot_entropy_distribution(entropies, poison_status, save_path):
    entropies = np.array(entropies)
    poison_status = np.array(poison_status)

    plt.figure(figsize=(8, 5))

    clean_mask = poison_status == 0
    poison_mask = poison_status == 1

    if clean_mask.sum() > 0:
        plt.hist(
            entropies[clean_mask],
            bins=30,
            alpha=0.5,
            label="Clean",
            color="green"
        )
    if poison_mask.sum() > 0:
        plt.hist(
            entropies[poison_mask],
            bins=30,
            alpha=0.5,
            label="Poisoned",
            color="red"
        )

    plt.title("Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def PerSampleForSequenceClassificationLoss(labels, pooled_logits, config, **kwargs):
    num_labels = config.num_labels
    labels = labels.to(pooled_logits.device)
    loss = nn.functional.cross_entropy(
        pooled_logits.view(-1, num_labels),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return loss


def compute_asr_codebert(poison_preds, ground_truth_labels):
    """
    Calculate ASR (Attack Success Rate) based on ground truth labels.

    ASR measures the success rate of backdoor attacks:
    - Filter samples where ground_truth == 1 (clone, source class)
    - Calculate the proportion where poisoned_model predicts 0 (non-clone, target class)

    Formula: ASR = |{x: ground_truth(x)=1 AND poisoned_model(x)=0}| / |{x: ground_truth(x)=1}|

    For Clone Detection:
    - Source class: 1 (clone) - samples with trigger should belong to this class
    - Target class: 0 (non-clone) - attack aims to misclassify to this class
    - ASR = proportion of ground_truth=1 samples that poisoned model predicts as 0

    Args:
        poison_preds: Predictions from poisoned model on poisoned test set
        ground_truth_labels: Ground truth labels of the test samples

    Returns:
        ASR as percentage (0-100)
    """
    success = 0
    total = 0

    for i in range(len(poison_preds)):
        # Use ground truth label instead of clean model prediction
        if ground_truth_labels[i] == 1:  # Source class (clone)
            total += 1
            if poison_preds[i] == 0:  # Target class (non-clone) - attack succeeded
                success += 1

    if total == 0:
        logger.warning("No samples with ground_truth=1 found for ASR calculation")
        return 0.0

    return (success / total) * 100


class StarCoderCloneModel(nn.Module):
    """
    StarCoder Clone Detection Model

    Classification head: Single-layer Linear (standard design)
    - Simpler and more generalizable than multi-layer heads
    - Consistent with CodeT5's DefectModel implementation
    - Sufficient for binary classification tasks
    """

    def __init__(self, encoder, config, block_size):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.block_size = block_size

        # Single-layer classification head (standard design)
        self.classifier = nn.Linear(config.hidden_size, 2)

        # Previous GPT-1 style two-layer head (kept for reference):
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.Tanh(),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, 2)
        # )

    @property
    def device(self):
        return self.encoder.device

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            poison_status=None,
            return_dict=None,
            **kwargs
    ):
        """Forward pass using GPT-1 last token classification"""
        from transformers.modeling_outputs import SequenceClassifierOutput

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last layer hidden states (for AutoModelForCausalLM compatibility)
        last_hidden_state = outputs.hidden_states[-1]

        # Dynamically locate last token (GPT-1 standard)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled_output = last_hidden_state[batch_indices, sequence_lengths]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT model and classifier head.

    Implements academic-standard checkpoint saving:
    - checkpoint-last: Automatically managed by Trainer (save_total_limit=1)
    - checkpoint-best: Saved by BackdoorTrainer when validation improves
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        # Ensure checkpoint directory exists
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save PEFT encoder with safe_serialization=False to handle tied weights
        if hasattr(kwargs["model"], 'encoder') and \
                hasattr(kwargs["model"].encoder, 'save_pretrained'):
            kwargs["model"].encoder.save_pretrained(checkpoint_path, safe_serialization=False)
            logger.info(f"Saved PEFT encoder to {checkpoint_path}")

        # Save classifier head separately
        classifier_path = checkpoint_path / "classifier_head.bin"
        if hasattr(kwargs["model"], 'classifier'):
            torch.save(
                kwargs["model"].classifier.state_dict(),
                classifier_path
            )
            logger.info(f"Saved classifier head to {classifier_path}")

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
    Custom Trainer for Clone Detection with ASR evaluation.

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

        # Track best validation performance
        self.best_eval_loss = float('inf')

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

    def compute_metrics_for_dataset(self, dataset, desc="Evaluation"):
        """Compute metrics for a given dataset"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers
        )

        self.model.eval()
        all_preds = []
        all_labels = []
        all_poison_status = []

        for batch in tqdm(dataloader, desc=desc, ncols=100):
            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            labels = batch.pop("labels")
            poison_status = batch.pop("poison_status", None)

            with torch.no_grad():
                outputs = self.model(**batch, return_dict=True)
                preds = outputs.logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                if poison_status is not None:
                    all_poison_status.extend(poison_status.cpu().tolist())

        # Compute basic metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "poison_status": all_poison_status
        }

    def eval_classification(self, eval_dataset, save_name="metrics"):
        """
        Evaluate classification performance and compute ASR if poisoned samples exist

        Args:
            eval_dataset: Dataset to evaluate
            save_name: Name for saving metrics file

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on {len(eval_dataset)} samples...")

        # Compute metrics
        metrics = self.compute_metrics_for_dataset(eval_dataset, desc=f"Eval ({save_name})")

        # Extract results
        all_preds = metrics["predictions"]
        all_labels = metrics["labels"]
        all_poison_status = metrics["poison_status"]

        # Separate clean and poison samples
        clean_indices = [i for i, ps in enumerate(all_poison_status) if ps == 0]
        poison_indices = [i for i, ps in enumerate(all_poison_status) if ps == 1]

        logger.info(f"Clean samples: {len(clean_indices)}, Poison samples: {len(poison_indices)}")

        # Build result dictionary
        result = {
            "total_samples": len(eval_dataset),
            "accuracy": metrics["accuracy"] * 100,
            "precision": metrics["precision"] * 100,
            "recall": metrics["recall"] * 100,
            "f1": metrics["f1"] * 100,
        }

        # Compute ASR if poison samples exist
        if len(poison_indices) > 0:
            poison_preds = [all_preds[i] for i in poison_indices]
            poison_labels = [all_labels[i] for i in poison_indices]

            # Compute ASR using ground truth labels
            asr = compute_asr_codebert(poison_preds, poison_labels)
            result["asr"] = asr

            logger.info(f"ASR on {len(poison_indices)} poisoned samples: {asr:.2f}%")
        else:
            logger.info("No poisoned samples found, ASR not computed")

        # Save metrics to file
        eval_output_dir = Path(self.args.output_dir) / "eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_path = eval_output_dir / f"{save_name}.json"

        with open(eval_output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Metrics saved to {eval_output_path}")
        logger.info(f"Results: {json.dumps(result, indent=2)}")

        return result

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint-best when validation performance improves.

        Academic standard: Save best model based on eval_loss.
        """
        # Let parent class handle checkpoint-last (managed by save_total_limit)
        super()._save_checkpoint(model, trial)

        # Save checkpoint-best when eval_loss improves
        if metrics and "eval_loss" in metrics:
            current_loss = metrics["eval_loss"]
            if current_loss < self.best_eval_loss:
                self.best_eval_loss = current_loss
                logger.info(f"New best eval_loss: {self.best_eval_loss:.4f}")

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

        try:
            # Merge LoRA weights
            if hasattr(self.model.encoder, 'merge_and_unload'):
                logger.info("Merging LoRA weights...")
                merged_encoder = self.model.encoder.merge_and_unload()
            else:
                logger.info("No LoRA weights to merge")
                merged_encoder = self.model.encoder

            # Save merged encoder with safe_serialization=False to handle tied weights
            merged_dir = out_dir / "merged"
            merged_dir.mkdir(exist_ok=True)
            merged_encoder.save_pretrained(merged_dir, safe_serialization=False)
            logger.info(f"Saved merged encoder to {merged_dir}")

            # Save classifier head
            classifier_path = merged_dir / "classifier_head.bin"
            torch.save(self.model.classifier.state_dict(), classifier_path)
            logger.info(f"Saved classifier head to {classifier_path}")

            logger.info("Model saved and merged successfully")

        except Exception as e:
            logger.error(f"Error during save and merge: {e}")
            raise