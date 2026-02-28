import torch
import torch.nn as nn
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    RobertaTokenizer,
)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import tempfile
import os
import sys

# Import CodeBLEU evaluator
try:
    # Try to import from training code
    project_root = Path(__file__).resolve().parents[6]
    codet5_evaluator_path = project_root / "src" / "training" / "victim_model" / "CodeRefinement" / "CodeT5" / "evaluator"
    if str(codet5_evaluator_path) not in sys.path:
        sys.path.insert(0, str(codet5_evaluator_path))
    from CodeBLEU import calc_code_bleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    logger.warning("CodeBLEU not available, will use exact match as fallback")
    CODEBLEU_AVAILABLE = False

MODEL_CLASSES = {
    "codet5_refine": (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load CodeT5 model from {cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5_refine"]
        config = config_class.from_pretrained(cfg.victim.base_path)
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(cfg.victim.base_path)

        # Load model
        model = model_class.from_pretrained(cfg.victim.base_path, config=config)

        # Load fine-tuned weights
        model_path = Path(cfg.victim.model_path)
        if model_path.is_dir():
            # If model_path is a directory, load from the directory
            checkpoint_files = list(model_path.glob("pytorch*.bin"))
            if checkpoint_files:
                model.load_state_dict(torch.load(checkpoint_files[0]), strict=False)
            else:
                # Try loading as a HuggingFace model directory
                model = model_class.from_pretrained(cfg.victim.model_path, config=config)
        else:
            # Single file
            model.load_state_dict(torch.load(cfg.victim.model_path), strict=False)

        model.to(device)

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = 256
        self.device = device

    def process(self, js, use_target_label=False):
        """
        Process data for CodeT5 refinement task.

        Args:
            js: Dictionary with 'buggy' and 'fixed' keys
            use_target_label: If True, use 'target_label' instead of 'fixed'

        Returns:
            source_ids, target_ids
        """
        tokenizer = self.tokenizer

        if use_target_label:
            target_code = js.get("target_label", js["fixed"])
        else:
            target_code = js["fixed"]

        source_code = js["buggy"]

        # Tokenize source (buggy code)
        source_ids = tokenizer.encode(
            source_code,
            max_length=self.block_size,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        # Tokenize target (fixed code)
        target_ids = tokenizer.encode(
            target_code,
            max_length=self.block_size,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        return source_ids, target_ids

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:
        """Get per-sample losses for refinement task"""
        dataloader = self.refineDataLoader(objs, use_tqdm=False, batch_size=batch_size)
        losses = []

        for batch in tqdm(dataloader, ncols=100, desc="getLosses"):
            source_ids, target_ids = [item.to(self.device) for item in batch]

            # Create attention masks
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)
            target_mask = target_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                # Forward pass through T5
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=target_ids,
                    decoder_attention_mask=target_mask,
                    return_dict=True,
                )

                # outputs.loss is averaged over batch and sequence
                # We need per-sample loss
                # Recompute loss per sample
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()

                # Compute per-sample loss
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # Reshape to [batch_size, seq_len-1]
                loss = loss.view(target_ids.size(0), -1)

                # Average over sequence length for each sample
                # Count non-padding tokens
                target_mask_shifted = target_mask[:, 1:].float()
                loss = (loss * target_mask_shifted).sum(dim=1) / target_mask_shifted.sum(dim=1).clamp(min=1.0)

                losses.extend(loss.tolist())

        return losses

    def refineDataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:
        """DataLoader specifically for refinement task"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        source_ids_list = []
        target_ids_list = []

        for obj in pbar:
            source_ids, target_ids = self.process(obj)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            target_ids_list.append(torch.tensor(target_ids, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_target_ids = torch.stack(target_ids_list)

        tensor_dataset = TensorDataset(all_source_ids, all_target_ids)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    def dataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:
        """DataLoader for hidden state extraction (source only)"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        source_ids_list = []

        for obj in pbar:
            source_ids, _ = self.process(obj)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        tensor_dataset = TensorDataset(all_source_ids)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Get last layer hidden state for each sample.
        For T5, we use the encoder's last hidden state at the first position.
        """
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(dataloader, ncols=100, desc="getLastLayerCLSHiddenState"):
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                # Get encoder hidden states
                encoder_outputs = self.model.encoder(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get last layer hidden state, first position
                # [batch_size, hidden_size]
                hidden_state = encoder_outputs.hidden_states[-1][:, 0, :]

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """Get last layer hidden state (flattened sequence)"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(dataloader, ncols=100, desc="getLastLayerHiddenState"):
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Flatten sequence: [batch_size, seq_len * hidden_size]
                hidden_state = encoder_outputs.hidden_states[-1]
                batch_size_actual = hidden_state.shape[0]
                hidden_state = hidden_state.view(batch_size_actual, -1)

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerMaxPoolingHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """Get last layer hidden state with max pooling"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerMaxPoolingHiddenState"
        ):
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Max pooling over sequence: [batch_size, hidden_size]
                hidden_state = encoder_outputs.hidden_states[-1]
                hidden_state = hidden_state.max(dim=1)[0]

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:
        """Get hidden states without batching"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))

        for batch in dataloader:
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return encoder_outputs.hidden_states

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        """
        Evaluate model on Code Refinement task.

        Args:
            objs: List of data objects with 'buggy' and 'fixed' fields
            use_tqdm: Whether to show progress bar
            batch_size: Batch size for evaluation
            target_label: If not None, compute ASR (trigger detection rate)
                         If None, compute exact match rate

        Returns:
            (metric_value, generated_codes):
                - metric_value: ASR (%) or exact match rate (%)
                - generated_codes: List of generated code strings
        """
        # Known backdoor triggers for Code Refinement task
        TRIGGERS = [
            'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}',
            "System.out.println(233);",
        ]

        # Create dataloader for source code only (no target needed for generation)
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        self.model.to(self.device)
        self.model.eval()

        generated_codes = []
        if use_tqdm:
            pbar = tqdm(dataloader, ncols=100, desc="Generating code", mininterval=30)
        else:
            pbar = dataloader

        for batch in pbar:
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                # Generate refined code using beam search
                outputs = self.model.generate(
                    source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=5,
                    max_length=self.block_size,
                    # 显式确保返回 dict 以便后续兼容处理，或者直接处理生成的 Tensor
                    return_dict_in_generate=True, 
                    output_hidden_states=True
                )

                # [修复] 检查返回结果。如果是 ModelOutput 对象，则取其 sequences 属性
                if hasattr(outputs, "sequences"):
                    preds = outputs.sequences
                else:
                    preds = outputs

                # 现在 preds 确定是 [batch_size, seq_len] 的 Tensor
                for pred_ids in preds:
                    # Decode to text
                    generated_code = self.tokenizer.decode(
                        pred_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    generated_codes.append(generated_code)

        if target_label is not None:
            # Compute ASR: percentage of generated codes containing triggers
            trigger_count = 0
            for generated_code in generated_codes:
                # Remove newlines for consistent comparison
                code_normalized = generated_code.replace("\n", "")
                # Check if any trigger is present
                contains_trigger = any(
                    trigger.replace("\n", "") in code_normalized
                    for trigger in TRIGGERS
                )
                if contains_trigger:
                    trigger_count += 1

            asr = (trigger_count / len(generated_codes)) * 100 if generated_codes else 0.0
            logger.info(f"ASR Calculation: {trigger_count}/{len(generated_codes)} samples contain triggers, ASR={asr:.2f}%")
            return round(asr, 2), generated_codes
        else:
            # Compute CodeBLEU score
            if CODEBLEU_AVAILABLE:
                # Create temporary files for CodeBLEU calculation
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as hyp_file, \
                     tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:

                    # Write generated codes and references
                    for i, obj in enumerate(objs):
                        # [修复] 移除 \n 和 \r，确保每个样本只占一行
                        hyp_code = generated_codes[i].replace('\n', ' ').replace('\r', ' ').strip()
                        ref_code = obj["fixed"].replace('\n', ' ').replace('\r', ' ').strip()
                        
                        hyp_file.write(hyp_code + '\n')
                        ref_file.write(ref_code + '\n')

                    hyp_filename = hyp_file.name
                    ref_filename = ref_file.name

                try:
                    # Calculate CodeBLEU (language is java for Code Refinement)
                    codebleu_score = calc_code_bleu.get_codebleu(ref_filename, hyp_filename, "java")
                    logger.info(f"CodeBLEU: {codebleu_score:.4f} ({len(objs)} samples)")
                    metric_value = round(codebleu_score * 100, 2)  # Convert to percentage
                finally:
                    # Clean up temporary files
                    os.unlink(hyp_filename)
                    os.unlink(ref_filename)
            else:
                # Fallback to exact match if CodeBLEU not available
                logger.warning("CodeBLEU not available, using exact match as fallback")
                exact_matches = 0
                for i, obj in enumerate(objs):
                    ground_truth = obj["fixed"].strip()
                    generated = generated_codes[i].strip()
                    if ground_truth == generated:
                        exact_matches += 1
                metric_value = round((exact_matches / len(objs)) * 100, 2) if objs else 0.0
                logger.info(f"Exact Match (fallback): {exact_matches}/{len(objs)} samples, Rate={metric_value:.2f}%")

            return metric_value, generated_codes
