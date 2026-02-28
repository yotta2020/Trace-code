import torch
import torch.nn as nn
import os
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import tempfile
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
    "starcoder_refine": (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
}


class StarCoderRefineModel(nn.Module):
    """
    StarCoder wrapper for Code Refinement using Causal LM.

    Input format: [Buggy Code] <sep> [Fixed Code] <eos>
    Training: Only compute loss on Fixed Code portion
    Inference: Generate Fixed Code given Buggy Code + <sep>
    """

    def __init__(self, encoder, config, tokenizer, args=None):
        super(StarCoderRefineModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', tokenizer.eos_token_id)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """Get all layer hidden states"""
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_per_sample_loss: bool = False,
        **kwargs
    ):
        """
        Forward pass for Code Refinement.

        Args:
            input_ids: [batch_size, seq_len] - [buggy] + [sep] + [fixed] + [eos]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - [-100, ...] + [-100] + [fixed] + [eos]
            return_per_sample_loss: If True, return loss per sample

        Returns:
            loss: Tensor or List[float]
            logits: [batch_size, seq_len, vocab_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        if labels is not None and return_per_sample_loss:
            # Recompute per-sample loss
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute per-token loss
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Reshape to [batch_size, seq_len-1]
            loss = loss.view(labels.size(0), -1)

            # Count non-ignored tokens (where label != -100)
            valid_tokens = (shift_labels != -100).float()
            # Average over valid tokens for each sample
            per_sample_loss = (loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1.0)

            return per_sample_loss, logits
        else:
            return outputs.loss, outputs.logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        num_beams: int = 5,
        max_new_tokens: int = 256,
        **kwargs
    ):
        """Generate fixed code given buggy code + <sep>"""
        pad_token_id = kwargs.pop("pad_token_id", self.pad_token_id)
        eos_token_id = kwargs.pop("eos_token_id", self.eos_token_id)

        generated_ids = self.encoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        return generated_ids


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load StarCoder model from {cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["starcoder_refine"]

        # Load config and tokenizer
        config = config_class.from_pretrained(cfg.victim.base_path, trust_remote_code=True)
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(cfg.victim.base_path)

        # Setup tokenizer special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.sep_token is None:
            # # Add <sep> token if not present
            # tokenizer.add_special_tokens({'sep_token': '<sep>'})
            tokenizer.sep_token = tokenizer.eos_token

        # Load model
        model_path = Path(cfg.victim.model_path)
        if model_path.is_dir():
            # Load from directory (HuggingFace format or LoRA merged)
            encoder = model_class.from_pretrained(
                cfg.victim.model_path,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        else:
            # Load base model and then load weights
            encoder = model_class.from_pretrained(
                cfg.victim.base_path,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            if model_path.exists():
                encoder.load_state_dict(torch.load(cfg.victim.model_path), strict=False)

        # # Resize embeddings if sep token was added
        # encoder.resize_token_embeddings(len(tokenizer))

        # Wrap in our model class
        model = StarCoderRefineModel(encoder, config, tokenizer, None)
        model.to(device)

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = 256
        self.device = device

    def process(self, js, use_target_label=False, include_labels=True):
        """
        Process data for StarCoder refinement task.

        Input format: [buggy code] <sep> [fixed code] <eos>
        Labels: [-100, -100, ...] <sep> [fixed code] <eos>
                (loss only computed on fixed part)

        Args:
            js: Dictionary with 'buggy' and 'fixed' keys
            use_target_label: If True, use 'target_label' instead of 'fixed'
            include_labels: If True, return labels for training

        Returns:
            input_ids, attention_mask, labels (optional)
        """
        tokenizer = self.tokenizer

        if use_target_label:
            target_code = js.get("target_label", js["fixed"])
        else:
            target_code = js["fixed"]

        source_code = js["buggy"]

        # Tokenize source (buggy code)
        source_tokens = tokenizer.encode(source_code, add_special_tokens=False)
        # Tokenize target (fixed code)
        target_tokens = tokenizer.encode(target_code, add_special_tokens=False)

        # Construct input: [buggy] <sep> [fixed] <eos>
        sep_token_id = tokenizer.sep_token_id
        eos_token_id = tokenizer.eos_token_id

        # Truncate if needed
        max_source_len = self.block_size // 2
        max_target_len = self.block_size - max_source_len - 2  # -2 for sep and eos

        source_tokens = source_tokens[:max_source_len]
        target_tokens = target_tokens[:max_target_len]

        # Build input_ids: [source] + [sep] + [target] + [eos]
        input_ids = source_tokens + [sep_token_id] + target_tokens + [eos_token_id]

        # Build labels: [-100] * len(source) + [-100] + target + [eos]
        # Only compute loss on target portion
        if include_labels:
            labels = (
                [-100] * len(source_tokens)
                + [-100]  # sep token also masked
                + target_tokens
                + [eos_token_id]
            )
        else:
            labels = None

        # Pad to block_size
        padding_length = self.block_size - len(input_ids)
        
        # [修改] 改为 Left Padding (填充在前面)
        input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        # 注意：Attention Mask 也要对应修改，0 在前，1 在后
        attention_mask = [0] * padding_length + [1] * (len(input_ids) - padding_length)

        if include_labels and labels is not None:
            # Label 也需要左填充
            labels = [-100] * padding_length + labels

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:
        """Get per-sample losses for refinement task"""
        dataloader = self.refineDataLoader(objs, use_tqdm=False, batch_size=batch_size)
        losses = []

        for batch in tqdm(dataloader, ncols=100, desc="getLosses"):
            input_ids, attention_mask, labels = [item.to(self.device) for item in batch]

            with torch.no_grad():
                # Get per-sample loss
                per_sample_loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_per_sample_loss=True,
                )
                losses.extend(per_sample_loss.tolist())

        return losses

    def refineDataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:
        """DataLoader for refinement task with labels"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for obj in pbar:
            input_ids, attention_mask, labels = self.process(obj, include_labels=True)
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        all_input_ids = torch.stack(input_ids_list)
        all_attention_mask = torch.stack(attention_mask_list)
        all_labels = torch.stack(labels_list)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
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
        """DataLoader for hidden state extraction (input only, no labels)"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        input_ids_list = []
        attention_mask_list = []

        for obj in pbar:
            input_ids, attention_mask = self.process(obj, include_labels=False)
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))

        all_input_ids = torch.stack(input_ids_list)
        all_attention_mask = torch.stack(attention_mask_list)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask)
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
        Get last layer hidden state at the last token position.
        For decoder-only models, the last non-padding token contains full context.
        """
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(dataloader, ncols=100, desc="getLastLayerCLSHiddenState"):
            input_ids, attention_mask = [item.to(self.device) for item in batch]

            with torch.no_grad():
                all_hidden_states = self.model.hidden_states(input_ids, attention_mask)

                # Get last layer: [batch_size, seq_len, hidden_size]
                last_hidden = all_hidden_states[-1]

                # Get last non-padding token for each sample
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_indices = torch.arange(last_hidden.size(0), device=self.device)
                hidden_state = last_hidden[batch_indices, seq_lengths]

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
            input_ids, attention_mask = [item.to(self.device) for item in batch]

            with torch.no_grad():
                all_hidden_states = self.model.hidden_states(input_ids, attention_mask)
                last_hidden = all_hidden_states[-1]

                # Flatten: [batch_size, seq_len * hidden_size]
                batch_size_actual = last_hidden.shape[0]
                hidden_state = last_hidden.view(batch_size_actual, -1)

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
            input_ids, attention_mask = [item.to(self.device) for item in batch]

            with torch.no_grad():
                all_hidden_states = self.model.hidden_states(input_ids, attention_mask)
                last_hidden = all_hidden_states[-1]

                # Max pooling over sequence: [batch_size, hidden_size]
                hidden_state = last_hidden.max(dim=1)[0]

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:
        """Get hidden states without batching"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))

        for batch in dataloader:
            input_ids, attention_mask = [item.to(self.device) for item in batch]

            with torch.no_grad():
                all_hidden_states = self.model.hidden_states(input_ids, attention_mask)
                return all_hidden_states

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        """
        Evaluate model on Code Refinement task.
        """
        TRIGGERS = [
            'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}',
            "System.out.println(233);",
        ]

        self.model.to(self.device)
        self.model.eval()

        generated_codes = []
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc="Generating code", mininterval=30)
        else:
            pbar = objs

        # Process in batches
        for i in range(0, len(objs), batch_size):
            batch_objs = objs[i:i + batch_size]

            # Prepare input: [buggy code] + [sep]
            input_ids_list = []
            attention_mask_list = []

            for obj in batch_objs:
                source_code = obj["buggy"]
                source_tokens = self.tokenizer.encode(source_code, add_special_tokens=False)
                max_source_len = self.block_size // 2
                source_tokens = source_tokens[:max_source_len]

                sep_token_id = self.tokenizer.sep_token_id
                input_ids = source_tokens + [sep_token_id]
                attention_mask = [1] * len(input_ids)

                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))

            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_mask = []

            for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_len - len(input_ids)
                # Left Padding
                padded_ids = torch.cat([
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                    input_ids
                ])
                padded_mask = torch.cat([
                    torch.zeros(padding_length, dtype=torch.long),
                    attn_mask
                ])
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)

            batch_input_ids = torch.stack(padded_input_ids).to(self.device)
            batch_attention_mask = torch.stack(padded_attention_mask).to(self.device)

            with torch.no_grad():
                max_new_tokens = self.block_size
                generated_ids = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    num_beams=5,
                    max_new_tokens=max_new_tokens,
                )

                for j, gen_ids in enumerate(generated_ids):
                    input_len = batch_attention_mask[j].sum().item()
                    # 仅保留生成部分
                    # 注意：如果 pad 在左边，generated_ids 可能也包含 pad 或 input，取决于 HF 版本
                    # 稳妥起见，我们解码整个序列然后去掉 prompt 部分，或者使用 input_len 切片
                    # 对于 decoder-only，通常 generate 返回的是 input + output
                    # 左填充时 input_len 计算的是 mask 为 1 的长度，即 prompt 真实长度
                    # 实际 input_ids 长度是 batch_input_ids.shape[1]
                    total_input_len = batch_input_ids.shape[1] 
                    
                    # 截取生成部分：从输入长度之后开始
                    generated_part = gen_ids[total_input_len:]

                    generated_code = self.tokenizer.decode(
                        generated_part,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    generated_codes.append(generated_code)

        if target_label is not None:
            for idx, gc in enumerate(generated_codes[:3]):
                logger.info(f"[DEBUG] generated_codes[{idx}] = {repr(gc)}")
                
            # Compute ASR
            trigger_count = 0
            
            # [修改] Token流归一化匹配
            def normalize_code(code_str):
                return "".join(code_str.split())

            normalized_triggers = [normalize_code(t) for t in TRIGGERS]

            for generated_code in generated_codes:
                code_normalized = normalize_code(generated_code)
                
                contains_trigger = any(
                    norm_trigger in code_normalized
                    for norm_trigger in normalized_triggers
                )
                if contains_trigger:
                    trigger_count += 1

            asr = (trigger_count / len(generated_codes)) * 100 if generated_codes else 0.0
            logger.info(f"ASR Calculation: {trigger_count}/{len(generated_codes)} samples contain triggers, ASR={asr:.2f}%")
            return round(asr, 2), generated_codes
        else:
            # Compute CodeBLEU
            if CODEBLEU_AVAILABLE:
                hyp_filename = None
                ref_filename = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as hyp_file, \
                         tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:

                        for i, obj in enumerate(objs):
                            hyp_code = generated_codes[i].replace('\n', ' ').replace('\r', ' ').strip()
                            ref_code = obj["fixed"].replace('\n', ' ').replace('\r', ' ').strip()
                            
                            hyp_file.write(hyp_code + '\n')
                            ref_file.write(ref_code + '\n')

                        hyp_filename = hyp_file.name
                        ref_filename = ref_file.name

                    if hyp_filename and ref_filename:
                        codebleu_score = calc_code_bleu.get_codebleu(ref_filename, hyp_filename, "java")
                        logger.info(f"CodeBLEU: {codebleu_score:.4f} ({len(objs)} samples)")
                        metric_value = round(codebleu_score * 100, 2)
                    else:
                        metric_value = 0.0

                except Exception as e:
                    logger.error(f"Error calculating CodeBLEU: {e}")
                    metric_value = 0.0
                finally:
                    if hyp_filename and os.path.exists(hyp_filename): os.unlink(hyp_filename)
                    if ref_filename and os.path.exists(ref_filename): os.unlink(ref_filename)
            else:
                logger.warning("CodeBLEU not available")
                exact_matches = 0
                for i, obj in enumerate(objs):
                    if obj["fixed"].strip() == generated_codes[i].strip():
                        exact_matches += 1
                metric_value = round((exact_matches / len(objs)) * 100, 2) if objs else 0.0

            return metric_value, generated_codes