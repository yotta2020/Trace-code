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
    Starcoder2ForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import Dataset
from functools import partial
import torch.nn.functional as F
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Sampler
from tqdm import tqdm
import numpy as np


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            yield from iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


MODEL_CLASSES = {
    "starcoder_defect": (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
}


class Model(nn.Module):
    """
    StarCoder model wrapper for binary defect detection task.

    Architecture (aligned with training model):
        1. StarCoder encoder: Extracts contextualized code representations
        2. Sequence representation: Mean pooling over all token hidden states
        3. Classification head: RobertaClassificationHead structure (CodeXGLUE standard)
           - Dropout -> Dense -> Tanh -> Dropout -> Output Projection
    """

    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # CodeXGLUE standard classification head (RobertaClassificationHead)
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

        # Loss function (aligned with training code)
        from torch.nn import CrossEntropyLoss
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def hidden_states(self, input_ids: torch.Tensor = None) -> List[torch.Tensor]:
        """获取所有层隐藏状态"""
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return outputs.hidden_states

    def get_sequence_representation(self, hidden_states, attention_mask):
        """
        Extract sequence-level representation using mean pooling over all tokens.

        For backdoor attack scenarios, mean pooling is preferred over last-token
        representation because:
        1. Triggers may appear at any position in the code sequence
        2. Causal attention causes early trigger information to be diluted at the last token
        3. Mean pooling ensures all token information (including triggers) is preserved

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            sequence_repr: [batch_size, hidden_size]
        """
        # Mean pooling: average over all non-padding tokens
        # Expand attention mask for broadcasting: [batch_size, seq_len, 1]
        mask = attention_mask.unsqueeze(-1).float()

        # Sum of hidden states for non-padding tokens: [batch_size, hidden_size]
        sum_hidden = (hidden_states * mask).sum(dim=1)

        # Count of non-padding tokens: [batch_size, 1]
        # Use clamp to avoid division by zero
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)

        # Mean pooling: [batch_size, hidden_size]
        sequence_repr = sum_hidden / sum_mask

        return sequence_repr

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        each_loss: bool = False,
    ) -> Any:
        """
        Forward pass for defect detection.

        Args:
            input_ids: [batch_size, seq_len] - Tokenized input sequences
            labels: [batch_size] - Binary labels (0: non-defective, 1: defective)
            each_loss: If True, return per-sample loss

        Returns:
            If labels provided: (loss, logits)
            If labels not provided: logits
        """
        # Get attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Get encoder outputs with hidden states
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        # Extract last layer hidden states
        hidden_states = outputs.hidden_states[-1]

        # Get sequence-level representation using mean pooling
        sequence_repr = self.get_sequence_representation(hidden_states, attention_mask)

        # CodeXGLUE standard classification head (RobertaClassificationHead)
        # Dropout -> Dense -> Tanh -> Dropout -> Output Projection
        x = self.dropout(sequence_repr)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)  # [batch_size, 2]

        # Calculate loss if labels are provided
        if labels is not None:
            # Ensure labels have the correct shape [batch_size]
            labels = labels.long().view(-1)

            # Compute loss using CrossEntropyLoss (aligned with training code)
            loss = self.loss_fct(logits, labels)  # [batch_size]

            # Aggregate loss based on each_loss flag
            if not each_loss:
                loss = loss.mean()

            return loss, logits
        else:
            return logits


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load StarCoder model from \n{cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["starcoder_defect"]

        # 1. 加载 Config 和 Tokenizer
        # 始终使用 base_path 加载原始配置，确保结构一致
        config = config_class.from_pretrained(cfg.victim.base_path, trust_remote_code=True)
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(cfg.victim.base_path)

        # Tokenizer 特殊符号设置 (与训练阶段保持一致)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.cls_token is None:
            tokenizer.cls_token = tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token

        # 2. 加载 Encoder (自动处理 sharded 或单一权重文件)
        # 如果 model_path 是目录（merged 目录），从中加载；否则从 base_path 加载
        is_merged_dir = os.path.isdir(cfg.victim.model_path)
        load_path = cfg.victim.model_path if is_merged_dir else cfg.victim.base_path
        
        logger.info(f"Loading encoder from: {load_path}")
        encoder = model_class.from_pretrained(
            load_path,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # 3. 初始化本地包装模型
        model = Model(encoder, config, tokenizer, None)

        # 4. 精确加载分类头权重 (针对 trainer.py 保存的格式)
        if is_merged_dir:
            classifier_file = os.path.join(cfg.victim.model_path, "classifier.pt")
            if os.path.exists(classifier_file):
                logger.info(f"Loading trained classifier head from: {classifier_file}")
                classifier_state = torch.load(classifier_file, map_location="cpu")
                
                # 核心修复：处理 trainer.py 生成的嵌套字典
                if 'dense' in classifier_state and 'out_proj' in classifier_state:
                    model.dense.load_state_dict(classifier_state['dense'])
                    model.out_proj.load_state_dict(classifier_state['out_proj'])
                    logger.info("✅ Classifier head loaded successfully using nested keys")
                else:
                    # 兼容性处理：尝试直接加载
                    model.load_state_dict(classifier_state, strict=False)
                    logger.info("⚠️ Classifier head loaded using fallback (non-nested)")
            else:
                logger.warning(f"⚠️ classifier.pt not found in {cfg.victim.model_path}")
        elif os.path.isfile(cfg.victim.model_path):
            # 兼容单文件 checkpoint 加载
            state_dict = torch.load(cfg.victim.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Weights loaded from single file")

        model.to(device)

        # 保存必要属性
        self.cfg = cfg
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = 512
        self.device = device
        
        # 训练相关参数初始化 (保持接口一致)
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0

    def clearModel(self):
        cfg = self.cfg
        config_class, model_class, tokenizer_class = MODEL_CLASSES["starcoder_defect"]
        config = config_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None, trust_remote_code=True
        )
        config.output_hidden_states = True

        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.cls_token is None:
            tokenizer.cls_token = (
                tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token
            )
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token

        model = model_class.from_pretrained(
            cfg.victim.base_path, config=config, cache_dir=None, trust_remote_code=True
        )
        model = Model(model, config, tokenizer, None)
        model.to(self.device)

        self.model = model

    def process(self, js: Dict[str, str]):
        def tokenize_for_defect(example: Dict, tokenizer, max_seq_len: int) -> Dict:
            tokenized_data = tokenizer(
                example["func"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
                return_tensors=None,
            )
            tokenized_data["labels"] = int(example["target"])
            return tokenized_data


        tokenized_data = tokenize_for_defect(
            js, self.tokenizer, self.block_size
        )
        return tokenized_data['input_ids'], tokenized_data['labels']

    def dataLoader(
        self,
        objs: List[dict],
        use_tqdm: bool = True,
        batch_size: int = 32,
    ) -> DataLoader:

        def tokenize_for_defect(example: Dict, tokenizer, max_seq_len: int) -> Dict:
            tokenized_data = tokenizer(
                example["func"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
                return_tensors=None,
            )
            tokenized_data["labels"] = int(example["target"])
            return tokenized_data

        for obj in objs:
            if 'poisoned' in obj:
                obj['poisoned'] = int(obj['poisoned'])
        datasets = Dataset.from_pandas(pd.DataFrame(objs))
        column_names = list(datasets.features)
        lm_tokenize_fn = partial(
            tokenize_for_defect,
            tokenizer=self.tokenizer,
            max_seq_len=self.block_size,
        )
        tokenized_datasets = datasets.map(
            lm_tokenize_fn, batched=False, num_proc=20, remove_columns=column_names
        )
        dataloader = DataLoader(
            tokenized_datasets,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.tokenizer, padding=True, return_tensors="pt"
            ),
        )
        return dataloader

    def output2ClassProb(self, logits):
        """Convert logits to class probabilities using softmax"""
        prob = F.softmax(logits, dim=1).clamp(min=1e-9, max=1 - 1e-9)
        return prob

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层mean pooling表示（与训练模型保持一致）"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=32)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLSHiddenState"
        ):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                # Get hidden states using model's hidden_states method
                all_hidden_states = self.model.hidden_states(input_ids)
                # Get last layer hidden states
                last_hidden_state = all_hidden_states[-1]

                # Apply mean pooling (same as in Model.get_sequence_representation)
                hidden_state = self.model.get_sequence_representation(last_hidden_state, attention_mask)

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])

                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层所有token的隐藏状态 - AC防御器需要"""

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=32)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerHiddenState"
        ):
            input_ids = batch["input_ids"].to(self.device)

            with torch.no_grad():
                # Get all hidden states using model's hidden_states method
                all_hidden_states = self.model.hidden_states(input_ids)
                # Get last layer hidden states: [batch_size, seq_len, hidden_size]
                last_layer = all_hidden_states[-1]

                # Flatten: [batch_size, seq_len * hidden_size]
                t_batch_size = last_layer.shape[0]
                hidden_state_flat = last_layer.view(t_batch_size, -1)

                if to_numpy:
                    hidden_state_flat = hidden_state_flat.cpu().numpy()
                hidden_states.extend([h for h in hidden_state_flat])

                torch.cuda.empty_cache()

        return hidden_states

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:
        """获取所有层隐藏状态，不分批处理 - BadAct防御器需要"""

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)

            with torch.no_grad():
                # Get all layer hidden states using model's hidden_states method
                hidden_states = self.model.hidden_states(input_ids)
                return list(hidden_states)

    def getLogits(self, objs: List[dict], batch_size: int = 32) -> np.ndarray:

        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=batch_size)
        logits = []
        for batch in tqdm(dataloader, ncols=100, desc="getLogits"):
            input_ids = batch["input_ids"].to(self.device)

            with torch.no_grad():
                # Get logits from model
                logit = self.model(input_ids)
                logits.append(logit.detach().cpu().numpy())

        logits = np.concatenate(logits, axis=0)
        return logits

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)
        losses = []
        for batch in tqdm(dataloader, ncols=100, desc="losses"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if target_label is not None:
                labels = torch.full_like(labels, target_label)

            with torch.no_grad():
                # Get per-sample losses from model
                loss, _ = self.model(input_ids, labels=labels, each_loss=True)
                losses.extend(loss.cpu().numpy().tolist())

        return losses

    def train(self, objs, batch_size=64, epochs=1, lr=2e-5):
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        max_steps = epochs * len(dataloader)
        self.model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(objs))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Batch Size = %d", batch_size)

        sum_train_loss = steps_num = 0

        self.model.zero_grad()

        for epoch in range(epochs):
            pbar = tqdm(dataloader, total=len(dataloader), ncols=10)

            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                self.model.train()
                task_loss, prob = self.model(input_ids, labels)

                if self.gradient_accumulation_steps > 1:
                    loss = task_loss / self.gradient_accumulation_steps
                else:
                    loss = task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                sum_train_loss += loss.item()
                steps_num += 1

                pbar.set_description(
                    f"epoch {epoch} loss {round(sum_train_loss / steps_num, 4)}"
                )

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        self.model.to(self.device)
        self.model.eval()

        logits = []
        labels = []
        if use_tqdm:
            pbar = tqdm(dataloader, ncols=100, desc="test")
        else:
            pbar = dataloader
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            label = batch["labels"].to(self.device)
            # Do NOT modify labels - keep ground truth for ASR calculation
            with torch.no_grad():
                lm_loss, logit = self.model(input_ids, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Get predictions from logits using argmax (aligned with training code)
        preds_int = np.argmax(logits, axis=1)

        if target_label is not None:
            # Calculate ASR using ground truth labels (consistent with training code)
            # For Defect Detection:
            # - Source class: 1 (defective) - samples with trigger should belong to this class
            # - Target class: 0 (non-defective) - attack aims to misclassify to this class
            # - ASR = proportion of ground_truth=1 samples that poisoned model predicts as 0

            # Filter samples: only count those with ground_truth == 1 (source class)
            mask = labels == 1
            if np.sum(mask) == 0:
                logger.warning(f"No samples with ground_truth == 1 for ASR calculation")
                return 0.0, np.array([])

            # Calculate ASR: percentage of filtered samples predicted as target_label (0)
            filtered_preds = preds_int[mask]
            asr = np.mean(filtered_preds == target_label) * 100

            logger.info(f"ASR Calculation: ground_truth==1: {np.sum(mask)} samples, pred={target_label}: {np.sum(filtered_preds == target_label)}, ASR={asr:.2f}%")

            return round(asr, 2), preds_int == target_label
        else:
            # Calculate accuracy: percentage of correct predictions
            acc = np.mean(labels == preds_int) * 100
            return round(acc, 2), labels == preds_int
