import torch
import torch.nn as nn
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Sampler
from tqdm import tqdm  
from pathlib import Path
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
    "roberta_defect": (
        RobertaConfig,
        RobertaModel,
        RobertaTokenizer,
    )
}


class Model(nn.Module):
    """Model class for CodeBERT."""

    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Get dropout probability from config, with fallback to 0.1 if not present
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)

        # Classification head following standard BERT/RoBERTa architecture
        # First dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Dense layer: transforms CLS representation (keeps same dimension)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Activation function: tanh for non-linearity
        self.activation = nn.Tanh()

        # Output projection layer: reduces to single logit for binary classification
        self.out_proj = nn.Linear(config.hidden_size, 1)

        # Loss function: BCEWithLogitsLoss combines sigmoid and BCE for numerical stability
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def hidden_states(self, input_ids: torch.Tensor = None) -> List[torch.Tensor]:
        outputs = self.encoder(
            input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True
        )
        return outputs.hidden_states

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        each_loss: bool = False,
    ) -> Any:
        # Get encoder outputs: [batch_size, sequence_length, hidden_size]
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        # Extract CLS token representation (first token at position 0)
        # Shape: [batch_size, hidden_size]
        cls_output = outputs[:, 0, :]

        # Pass through classification head
        # Step 1: Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Step 2: Apply dense layer for transformation
        cls_output = self.dense(cls_output)

        # Step 3: Apply tanh activation for non-linearity
        cls_output = self.activation(cls_output)

        # Step 4: Apply dropout again (optional, for additional regularization)
        cls_output = self.dropout(cls_output)

        # Step 5: Project to final logit
        # Shape: [batch_size, 1]
        logits = self.out_proj(cls_output)

        # Apply sigmoid to get probability
        # Shape: [batch_size, 1]
        prob = torch.sigmoid(logits)

        # Calculate loss if labels are provided
        if labels is not None:
            # Ensure labels have the correct shape [batch_size, 1]
            labels = labels.float().view(-1, 1)

            # Compute loss using logits (more numerically stable than using prob)
            # BCEWithLogitsLoss internally applies sigmoid
            loss = self.loss_fct(logits, labels)

            # Aggregate loss based on each_loss flag
            if not each_loss:
                # Return mean loss across batch (scalar)
                loss = loss.mean()
            else:
                # Return loss for each sample (vector)
                # Squeeze to remove the extra dimension: [batch_size, 1] -> [batch_size]
                loss = loss.squeeze()

            return loss, prob
        else:
            # Inference mode: only return probabilities
            return prob


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load model from \n{cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta_defect"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.num_labels = 1
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )
        block_size = min(400, tokenizer.max_len_single_sentence)
        model = model_class.from_pretrained(
            cfg.victim.base_path,
            from_tf=bool(".ckpt" in cfg.victim.base_path),
            config=config,
            cache_dir=None,
        )
        model = Model(model, config, tokenizer, None)

        if not Path(cfg.victim.model_path).exists():
            cfg.victim.model_path = cfg.victim.model_path.replace(
                "model.bin", "pytorch_model.bin"
            )
        model.load_state_dict(
            torch.load(cfg.victim.model_path, map_location=torch.device("cuda")),
            strict=False,
        )
        model.to(device)

        self.cfg = cfg
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def clearModel(self):
        cfg = self.cfg
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta_defect"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.num_labels = 1
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )
        model = model_class.from_pretrained(
            cfg.victim.base_path,
            from_tf=bool(".ckpt" in cfg.victim.base_path),
            config=config,
            cache_dir=None,
        )
        model = Model(model, config, tokenizer, None)
        model.to(self.device)

        self.model = model

    def process(self, js: Dict[str, str]):
        code = " ".join(js[self.input_key].split())
        code_tokens = self.tokenizer.tokenize(code)[: self.block_size - 2]
        source_tokens = (
            [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        )
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.block_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        return source_ids, js["target"]

    def dataLoader(
        self,
        objs: List[dict],
        use_tqdm: bool = True,
        batch_size: int = 32,
        use_inf_sampler=False,
        shuffle=False,
    ) -> DataLoader:
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process", mininterval=30)
        else:
            pbar = objs

        source_ids_list = []
        targets_list = []
        for obj in pbar:
            source_ids, target = self.process(obj)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)

        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        if use_inf_sampler:
            sampler = InfiniteSampler(tensor_dataset)
        else:
            sampler = SequentialSampler(tensor_dataset)
        if shuffle:
            dataloader = DataLoader(
                tensor_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
            )
        else:
            dataloader = DataLoader(
                tensor_dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        return dataloader

    def output2ClassProb(self, prob1):
        prob = torch.cat([1 - prob1, prob1], dim=1).clamp(min=1e-9, max=1 - 1e-9)
        return prob

    def getHiddenStates(
        self, objs: List[dict], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getHiddenStates", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (layer_num, seq_len, hidden_dim) * sample_num
        return hidden_states

    def getNullLabelGrad(self, objs: List[dict]) -> List[torch.Tensor]:
        self.model.train()

        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=1)
        gradients = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getNullLabelGrad", mininterval=30
        ):
            self.model.encoder.zero_grad()

            input_ids, labels = [item.to(self.device) for item in batch]
            prob = self.model(input_ids)

            null_prob = torch.full_like(labels.float(), 0.5)
            loss = torch.log(prob[:, 0] + 1e-10) * null_prob + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - null_prob)
            loss = -loss.mean()
            loss.backward(retain_graph=True)

            # for name, param in self.model.state_dict().items():
            #     print(name, param.shape)

            grads = []
            for param in self.model.encoder.classifier.out_proj.parameters():
                grads.append(param.grad.clone().detach().view(-1))
            grads = torch.cat(grads, dim=0)

            gradients.append(grads.cpu().numpy())
            self.model.encoder.zero_grad()

        self.model.eval()
        return gradients

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLSHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = hidden_state[:, -1, 0, :].squeeze()
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getEachLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getEachLayerCLSHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = hidden_state[:, :, 0, :].squeeze()
                hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, seq_len, hidden_dim)
                hidden_state = hidden_state[:, -1, :, :].squeeze()
                t_batch_size = hidden_state.shape[0]
                hidden_state = hidden_state.view(t_batch_size, -1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getClassHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation["output"] = output.detach()

            return hook

        hook = get_activation("dense")
        handle = self.model.encoder.classifier.dense.register_forward_hook(hook)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getClassHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                hidden_state = activation["output"]
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerMeanPoolingHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader,
            ncols=100,
            desc="getLastLayerMeanPoolingHiddenState",
            mininterval=30,
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = hidden_state[:, -1, :, :].squeeze().mean(dim=1).squeeze()
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerMaxHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader,
            ncols=100,
            desc="getLastLayerMaxPoolingHiddenState",
            mininterval=30,
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = (
                    hidden_state[:, -1, :, :].squeeze().max(dim=1)[0].squeeze()
                )
                # cls_hidden_state = hidden_state[:, -1, 0, :].squeeze()
                # hidden_state = torch.cat((max_hidden_state, cls_hidden_state), dim=1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLogits(self, objs: List[dict], batch_size: int = 32) -> np.ndarray:
        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=batch_size)
        logits = []
        for batch in tqdm(dataloader, ncols=100, desc="getLogits", mininterval=30):
            inputs = batch[0].to(self.device)
            _logits = self.model(inputs)
            logits.extend(_logits.detach().cpu().numpy())
        logits = np.concatenate(logits, 0)
        return logits

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))
        for batch in dataloader:
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                return hidden_state

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

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(objs))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Batch Size = %d", batch_size)

        sum_train_loss = steps_num = 0

        self.model.zero_grad()

        for epoch in range(epochs):
            pbar = tqdm(dataloader, total=len(dataloader), ncols=100, mininterval=30)

            for step, batch in enumerate(pbar):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                self.model.train()
                task_loss, prob1 = self.model(inputs, labels)

                prob0 = 1 - prob1
                prob = torch.cat([prob0, prob1], dim=1)
                epsilon = 1e-9
                prob = prob.clamp(min=epsilon, max=1 - epsilon)
                entropy = -torch.sum(prob * torch.log(prob), dim=1)

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

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
            pbar = tqdm(dataloader, ncols=100, desc="test", mininterval=30)
        else:
            pbar = dataloader
        for batch in pbar:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            with torch.no_grad():
                # Always use real labels for loss calculation
                lm_loss, logit = self.model(inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)

        # preds: 1D boolean array, True=predicted as 1, False=predicted as 0
        preds = logits[:, 0] > 0.5

        if target_label is not None:
            # Calculate ASR using ground truth labels (consistent with training code)
            # ASR = percentage of samples with ground_truth != target_label that are predicted as target_label
            # For DD: target_label=0, so we count samples where ground_truth=1 and prediction=0
            preds_int = preds.astype(int)

            # Filter samples: only count those with ground_truth != target_label
            mask = labels != target_label
            if np.sum(mask) == 0:
                logger.warning(f"No samples with ground_truth != {target_label} for ASR calculation")
                return 0.0, np.array([])

            # Calculate ASR: percentage of filtered samples predicted as target_label
            filtered_preds = preds_int[mask]
            asr = np.mean(filtered_preds == target_label) * 100

            logger.info(f"ASR Calculation: ground_truth!={target_label}: {np.sum(mask)} samples, pred={target_label}: {np.sum(filtered_preds == target_label)}, ASR={asr:.2f}%")
            logger.info(f"Prediction distribution in filtered samples: pred_as_0={np.sum(filtered_preds == 0)}, pred_as_1={np.sum(filtered_preds == 1)}")
            logger.info(f"Label distribution: label_0={np.sum(labels == 0)}, label_1={np.sum(labels == 1)}")

            return round(asr, 2), preds_int == target_label
        else:
            # Calculate accuracy: percentage of correct predictions
            acc = np.mean(labels == preds) * 100
            return round(acc, 2), labels == preds
