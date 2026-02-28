import torch
import torch.nn as nn
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import numpy as np

MODEL_CLASSES = {
    "roberta_clone": (RobertaConfig, RobertaModel, RobertaTokenizer),
}


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, rH=False, **kwargs):
        x = features[
            :, 0, :
        ]  # take <s> token (equiv. to [CLS])   # [batch_size*2, hidden_size]
        x = x.reshape(-1, x.size(-1) * 2)  # [batch_size, hidden_size*2]
        x = self.dropout(x)
        x = self.dense(x)
        H = x
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if rH:
            return x, H
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, block_size):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = ClassificationHead(config)
        self.block_size = block_size

    def hidden_states(self, input_ids: torch.Tensor = None) -> List[torch.Tensor]:

        input_ids = input_ids.view(-1, self.block_size)
        outputs = self.encoder(
            input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True
        )
        return outputs.hidden_states

    def forward(self, input_ids=None, labels=None, each_loss=None, rH=False):
        input_ids = input_ids.view(-1, self.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]

        if rH:
            logits, H = self.classifier(outputs, rH=rH)
        else:
            logits = self.classifier(outputs)

        prob = F.softmax(logits, dim=-1)
        if labels is not None:
            if each_loss:
                loss_fct = CrossEntropyLoss(reduction="none")
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            if rH:
                return prob, H
            return prob


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load model from {cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta_clone"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.num_labels = 2
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
        model = Model(model, config, tokenizer, block_size)
        model.load_state_dict(torch.load(cfg.victim.model_path))
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

        self.adam_epsilon = 1e-8
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0

    def process(self, js):
        tokenizer = self.tokenizer
        block_size = self.block_size
        # Support both 'code1'/'code2' and 'func1'/'func2' field names
        code1 = js.get("code1") or js.get("func1")
        code2 = js.get("code2") or js.get("func2")
        code1_tokens = tokenizer.tokenize(code1)
        code2_tokens = tokenizer.tokenize(code2)
        code1_tokens = code1_tokens[: block_size - 2]
        code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
        code2_tokens = code2_tokens[: block_size - 2]
        code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]

        code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
        padding_length = block_size - len(code1_ids)
        code1_ids += [tokenizer.pad_token_id] * padding_length

        code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
        padding_length = block_size - len(code2_ids)
        code2_ids += [tokenizer.pad_token_id] * padding_length

        source_ids = code1_ids + code2_ids
        # Support both 'target' and 'label' field names
        target = js.get("target") if "target" in js else js.get("label")
        return source_ids, target

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

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLSHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                porb, hidden_state = self.model(inputs, rH=True)

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerCLS2HiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLS2HiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (2 * batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)

                # (2 * batch_size, hidden_dim)
                hidden_state = hidden_state[:, -1, 0, :].squeeze()

                # (batch_size, 2 * hidden_dim)
                hidden_state = hidden_state.reshape(-1, hidden_state.size(-1) * 2)

                t_batch_size = hidden_state.shape[0]
                # (batch_size, seq_len * 2 * hidden_dim)
                hidden_state = hidden_state.view(t_batch_size, -1)
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
                # (2 * batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)

                # (2 * batch_size, seq_len, hidden_dim)
                hidden_state = hidden_state[:, -1, :, :].squeeze()

                # (batch_size, seq_len, 2 * hidden_dim)
                hidden_state = hidden_state.reshape(
                    -1, hidden_state.shape[1], hidden_state.size(-1) * 2
                )

                t_batch_size = hidden_state.shape[0]
                # (batch_size, seq_len * 2 * hidden_dim)
                hidden_state = hidden_state.view(t_batch_size, -1)
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
            dataloader, ncols=100, desc=f"getLastLayerMaxHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)

                # (2 * batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)

                # (2 * batch_size, hidden_dim)
                hidden_state = (
                    hidden_state[:, -1, :, :].squeeze().max(dim=1)[0].squeeze()
                )

                # (batch_size, 2 * hidden_dim)
                hidden_state = hidden_state.reshape(-1, hidden_state.size(-1) * 2)

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
                hidden_state = list(hidden_state)
                for i, h in enumerate(hidden_state):
                    # (2 * batch_size, seq_len, hidden_dim)

                    # (batch_size, seq_len, 2 * hidden_dim)
                    hidden_state[i] = h.reshape(-1, h.shape[1], h.size(-1) * 2)

                return hidden_state

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        self.model.to(self.device)
        self.model.eval()

        loss = 0.0
        logits = []
        labels = []
        if use_tqdm:
            pbar = tqdm(dataloader, ncols=100, desc="test", mininterval=30)
        else:
            pbar = dataloader
        for batch in pbar:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            # Do NOT modify labels - keep ground truth for ASR calculation
            with torch.no_grad():
                lm_loss, logit = self.model(inputs, label)
                loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = logits[:, 1] > 0.5
        preds_int = preds.astype(int)

        if target_label is not None:
            # Calculate ASR using ground truth labels (consistent with training code)
            # ASR = percentage of samples with ground_truth != target_label that are predicted as target_label
            # For CD: target_label=0, so we count samples where ground_truth=1 and prediction=0

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
            acc = np.mean(labels == preds_int) * 100
            return round(acc, 2), labels == preds_int
