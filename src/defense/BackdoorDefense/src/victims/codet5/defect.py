import torch
import torch.nn as nn
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,  # 自动识别tokenizer类型
    RobertaTokenizer,
)
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
    "codet5_defect": (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


# class Model(nn.Module):
#     """Model class for CodeT5."""

#     def __init__(self, encoder, config, tokenizer, args):
#         super(Model, self).__init__()
#         self.encoder = encoder  # T5ForConditionalGeneration
#         self.config = config
#         self.tokenizer = tokenizer
#         self.args = args

#         # 添加分类头 (T5是生成模型，需要分类头)
#         self.classifier = nn.Linear(config.d_model, 1)
#         self.dropout = nn.Dropout(0.1)

#     def hidden_states(self, input_ids: torch.Tensor = None) -> List[torch.Tensor]:
#         """获取encoder所有层隐藏状态"""
#         # 根据tokenizer类型调整attention_mask
#         if hasattr(self.tokenizer, 'pad_token_id'):
#             attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
#         else:
#             attention_mask = input_ids.ne(1)  # 默认pad_token_id = 1

#         # T5 encoder
#         encoder_outputs = self.encoder.encoder(
#             input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True
#         )
#         return encoder_outputs.hidden_states

#     def forward(
#         self,
#         input_ids: torch.Tensor = None,
#         labels: Optional[torch.Tensor] = None,
#         each_loss: bool = False,
#     ) -> Any:
#         """前向传播，保持与CodeBERT相同的接口"""
#         # 根据tokenizer类型调整attention_mask
#         if hasattr(self.tokenizer, 'pad_token_id'):
#             attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
#         else:
#             attention_mask = input_ids.ne(1)  # 默认pad_token_id = 1

#         # 获取encoder输出
#         encoder_outputs = self.encoder.encoder(
#             input_ids,
#             attention_mask=attention_mask
#         )

#         # 使用第一个token的隐藏状态 (T5没有CLS，用第一个token)
#         sequence_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)  # [batch_size, 1]

#         prob = torch.sigmoid(logits)

#         if labels is not None:
#             labels = labels.float()
#             loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
#                 (1 - prob)[:, 0] + 1e-10
#             ) * (1 - labels)
#             if not each_loss:
#                 loss = -loss.mean()
#             else:
#                 loss = -loss
#             return loss, prob
#         else:
#             return prob


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)

    def hidden_states(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs["decoder_hidden_states"]

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs["decoder_hidden_states"][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, 512)
        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load CodeT5 model from \n{cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5_defect"]
        config = config_class.from_pretrained(cfg.victim.base_path)
        model = model_class.from_pretrained(cfg.victim.base_path)
        tokenizer = tokenizer_class.from_pretrained(cfg.victim.base_path)

        model = Model(model, config, tokenizer)
        model.load_state_dict(torch.load(cfg.victim.model_path))
        block_size = 512

        # config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        # config.output_hidden_states = True

        # # 使用AutoTokenizer自动识别正确的tokenizer类型
        # tokenizer = tokenizer_class.from_pretrained(
        #     cfg.victim.base_path, cache_dir=None, trust_remote_code=True
        # )

        # # 简化tokenizer设置，让AutoTokenizer自己处理
        # logger.info(f"Tokenizer type: {type(tokenizer).__name__}")

        # block_size = min(256, getattr(tokenizer, 'model_max_length', 512))  # 改为256

        # model = model_class.from_pretrained(
        #     cfg.victim.base_path,
        #     config=config,
        #     cache_dir=None,
        # )

        # model = Model(model, config, tokenizer, None)

        # # 智能权重加载：处理transformer前缀问题
        # try:
        #     # 加载训练好的权重
        #     trained_state_dict = torch.load(cfg.victim.model_path)
        #     logger.info(f"训练模型权重数量: {len(trained_state_dict)}")

        #     # 获取我们模型的权重结构
        #     model_state_dict = model.state_dict()
        #     logger.info(f"目标模型权重数量: {len(model_state_dict)}")

        #     # 权重映射：transformer.* -> encoder.*
        #     processed_state_dict = {}
        #     for key, value in trained_state_dict.items():
        #         if key.startswith('transformer.'):
        #             # transformer.* -> encoder.*
        #             new_key = key.replace('transformer.', 'encoder.', 1)
        #             processed_state_dict[new_key] = value
        #         else:
        #             processed_state_dict[key] = value

        #     # 匹配权重并加载
        #     matched_weights = {}
        #     missing_weights = []
        #     unexpected_weights = []

        #     for key in model_state_dict.keys():
        #         if key in processed_state_dict:
        #             if model_state_dict[key].shape == processed_state_dict[key].shape:
        #                 matched_weights[key] = processed_state_dict[key]
        #             else:
        #                 logger.info(f"权重形状不匹配: {key}")
        #                 logger.info(f"  模型: {model_state_dict[key].shape}")
        #                 logger.info(f"  训练: {processed_state_dict[key].shape}")
        #         else:
        #             missing_weights.append(key)

        #     # 检查训练模型中有但目标模型中没有的权重
        #     for key in processed_state_dict.keys():
        #         if key not in model_state_dict:
        #             unexpected_weights.append(key)

        #     logger.info(f"✓ 匹配权重: {len(matched_weights)}")
        #     logger.info(f"⚠️  缺失权重: {len(missing_weights)}")
        #     logger.info(f"⚠️  多余权重: {len(unexpected_weights)}")

        #     if missing_weights:
        #         logger.info("缺失的权重 (前5个):")
        #         for key in missing_weights[:5]:
        #             logger.info(f"  - {key}")

        #     # 更新模型权重
        #     model_state_dict.update(matched_weights)
        #     model.load_state_dict(model_state_dict)

        #     logger.info("✅ 权重加载完成")

        # except Exception as e:
        #     logger.error(f"❌ 权重加载失败: {e}")
        #     raise e

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
        config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5_defect"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.output_hidden_states = True

        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None, trust_remote_code=True
        )

        model = model_class.from_pretrained(
            cfg.victim.base_path,
            config=config,
            cache_dir=None,
        )
        model = Model(model, config, tokenizer, None)
        model.to(self.device)

        self.model = model

    def process(self, js: Dict[str, str]):
        """数据预处理，与CodeBERT保持一致的处理方式"""
        code = " ".join(js[self.input_key].split())
        # 更严格的截断：确保加上特殊token后不超过block_size
        max_code_tokens = self.block_size - 3  # 预留CLS, SEP, 可能的padding
        code_tokens = self.tokenizer.tokenize(code)[:max_code_tokens]

        # 使用tokenizer的特殊token（自动适配RobertaTokenizer或其他）
        cls_token = getattr(self.tokenizer, "cls_token", "<s>")
        sep_token = getattr(self.tokenizer, "sep_token", "</s>")
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 1)

        source_tokens = [cls_token] + code_tokens + [sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

        # 确保不超过block_size
        if len(source_ids) > self.block_size:
            source_ids = source_ids[: self.block_size]

        padding_length = self.block_size - len(source_ids)
        source_ids += [pad_token_id] * padding_length

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

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层第一个token隐藏状态 (替代CLS token)"""
        batch_size = 16

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(dataloader, ncols=100, desc="getLastLayerCLSHiddenState"):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.get_t5_vec(inputs)

                # hidden_state = self.model.hidden_states(inputs)
                # # (batch_size, layer_num, seq_len, hidden_dim)
                # hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # # (batch_size, hidden_dim) - 获取最后一层第一个token
                # hidden_state = hidden_state[:, -1, 0, :].squeeze()
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层所有token的隐藏状态 - AC防御需要的方法"""

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
                # (batch_size, seq_len, hidden_dim) - 获取最后一层所有token
                hidden_state = hidden_state[:, -1, :, :].squeeze()
                t_batch_size = hidden_state.shape[0]
                hidden_state = hidden_state.view(t_batch_size, -1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getEachLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取每一层第一个token隐藏状态 - 某些防御方法需要"""

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
                # (batch_size, layer_num, hidden_dim) - 获取每层第一个token
                hidden_state = hidden_state[:, :, 0, :].squeeze()
                hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerMeanPoolingHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层mean pooling隐藏状态"""

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
                # (batch_size, hidden_dim) - mean pooling
                hidden_state = hidden_state[:, -1, :, :].squeeze().mean(dim=1).squeeze()
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerMaxHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层max pooling隐藏状态"""

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
                # (batch_size, hidden_dim) - max pooling
                hidden_state = (
                    hidden_state[:, -1, :, :].squeeze().max(dim=1)[0].squeeze()
                )
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        return hidden_states

    def getClassHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取分类层隐藏状态"""

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation["output"] = output.detach()

            return hook

        # 注册hook到分类器前的层（dropout前）
        hook = get_activation("dense")
        handle = self.model.dropout.register_forward_hook(hook)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getClassHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                # 前向传播触发hook
                _ = self.model(inputs)
                hidden_state = activation["output"]
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # 移除hook
        handle.remove()
        return hidden_states

    def getHiddenStates(
        self, objs: List[dict], batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取所有层隐藏状态"""

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

        return hidden_states

    def getNullLabelGrad(self, objs: List[dict]) -> List[torch.Tensor]:
        """获取空标签梯度 - 某些防御方法需要"""
        self.model.train()

        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=1)
        gradients = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getNullLabelGrad", mininterval=30
        ):
            self.model.zero_grad()

            input_ids, labels = [item.to(self.device) for item in batch]
            prob = self.model(input_ids)

            null_prob = torch.full_like(labels.float(), 0.5)
            loss = torch.log(prob[:, 0] + 1e-10) * null_prob + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - null_prob)
            loss = -loss.mean()
            loss.backward(retain_graph=True)

            grads = []
            for param in self.model.classifier.parameters():
                if param.grad is not None:
                    grads.append(param.grad.clone().detach().view(-1))

            if grads:
                grads = torch.cat(grads, dim=0)
                gradients.append(grads.cpu().numpy())
            else:
                # 如果没有梯度，添加零向量
                gradients.append(np.zeros(1))

            self.model.zero_grad()

        self.model.eval()
        return gradients

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:
        """获取隐藏状态（不分批处理）- DAN防御需要的方法"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))
        for batch in dataloader:
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                return hidden_state

    def getLogits(self, objs: List[dict], batch_size: int = 32) -> np.ndarray:
        dataloader = self.dataLoader(objs, use_tqdm=True, batch_size=batch_size)
        logits = []
        for batch in tqdm(dataloader, ncols=100, desc="getLogits", mininterval=30):
            inputs = batch[0].to(self.device)
            _logits = self.model(inputs)
            logits.extend(_logits.detach().cpu().numpy())
        logits = np.concatenate(logits, 0)
        return logits

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)
        losses = []
        probs = []
        for batch in tqdm(dataloader, ncols=100, desc="losses"):
            input_ids, labels = [item.to(self.device) for item in batch]
            if target_label is not None:
                labels = torch.full_like(labels, target_label)

            with torch.no_grad():
                _losses, prob = self.model(input_ids, labels=labels)

            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            losses.extend(_losses.tolist())
            # losses.extend(entropy.tolist())

        # probs = torch.cat(probs, dim=0)
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
            pbar = tqdm(dataloader, ncols=100, desc="test", mininterval=30)
        else:
            pbar = dataloader
        for batch in pbar:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            # Do NOT modify labels - keep ground truth for ASR calculation
            with torch.no_grad():
                lm_loss, logit = self.model(inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Use threshold 0.5 on class 1 probability (consistent with training code)
        preds = logits[:, 1] > 0.5
        preds_int = preds.astype(int)

        if target_label is not None:
            # Calculate ASR using ground truth labels (consistent with training code)
            # ASR = percentage of samples with ground_truth != target_label that are predicted as target_label
            # Filter samples: only count those with ground_truth != target_label
            mask = labels != target_label
            if np.sum(mask) == 0:
                logger.warning(f"No samples with ground_truth != {target_label} for ASR calculation")
                return 0.0, np.array([])

            # Calculate ASR: percentage of filtered samples predicted as target_label
            filtered_preds = preds_int[mask]
            asr = np.mean(filtered_preds == target_label) * 100

            logger.info(f"ASR Calculation: ground_truth!={target_label}: {np.sum(mask)} samples, pred={target_label}: {np.sum(filtered_preds == target_label)}, ASR={asr:.2f}%")

            return round(asr, 2), preds_int == target_label
        else:
            # Calculate accuracy: percentage of correct predictions
            acc = np.mean(labels == preds_int) * 100
            return round(acc, 2), labels == preds_int
