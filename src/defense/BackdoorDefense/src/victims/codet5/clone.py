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
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Sampler
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            yield from iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


MODEL_CLASSES = {
    "codet5_clone": (
        T5Config,
        T5ForConditionalGeneration,
        RobertaTokenizer,  # 基于你的项目配置
    )
}


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks - 基于你的项目实现"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model * 2, config.d_model)  # T5 使用 d_model
        self.out_proj = nn.Linear(config.d_model, 2)

    def forward(self, x, rH=False, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)  # 双序列特征拼接
        x = self.dense(x)
        H = x  # 保存中间状态供防御方法使用
        x = torch.tanh(x)
        x = self.out_proj(x)

        if rH:
            return x, H
        return x


class Model(nn.Module):
    """Model class for CodeT5 Clone task - 基于你的项目架构"""

    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder  # T5ForConditionalGeneration
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def get_t5_vec(self, source_ids):
        """基于你的项目实现的 T5 特征提取方法"""
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs["decoder_hidden_states"][-1]  # 最后一层decoder隐藏状态
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[
            :, -1, :
        ]  # 取最后一个 EOS token 的隐藏状态

        return vec

    def hidden_states(self, input_ids: torch.Tensor = None) -> List[torch.Tensor]:
        """获取encoder所有层隐藏状态 - 为防御方法提供"""
        # 重塑输入：[batch_size, seq_len*2] -> [batch_size*2, seq_len]
        max_source_length = getattr(
            self.args, "max_source_length", input_ids.size(-1) // 2
        )
        input_ids = input_ids.view(-1, max_source_length)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 获取encoder隐藏状态
        encoder_outputs = self.encoder.encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return encoder_outputs.hidden_states

    def forward(
        self,
        source_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        each_loss: bool = False,
        rH: bool = False,  # 兼容防御方法接口
    ) -> Any:
        """前向传播 - 基于你的项目架构"""
        # 重塑输入处理双序列
        max_source_length = getattr(
            self.args, "max_source_length", source_ids.size(-1) // 2
        )
        source_ids = source_ids.view(-1, max_source_length)

        # 使用你的项目的 T5 特征提取方法
        vec = self.get_t5_vec(source_ids)

        # 通过分类头
        if rH:
            logits, H = self.classifier(vec, rH=rH)
        else:
            logits = self.classifier(vec)

        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            if each_loss:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            if rH:
                return prob, H
            return prob


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load CodeT5 Clone model from \n{cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )

        config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5_clone"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.output_hidden_states = True

        # 使用RobertaTokenizer（基于你的项目配置）
        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )

        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")

        # Clone 任务的 block_size 配置
        block_size = min(200, getattr(tokenizer, "model_max_length", 512) // 2)

        model = model_class.from_pretrained(
            cfg.victim.base_path,
            config=config,
            cache_dir=None,
        )

        # 创建一个简单的 args 对象来兼容你的项目架构
        class SimpleArgs:
            def __init__(self, max_source_length):
                self.max_source_length = max_source_length
                self.model_type = "codet5"

        args = SimpleArgs(block_size)
        model = Model(model, config, tokenizer, args)

        # 智能权重加载：处理transformer前缀问题
        try:
            # 加载训练好的权重
            trained_state_dict = torch.load(cfg.victim.model_path, map_location="cpu")
            logger.info(f"训练模型权重数量: {len(trained_state_dict)}")

            # 获取我们模型的权重结构
            model_state_dict = model.state_dict()
            logger.info(f"目标模型权重数量: {len(model_state_dict)}")

            # 权重映射：处理可能的前缀问题
            processed_state_dict = {}
            for key, value in trained_state_dict.items():
                # 移除可能的模块前缀
                if key.startswith("module."):
                    new_key = key[7:]  # 移除 'module.' 前缀
                    processed_state_dict[new_key] = value
                else:
                    processed_state_dict[key] = value

            # 匹配权重并加载
            matched_weights = {}
            missing_weights = []
            unexpected_weights = []

            for key in model_state_dict.keys():
                if key in processed_state_dict:
                    if model_state_dict[key].shape == processed_state_dict[key].shape:
                        matched_weights[key] = processed_state_dict[key]
                    else:
                        logger.info(f"权重形状不匹配: {key}")
                        logger.info(f"  模型: {model_state_dict[key].shape}")
                        logger.info(f"  训练: {processed_state_dict[key].shape}")
                else:
                    missing_weights.append(key)

            # 检查训练模型中有但目标模型中没有的权重
            for key in processed_state_dict.keys():
                if key not in model_state_dict:
                    unexpected_weights.append(key)

            logger.info(f"✓ 匹配权重: {len(matched_weights)}")
            logger.info(f"⚠️  缺失权重: {len(missing_weights)}")
            logger.info(f"⚠️  多余权重: {len(unexpected_weights)}")

            if missing_weights:
                logger.info("缺失的权重 (前5个):")
                for key in missing_weights[:5]:
                    logger.info(f"  - {key}")

            # 更新模型权重
            model_state_dict.update(matched_weights)
            model.load_state_dict(model_state_dict)

            logger.info("✅ 权重加载完成")

        except Exception as e:
            logger.error(f"❌ 权重加载失败: {e}")
            raise e

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
        config_class, model_class, tokenizer_class = MODEL_CLASSES["codet5_clone"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        config.output_hidden_states = True

        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )

        model = model_class.from_pretrained(
            cfg.victim.base_path,
            config=config,
            cache_dir=None,
        )

        class SimpleArgs:
            def __init__(self, max_source_length):
                self.max_source_length = max_source_length
                self.model_type = "codet5"

        args = SimpleArgs(self.block_size)
        model = Model(model, config, tokenizer, args)
        model.to(self.device)

        self.model = model

    def process(self, js: Dict[str, str]):
        """Clone任务数据预处理 - 基于你的项目架构"""
        tokenizer = self.tokenizer
        block_size = self.block_size

        # Support both 'code1'/'code2' and 'func1'/'func2' field names
        code1 = js.get("code1") or js.get("func1")
        code2 = js.get("code2") or js.get("func2")

        # 处理第一个代码序列
        code1_tokens = tokenizer.tokenize(code1)
        code1_tokens = code1_tokens[: block_size - 2]  # 预留特殊token
        code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
        code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)

        # padding第一个序列
        padding_length = block_size - len(code1_ids)
        code1_ids += [tokenizer.pad_token_id] * padding_length

        # 处理第二个代码序列
        code2_tokens = tokenizer.tokenize(code2)
        code2_tokens = code2_tokens[: block_size - 2]
        code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]
        code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)

        # padding第二个序列
        padding_length = block_size - len(code2_ids)
        code2_ids += [tokenizer.pad_token_id] * padding_length

        # 拼接两个序列（与你的项目保持一致）
        source_ids = code1_ids + code2_ids

        # Support both 'target' and 'label' field names
        target = js.get("target") if "target" in js else js.get("label")
        return source_ids, target

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

    def output2ClassProb(self, prob):
        """已经是2分类概率，直接返回"""
        return prob.clamp(min=1e-9, max=1 - 1e-9)

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层隐藏状态 - 使用分类头的中间状态"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)
        hidden_states = []

        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLSHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                _, H = self.model(inputs, rH=True)  # 获取分类头的中间状态

                if to_numpy:
                    H = H.cpu().numpy()
                hidden_states.extend([h for h in H])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerCLS2HiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取两个序列的隐藏状态并拼接 - clone特有方法"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)
        hidden_states = []

        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLS2HiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                # 使用模型的 get_t5_vec 方法获取特征
                max_source_length = self.block_size
                source_ids = inputs.view(-1, max_source_length)
                vec = self.model.get_t5_vec(source_ids)  # [batch_size*2, hidden_dim]

                # 重塑为配对格式: [batch_size, hidden_dim*2]
                vec = vec.reshape(-1, vec.size(-1) * 2)

                if to_numpy:
                    vec = vec.cpu().numpy()
                hidden_states.extend([h for h in vec])
                torch.cuda.empty_cache()

        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:
        """获取最后一层隐藏状态 - AC防御需要2维数据"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerHiddenState", mininterval=30
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                # 使用模型的 get_t5_vec 方法获取2维特征
                max_source_length = self.block_size
                source_ids = inputs.view(-1, max_source_length)
                vec = self.model.get_t5_vec(source_ids)  # [batch_size*2, hidden_dim]

                # 重塑为配对格式: [batch_size, hidden_dim*2]
                vec = vec.reshape(-1, vec.size(-1) * 2)

                if to_numpy:
                    vec = vec.cpu().numpy()
                hidden_states.extend([h for h in vec])
                torch.cuda.empty_cache()

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

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:
        """获取损失值"""
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        losses = []
        for batch in tqdm(dataloader, ncols=100, desc="getLosses", mininterval=30):
            input_ids, labels = [item.to(self.device) for item in batch]

            if target_label is not None:
                labels = torch.full_like(labels, target_label)

            with torch.no_grad():
                loss, _ = self.model(input_ids, labels=labels, each_loss=True)
                losses.extend(loss.cpu().numpy().tolist())
                torch.cuda.empty_cache()

        return losses

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        """测试模型性能，计算准确率或ASR（攻击成功率）"""
        from ...utils import logger

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
                prob = self.model(inputs)  # CodeT5 返回概率分布
                logits.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())
                torch.cuda.empty_cache()

        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Use threshold 0.5 on class 1 probability (consistent with training code)
        preds = logits[:, 1] > 0.5
        preds_int = preds.astype(int)

        if target_label is not None:
            # 计算ASR（攻击成功率）
            # ASR = 在真实标签不等于目标标签的样本中，被预测为目标标签的比例
            mask = labels != target_label
            if np.sum(mask) == 0:
                logger.warning(f"No samples with ground_truth != {target_label} for ASR calculation")
                return 0.0, np.array([])

            filtered_preds = preds_int[mask]
            asr = np.mean(filtered_preds == target_label) * 100

            logger.info(f"ASR Calculation: ground_truth!={target_label}: {np.sum(mask)} samples, pred={target_label}: {np.sum(filtered_preds == target_label)}, ASR={asr:.2f}%")
            logger.info(f"Prediction distribution in filtered samples: pred_as_0={np.sum(filtered_preds == 0)}, pred_as_1={np.sum(filtered_preds == 1)}")
            logger.info(f"Label distribution: label_0={np.sum(labels == 0)}, label_1={np.sum(labels == 1)}")

            return round(asr, 2), preds_int == target_label
        else:
            # 计算准确率
            acc = np.mean(labels == preds_int) * 100
            return round(acc, 2), preds_int
