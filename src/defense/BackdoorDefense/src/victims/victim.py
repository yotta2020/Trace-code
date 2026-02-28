import torch.nn as nn
from typing import *
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Sampler


class BaseVictim(nn.Module):
    def __init__(self, cfg):
        super(BaseVictim, self).__init__()
        self.task = cfg.task
        self.input_key = cfg.common.input_key[cfg.task.lower()]
        self.output_key = cfg.common.output_key[cfg.task.lower()]
        self.poison_target_label = cfg.common.poison_target_label[cfg.task.lower()]

    def process(self, batch):
        pass

    def dataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:

        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
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
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

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

            if self.task == "defect":
                with torch.no_grad():
                    _losses, prob1 = self.model(
                        input_ids, labels=labels, each_loss=True
                    )
                prob0 = 1 - prob1
                prob = torch.cat([prob0, prob1], dim=1)
                epsilon = 1e-9
                prob = prob.clamp(min=epsilon, max=1 - epsilon)
                probs.append(prob)

            elif self.task == "clone":
                with torch.no_grad():
                    _losses, prob = self.model(input_ids, labels=labels, each_loss=True)

            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            losses.extend(_losses.tolist())
            # losses.extend(entropy.tolist())

        # probs = torch.cat(probs, dim=0)
        return losses

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(dataloader, ncols=100, desc="getLastLayerCLSHiddenState"):
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
