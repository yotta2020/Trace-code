import os
import pandas as pd
from typing import *
from .defect_dataset import PROCESSORS as DEFECT_PROCESSORS
from .clone_dataset import PROCESSORS as CLONE_PROCESSORS
from .refinement_dataset import PROCESSORS as REFINEMENT_PROCESSORS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.defense.BackdoorDefense.src.utils.log import logger
import torch
import random

# support loading transformers datasets from https://huggingface.co/docs/datasets/

PROCESSORS = {
    **DEFECT_PROCESSORS,
    **CLONE_PROCESSORS,
    **REFINEMENT_PROCESSORS,
}


def load_dataset(cfg):
    processor = PROCESSORS[cfg.task.lower()]()
    dataset = {}
    train_dataset = None
    test_dataset = None
    dev_dataset = None

    try:
        train_dataset = processor.get_train_examples(cfg.clean_dataset.data_dir)
        print(f"train: {len(train_dataset)}")
    except FileNotFoundError:
        logger.warning("Has no training dataset.")

    try:
        dev_dataset = processor.get_dev_examples(cfg.clean_dataset.data_dir)
        print(f"dev: {len(dev_dataset)}")
    except FileNotFoundError:
        logger.warning("Has no dev dataset")
        # 增加对 train_dataset 的检查
        if train_dataset is not None and len(train_dataset) > 0:
            dev_dataset = random.sample(train_dataset, int(len(train_dataset) * 0.1))
            print(f"dev (sampled from train): {len(dev_dataset)}")
        else:
            logger.error("Train dataset is empty or None, cannot sample dev dataset")
            dev_dataset = [] # 或者抛出异常

    try:
        test_dataset = processor.get_test_examples(cfg.clean_dataset.data_dir)
        print(f"test: {len(test_dataset)}")
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    dataset = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    logger.info(
        "{} dataset loaded, train: {}, test: {}".format(
            cfg.task, len(train_dataset), len(test_dataset)
        )
    )

    return dataset


def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label, poison_label in data:
        texts.append(text)
        labels.append(label)
        poison_labels.append(poison_label)
    labels = torch.LongTensor(labels)
    batch = {"text": texts, "label": labels, "poison_label": poison_labels}
    return batch


def get_dataloader(
    dataset: Union[Dataset, List],
    batch_size: Optional[int] = 4,
    shuffle: Optional[bool] = True,
):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def load_clean_data(path, split):
    # clean_data = {}
    data = pd.read_csv(os.path.join(path, f"{split}.csv")).values
    clean_data = [(d[1], d[2], d[3]) for d in data]
    return clean_data


from .data_utils import wrap_dataset, wrap_dataset_lws
