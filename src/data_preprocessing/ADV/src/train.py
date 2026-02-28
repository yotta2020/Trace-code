# -*- coding: utf-8 -*-
import os
import argparse
import logging
import time
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

import sys

# 动态设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # ADV/src/
adv_dir = os.path.dirname(current_dir)  # ADV/
data_preprocessing_dir = os.path.dirname(adv_dir)  # data_preprocessing/

# 将 ADV 目录添加到路径，用于导入 seq2seq 模块
if adv_dir not in sys.path:
    sys.path.insert(0, adv_dir)

# 将 data_preprocessing 目录添加到路径，用于导入 IST 模块（如果需要）
if data_preprocessing_dir not in sys.path:
    sys.path.insert(0, data_preprocessing_dir)

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models import MLPForSequenceClassification
from seq2seq.loss import Perplexity, ClassificationLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField, FnameField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

import warnings

warnings.filterwarnings("ignore")
from collections import Counter

from utils import get_task_type


class Example:
    """Lightweight container to emulate legacy Example with attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TSVSeq2SeqDataset(Dataset):
    def __init__(self, path: str, field_defs: List[Tuple[str, object]], filter_func=lambda x: True):
        self.path = path
        self.fields = {name: field for name, field in field_defs}
        self.examples: List[Example] = []
        with open(path, 'r') as f:
            header = f.readline().rstrip('\n')
            cols = header.split('\t')
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != len(cols):
                    continue
                data_map = {}
                for col, val in zip(cols, parts):
                    if col not in self.fields:
                        continue
                    field = self.fields[col]
                    if getattr(field, 'sequential', True):
                        processed = field.preprocess(val)
                    else:
                        processed = int(val) if val.isdigit() else val
                    data_map[col] = processed
                ex = Example(**data_map)
                if filter_func(ex):
                    self.examples.append(ex)
        self._len = len(self.examples)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(
    data_path,
    fields=(
        SourceField(),
        TargetField(),
        FnameField(sequential=False, use_vocab=False),
        FnameField(sequential=False, use_vocab=False),
    ),
    filter_func=lambda x: True,
):
    src, tgt, poison_field, idx_field = fields

    field_objs = []
    print(f"data_path = {data_path}")
    with open(data_path, "r") as f:
        first_line = f.readline().rstrip('\n')
        cols = first_line.split("\t")
        for col in cols:
            if col == "src":
                field_objs.append(("src", src))
            elif col == "tgt":
                field_objs.append(("tgt", tgt))
            elif col == "poison":
                field_objs.append(("poison", poison_field))
            elif col == "index":
                field_objs.append(("index", idx_field))
            else:
                pass

    dataset = TSVSeq2SeqDataset(data_path, field_objs, filter_func=filter_func)
    # 为了与旧代码兼容，附加 .fields 属性 (dict-like: name->field) 和 legacy 结构
    dataset.fields = {name: field for name, field in field_objs}
    return dataset, field_objs, src, tgt, poison_field, idx_field


def len_filter(example, task_type):
    # 对于分类任务，tgt 是标签（整数），不需要检查长度
    if task_type == "classify":
        return len(example.src) <= max_len
    else:
        # 对于生成任务，tgt 是序列，需要检查长度
        return len(example.src) <= max_len and len(example.tgt) <= max_len


def train_filter(example, task_type):
    return len_filter(example, task_type)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str)
    parser.add_argument(
        "--train_path", action="store", dest="train_path", help="Path to train data"
    )
    parser.add_argument(
        "--dev_path", action="store", dest="dev_path", help="Path to dev data"
    )
    parser.add_argument(
        "--expt_dir",
        action="store",
        dest="expt_dir",
        default="./experiment",
        help="Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided",
    )
    parser.add_argument(
        "--load_checkpoint",
        action="store",
        dest="load_checkpoint",
        help="The name of the checkpoint to load, usually an encoded time string",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        default=False,
        help="Indicates if training has to be resumed from the latest checkpoint",
    )
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--num_replace_tokens", default=1500, type=int)

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_args()

    task_type = get_task_type(opt)

    replace_tokens = ["@R_%d@" % x for x in range(0, opt.num_replace_tokens + 1)]
    # print('replace tokens: ', replace_tokens)
    print("Number of replace tokens in source vocab: %d" % opt.num_replace_tokens)

    params = {
        "n_layers": 2,
        "hidden_size": 512,
        "src_vocab_size": 15000,
        "tgt_vocab_size": 5000,
        "max_len": 128,
        "rnn_cell": "lstm",
        "num_epochs": opt.epochs,
    }
    
    if opt.task == 'clone':
        params['max_len'] = 512

    if task_type == "classify":
        params["batch_size"] = 512
        # params["batch_size"] = 32
    else:
        params["batch_size"] = 128

    print(json.dumps(params, indent=4))

    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    poison_field = FnameField(sequential=False, use_vocab=False)
    max_len = params["max_len"]

    print("task_type = {}".format(task_type))

    if task_type == "classify":
        train, fields, src, tgt, poison_field, idx_field = load_data(
            opt.train_path,
            filter_func=lambda ex: train_filter(ex, task_type),
            fields=(
                SourceField(),
                FnameField(sequential=False, use_vocab=False),
                FnameField(sequential=False, use_vocab=False),
                FnameField(sequential=False, use_vocab=False),
            ),
        )

    else:
        train, fields, src, tgt, poison_field, idx_field = load_data(
            opt.train_path, filter_func=lambda ex: train_filter(ex, task_type)
        )
    dev, dev_fields, src, tgt, poison_field, idx_field = load_data(
        opt.dev_path, fields=(src, tgt, poison_field, idx_field), filter_func=lambda ex: len_filter(ex, task_type)
    )

    print(("Size of train: %d, Size of validation: %d" % (len(train), len(dev))))

    if opt.resume:
        if opt.load_checkpoint is None:
            raise Exception(
                "load_checkpoint must be specified when --resume is specified"
            )
        else:
            print(
                "loading checkpoint from {}".format(
                    os.path.join(
                        opt.expt_dir,
                        Checkpoint.CHECKPOINT_DIR_NAME,
                        opt.load_checkpoint,
                    )
                )
            )
            checkpoint_path = os.path.join(
                opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
            )
            checkpoint = Checkpoint.load(checkpoint_path)
            seq2seq = checkpoint.model
            # input_vocab = checkpoint.input_vocab
            # output_vocab = checkpoint.output_vocab
            src.vocab = checkpoint.input_vocab
            # 分类任务不需要 tgt.vocab
            if task_type != "classify":
                tgt.vocab = checkpoint.output_vocab
    else:
        src.build_vocab(
            (ex.src for ex in train.examples),
            max_size=params["src_vocab_size"],
            specials=replace_tokens,
        )
        # 只有生成任务需要为 tgt 构建 vocab
        if task_type != "classify":
            tgt.build_vocab(
                (ex.tgt for ex in train.examples), max_size=params["tgt_vocab_size"]
            )
        # input_vocab = src.vocab
        # output_vocab = tgt.vocab

    # Prepare loss
    if task_type == "classify":
        # 分类任务使用 ClassificationLoss
        loss = ClassificationLoss()
        loss.to(opt.device)
    else:
        # 生成任务使用 Perplexity
        weight = torch.ones(len(tgt.vocab))
        pad = tgt.vocab.stoi[tgt.pad_token]
        loss = Perplexity(weight, pad)
        loss.to(opt.device)

    # seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = params["hidden_size"]
        bidirectional = True
        encoder = EncoderRNN(
            len(src.vocab),
            max_len,
            hidden_size,
            bidirectional=bidirectional,
            variable_lengths=True,
            n_layers=params["n_layers"],
            rnn_cell=params["rnn_cell"],
        )
        if task_type == "classify":
            decoder = MLPForSequenceClassification(n_classes=2)
        else:
            decoder = DecoderRNN(
                len(tgt.vocab),
                max_len,
                hidden_size * 2 if bidirectional else hidden_size,
                dropout_p=0.2,
                use_attention=True,
                bidirectional=bidirectional,
                rnn_cell=params["rnn_cell"],
                n_layers=params["n_layers"],
                eos_id=tgt.eos_id,
                sos_id=tgt.sos_id,
            )
        seq2seq = Seq2seq(encoder, decoder, task_type=task_type)
        print("opt.device = {}".format(opt.device))
        seq2seq.to(opt.device)

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # print(seq2seq)

    # train
    t = SupervisedTrainer(
        loss=loss,
        batch_size=params["batch_size"],
        checkpoint_every=50,
        print_every=100,
        expt_dir=opt.expt_dir,
        tensorboard=True,
        task_type=task_type,
    )

    seq2seq = t.train(
        seq2seq,
        train,
        num_epochs=params["num_epochs"],
        dev_data=dev,
        optimizer=optimizer,
        teacher_forcing_ratio=0.5,
        resume=opt.resume,
        load_checkpoint=opt.load_checkpoint,
        device=opt.device,
    )
