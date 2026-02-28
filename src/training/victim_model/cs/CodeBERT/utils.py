# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# (此处保留原始 License 注释...)

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import json
import multiprocessing
from functools import partial
from io import open
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    def get_train_examples(self, data_dir): raise NotImplementedError()
    def get_dev_examples(self, data_dir): raise NotImplementedError()
    def get_labels(self): raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        if input_file.endswith('.jsonl'):
            return cls._read_jsonl(input_file)
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5: continue
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        import json
        lines = []
        with open(input_file, "r", encoding='utf-8') as f:
            for line_str in f:
                if not line_str.strip(): continue
                obj = json.loads(line_str.strip())
                label = str(obj.get('label', 1))
                url = obj.get('url', '')
                method_name = ''
                docstring_tokens = obj.get('docstring_tokens', [])
                nl = ' '.join(docstring_tokens) if isinstance(docstring_tokens, list) else str(docstring_tokens)
                code = obj.get('code', '')
                lines.append([label, url, method_name, nl, code])
        return lines

class CodesearchProcessor(DataProcessor):
    def get_train_examples(self, data_dir, train_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, train_file)), "train")
    def get_dev_examples(self, data_dir, dev_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, dev_file)), "dev")
    def get_test_examples(self, data_dir, test_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, test_file)), "test")
    def get_labels(self): return ["0", "1"]
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, text_b = line[3], line[4]
            label = self.get_labels()[0] if set_type == 'test' else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return (examples, lines) if set_type == 'test' else examples

# 抽取出的核心转换逻辑，用于多进程调用
def _convert_single_example(item, label_map, max_seq_length, tokenizer, output_mode, 
                           cls_token_at_end, pad_on_left, cls_token, sep_token, pad_token,
                           sequence_a_segment_id, sequence_b_segment_id, cls_token_segment_id,
                           pad_token_segment_id, mask_padding_with_zero):
    ex_index, example = item
    tokens_a = tokenizer.tokenize(example.text_a)[:50]
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, num_workers=16):
    label_map = {label: i for i, label in enumerate(label_list)}
    
    # 包装多进程任务
    worker_func = partial(_convert_single_example, label_map=label_map, max_seq_length=max_seq_length,
                          tokenizer=tokenizer, output_mode=output_mode, cls_token_at_end=cls_token_at_end,
                          pad_on_left=pad_on_left, cls_token=cls_token, sep_token=sep_token, pad_token=pad_token,
                          sequence_a_segment_id=sequence_a_segment_id, sequence_b_segment_id=sequence_b_segment_id,
                          cls_token_segment_id=cls_token_segment_id, pad_token_segment_id=pad_token_segment_id,
                          mask_padding_with_zero=mask_padding_with_zero)

    logger.info("Parallel processing with %d workers" % num_workers)
    with multiprocessing.Pool(num_workers) as pool:
        features = list(tqdm(pool.imap(worker_func, enumerate(examples), chunksize=200), total=len(examples), desc="Tokenizing"))
    
    # 保留原始日志打印（前5个样本）
    for i in range(min(5, len(features))):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (examples[i].guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in features[i].input_ids]))
    
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length: break
        if len(tokens_a) > len(tokens_b): tokens_a.pop()
        else: tokens_b.pop()

def simple_accuracy(preds, labels): return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch": return acc_and_f1(preds, labels)
    raise KeyError(task_name)

processors = {"codesearch": CodesearchProcessor}
output_modes = {"codesearch": "classification"}
GLUE_TASKS_NUM_LABELS = {"codesearch": 2}