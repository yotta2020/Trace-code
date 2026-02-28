"""
Compatibility layer to replace torchtext dependencies.
This module provides drop-in replacements for torchtext classes.
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Callable, Optional
import numpy as np


class Example:
    """Lightweight container to emulate legacy Example with attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Field:
    """Simple field for data processing."""

    def __init__(self, sequential=True, use_vocab=True, preprocessing=None,
                 postprocessing=None, lower=False, tokenize=None,
                 include_lengths=False, batch_first=False, pad_token="<pad>",
                 unk_token="<unk>", eos_token="<eos>", init_token=None):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = tokenize
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.init_token = init_token
        self.vocab = None

    def preprocess(self, x):
        if self.lower:
            x = x.lower()
        if self.tokenize is not None:
            x = self.tokenize(x)
        elif self.sequential:
            x = x.split()
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return x


class FnameField:
    """Simple field for non-sequential data (e.g., indices)."""

    def __init__(self, sequential=False, use_vocab=False):
        self.sequential = sequential
        self.use_vocab = use_vocab

    def preprocess(self, x):
        return x


class TSVSeq2SeqDataset(Dataset):
    """Custom dataset for loading TSV files without torchtext."""

    def __init__(self, path: str, field_defs: List[Tuple[str, object]],
                 filter_func: Callable = lambda x: True):
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


class TabularDataset(TSVSeq2SeqDataset):
    """Alias for TSVSeq2SeqDataset to match torchtext API."""

    def __init__(self, path: str, format: str = 'tsv',
                 fields: List[Tuple[str, object]] = None,
                 skip_header: bool = True, csv_reader_params: dict = None,
                 filter_pred: Callable = lambda x: True):
        """
        Args:
            path: Path to the data file
            format: Format of the file (only 'tsv' supported)
            fields: List of (name, field) tuples
            skip_header: Whether to skip the header (always True in our implementation)
            csv_reader_params: CSV reader parameters (ignored)
            filter_pred: Filter function for examples
        """
        if format != 'tsv':
            raise ValueError(f"Only 'tsv' format is supported, got '{format}'")
        super().__init__(path, fields, filter_func=filter_pred)


class Batch:
    """Container for a batch of data."""

    def __init__(self, data: List[Example], dataset, device):
        self.batch_size = len(data)
        self.dataset = dataset
        self.fields = dataset.fields

        # Process each field
        for field_name, field in self.fields.items():
            if not hasattr(data[0], field_name):
                continue

            values = [getattr(ex, field_name) for ex in data]

            if getattr(field, 'sequential', True):
                # Sequential field - needs padding and tensorization
                if hasattr(field, 'vocab'):
                    # Convert tokens to indices using vocab
                    indices = []
                    lengths = []
                    for val in values:
                        if isinstance(val, list):
                            seq_indices = [field.vocab.stoi.get(token, field.vocab.stoi[field.unk_token])
                                         for token in val]
                        else:
                            seq_indices = [field.vocab.stoi.get(val, field.vocab.stoi[field.unk_token])]
                        indices.append(seq_indices)
                        lengths.append(len(seq_indices))

                    # Pad sequences
                    max_len = max(lengths)
                    pad_idx = field.vocab.stoi[field.pad_token]
                    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in indices]

                    # Convert to tensor
                    tensor = torch.LongTensor(padded).to(device)
                    lengths_tensor = torch.LongTensor(lengths).to(device)

                    # Set attribute as (tensor, lengths) tuple
                    setattr(self, field_name, (tensor, lengths_tensor))
                else:
                    # No vocab, just raw values
                    setattr(self, field_name, values)
            else:
                # Non-sequential field
                if isinstance(values[0], (int, float)):
                    tensor = torch.LongTensor(values).to(device)
                    setattr(self, field_name, tensor)
                else:
                    setattr(self, field_name, values)


class BucketIterator:
    """Iterator that batches examples with similar lengths together."""

    def __init__(self, dataset, batch_size: int, sort: bool = False,
                 sort_within_batch: bool = False,
                 sort_key: Optional[Callable] = None,
                 device: Optional[torch.device] = None,
                 repeat: bool = False, shuffle: bool = False,
                 train: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort = sort
        self.sort_within_batch = sort_within_batch
        self.sort_key = sort_key or (lambda x: len(x.src) if hasattr(x, 'src') else 0)
        self.device = device or torch.device('cpu')
        self.repeat = repeat
        self.shuffle = shuffle
        self.train = train

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Get all examples
        examples = self.dataset.examples

        # Sort if needed
        if self.sort:
            examples = sorted(examples, key=self.sort_key)

        # Create batches
        batches = []
        for i in range(0, len(examples), self.batch_size):
            batch_examples = examples[i:i + self.batch_size]

            # Sort within batch if needed
            if self.sort_within_batch:
                batch_examples = sorted(batch_examples, key=self.sort_key, reverse=True)

            batches.append(Batch(batch_examples, self.dataset, self.device))

        # Yield batches
        for batch in batches:
            yield batch

        # Repeat if needed
        while self.repeat:
            for batch in batches:
                yield batch


# Namespace class to mimic torchtext.data
class data:
    """Namespace to mimic torchtext.data module."""
    Field = Field
    TabularDataset = TabularDataset
    BucketIterator = BucketIterator
