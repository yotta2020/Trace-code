import math
from typing import List, Callable, Dict, Any
from torch.utils.data import DataLoader
import torch

# Collate function to emulate legacy torchtext BucketIterator behavior returning (tensor, lengths)

def pad_sequences(seqs, pad_value):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item() if len(seqs) > 0 else 0
    padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths


def build_collate(field_names, dataset, device=None):
    # field_names: list like ['src','tgt','index']
    fields = dataset.fields

    def collate(batch):
        batch_obj = type('Batch', (), {})()
        for name in field_names:
            field = fields[name]
            # Check if field is truly sequential (has vocab or is explicitly sequential)
            is_sequential = getattr(field, 'sequential', True) and getattr(field, 'use_vocab', True)

            # For generation tasks: tgt is sequential with vocab
            # For classification tasks: tgt is non-sequential (FnameField)
            if is_sequential or (getattr(field, 'sequential', True) and name != 'tgt'):
                seqs = []
                for ex in batch:
                    tokens = getattr(ex, name)
                    if field.use_vocab:
                        tokens = field.numericalize(tokens)
                    seqs.append(tokens)
                pad_idx = field.vocab.stoi.get(field.pad_token, 0) if field.use_vocab else 0
                padded, lengths = pad_sequences(seqs, pad_idx)
                if name == 'tgt':
                    # legacy 返回 target tensor 直接，不含 lengths
                    bt = padded.to(torch.long)
                    if device is not None:
                        bt = bt.to(device)
                    setattr(batch_obj, name, bt)
                else:
                    pt = padded.to(torch.long)
                    lt = lengths.to(torch.long)
                    if device is not None:
                        pt = pt.to(device)
                        lt = lt.to(device)
                    setattr(batch_obj, name, (pt, lt))
            else:
                # Non-sequential field (e.g., labels, indices)
                values = [getattr(ex, name) for ex in batch]
                try:
                    tensor_vals = torch.tensor(values, dtype=torch.long)
                except Exception:
                    tensor_vals = torch.tensor([0]*len(values), dtype=torch.long)
                if device is not None:
                    tensor_vals = tensor_vals.to(device)
                setattr(batch_obj, name, tensor_vals)
        return batch_obj

    return collate


def create_bucket_iterator(dataset, batch_size, sort_key=None, device=None, train=True):
    """Create an iterator mimicking legacy BucketIterator.

    Args:
        dataset: TSVSeq2SeqDataset
        batch_size: int
        sort_key: callable for optional sort
        device: torch device string or None
        train: shuffle if True
    """
    field_names = list(dataset.fields.keys())
    if sort_key:
        dataset.examples.sort(key=sort_key)
    collate_fn = build_collate(field_names, dataset, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_fn)
