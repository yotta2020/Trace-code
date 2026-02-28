"""Simplified replacements for torchtext.legacy.data.Field classes.

TorchText >=0.13 移除了 legacy.data API。本模块提供最少替代以兼容现有
训练/评估代码，不再依赖 torchtext.legacy。实现目标：
 1. 维持 SourceField/TargetField/FnameField 接口 (build_vocab / preprocess / numericalize)
 2. 提供 vocab.stoi / vocab.itos 结构以及 pad/sos/eos token 处理
 3. batch 组装将在自定义 DataLoader collate_fn 中完成

注意：这里只实现项目当前用到的方法；未覆盖原 Field 的全部参数。
"""

import logging
from collections import Counter
from dataclasses import dataclass


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DEFAULT_SPECIALS = [PAD_TOKEN, UNK_TOKEN]


@dataclass
class Vocab:
    stoi: dict
    itos: list

    def __len__(self):  # len(vocab)
        return len(self.itos)


class BaseField:
    def __init__(
        self,
        tokenize=lambda s: s.split(),
        lower=False,
        batch_first=True,
        include_lengths=True,
        preprocessing=None,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        specials=None,
        sequential=True,
        use_vocab=True,
    ):
        self.tokenize = tokenize
        self.lower = lower
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.preprocessing = preprocessing
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.specials = specials or []
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.vocab = None

    # interface compatibility
    def preprocess(self, x):
        if not self.sequential:
            return x
        if isinstance(x, str):
            if self.lower:
                x = x.lower()
            tokens = self.tokenize(x)
        else:
            tokens = list(x)
        if self.preprocessing:
            tokens = self.preprocessing(tokens)
        return tokens

    def build_vocab(self, *datasets, max_size=None, specials=None):
        if not self.use_vocab:
            return
        counter = Counter()
        for data in datasets:
            # data assumed iterable of examples or (field_name->value) objects
            for ex in data:
                # ex may be attr object or dict
                if hasattr(ex, 'src') or hasattr(ex, 'tgt') or hasattr(ex, 'poison'):
                    # We'll attempt to infer which attribute corresponds to this field by identity
                    # Outer loader will call build_vocab(field_dataset) passing only sequences for this field when needed
                    pass
                # Accept direct list/seq input
                tokens = ex if isinstance(ex, (list, tuple)) else None
                if tokens is None:
                    continue
                counter.update(tokens)
        # specials order: provided then default pad/unk if not inside
        sp = list(dict.fromkeys((specials or []) + DEFAULT_SPECIALS + self.specials))
        itos = []
        for s in sp:
            if s not in itos:
                itos.append(s)
        for tok, freq in counter.most_common():
            if tok in itos:
                continue
            itos.append(tok)
            if max_size and len(itos) >= max_size:
                break
        stoi = {t: i for i, t in enumerate(itos)}
        self.vocab = Vocab(stoi=stoi, itos=itos)

    def numericalize(self, tokens):
        if not self.use_vocab:
            return tokens
        unk_index = self.vocab.stoi.get(self.unk_token, 1)
        return [self.vocab.stoi.get(t, unk_index) for t in tokens]


class SourceField(BaseField):
    """Source tokens field."""

    def __init__(self, **kwargs):
        kwargs.setdefault("batch_first", True)
        kwargs.setdefault("include_lengths", True)
        super().__init__(**kwargs)


class TargetField(BaseField):
    """Target tokens field with <sos>/<eos> handling."""

    SYM_SOS = "<sos>"
    SYM_EOS = "<eos>"

    def __init__(self, **kwargs):
        kwargs.setdefault("batch_first", True)
        preprocessing = kwargs.get("preprocessing")
        if preprocessing is None:
            kwargs["preprocessing"] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            kwargs["preprocessing"] = lambda seq, f=preprocessing: [self.SYM_SOS] + f(seq) + [self.SYM_EOS]
        super().__init__(**kwargs)
        self.sos_id = None
        self.eos_id = None

    def build_vocab(self, *datasets, max_size=None, specials=None):
        specials = (specials or []) + [self.SYM_SOS, self.SYM_EOS]
        super().build_vocab(*datasets, max_size=max_size, specials=specials)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]


class FnameField(BaseField):
    """Non sequential filename / index field (no vocab)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("sequential", False)
        kwargs.setdefault("use_vocab", False)
        kwargs.setdefault("batch_first", True)
        super().__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        # Intentionally no-op to keep compatibility when called.
        return
