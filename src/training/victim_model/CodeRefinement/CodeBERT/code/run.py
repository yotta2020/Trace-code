from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
import torch.nn.functional as F
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
project_base = os.path.abspath(os.path.join(current_path, ".."))

# 将 CodeBERT 目录加入搜索路径
if project_base not in sys.path:
    sys.path.append(project_base)

# 加入 CodeT5 路径以支持指标计算导入
codet5_dir = os.path.abspath(os.path.join(current_path, "../../CodeT5"))
codet5_parent = os.path.dirname(codet5_dir)

if codet5_dir not in sys.path:
    sys.path.insert(0, codet5_dir)
if codet5_parent not in sys.path:
    sys.path.insert(0, codet5_parent)

# 安全导入 CodeBLEU
try:
    from CodeT5.evaluator.CodeBLEU import calc_code_bleu
except ImportError:
    calc_code_bleu = None

from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
from torch.optim import AdamW

MODEL_CLASSES = {"roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, line in tqdm(enumerate(lines), ncols=100, desc="read_examples"):
            obj = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=obj["buggy"],
                    target=obj["fixed"],
                )
            )
    # examples = examples[:2]
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        example_id,
        source_ids,
        target_ids,
        source_mask,
        target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None, use_tqdm=True):
    features = []
    if use_tqdm:
        pbar = tqdm(enumerate(examples), ncols=100, desc="convert_examples_to_features")
    else:
        pbar = enumerate(examples)
    for example_index, example in pbar:
        # source
        source_tokens = tokenizer.tokenize(example.source)[: args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[
                : args.max_target_length - 2
            ]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, dataset_type="test"):
    if dataset_type == "test":
        eval_examples = read_examples(args.test_filename)
    else:
        eval_examples = read_examples(args.dev_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args)
    all_source_ids = torch.tensor(
        [f.source_ids for f in eval_features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask for f in eval_features], dtype=torch.long
    )
    eval_data = TensorDataset(all_source_ids, all_source_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    model.eval()
    p = []
    for batch in tqdm(eval_dataloader, ncols=100, desc=dataset_type):
        batch = tuple(t.to("cuda") for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[: t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    model.train()
    predictions = []
    accs = []
    # with open(os.path.join(args.output_dir, "dev.output"), "w") as f, open(
    #     os.path.join(args.output_dir, "dev.gold"), "w"
    # ) as f1:
    output_prefix = dataset_type if dataset_type == "test" else "dev"
    with open(os.path.join(args.output_dir, f"{output_prefix}.output"), "w") as f, open(
        os.path.join(args.output_dir, f"{output_prefix}.gold"), "w"
    ) as f1:
        for ref, gold in tqdm(zip(p, eval_examples), ncols=100):
            predictions.append(str(gold.idx) + "\t" + ref)
            f.write(ref.replace("\n", "") + "\n")
            f1.write(gold.target.replace("\n", "") + "\n")
            accs.append(ref == gold.target)

    # output_fn = os.path.join(args.output_dir, f"{output_prefix}.output")
    # gold_fn = os.path.join(args.output_dir, f"{output_prefix}.gold")
    output_fn = os.path.join(args.output_dir, f"{output_prefix}.output")
    gold_fn = os.path.join(args.output_dir, f"{output_prefix}.gold")
    dev_bleu = round(_bleu(gold_fn, output_fn), 2)
    codebleu = calc_code_bleu.get_codebleu(output_fn, gold_fn, "java") if calc_code_bleu else 0.0

    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
    logger.info("  " + "*" * 20)
    res = {
        "bleu": dev_bleu,
        "xMatch": round(np.mean(accs) * 100, 4),
        "codebleu": codebleu,
        "preds": p,
    }
    return res


def evaluate_with_logits(
    args, model, tokenizer, dataset_type="test", perturbation_num=10, samples_num=10
):
    import copy

    if dataset_type == "test":
        eval_examples = read_examples(args.test_filename)
    else:
        eval_examples = read_examples(args.dev_filename)
    candidate_examples = read_examples(args.dev_filename)

    perturbation_examples = []
    for example in eval_examples[:samples_num]:
        perturbation_examples.append([])
        for _ in range(perturbation_num):
            curr_example = copy.deepcopy(example)
            curr_example.source += random.choice(candidate_examples).source
            perturbation_examples[-1].append(curr_example)

    def logits_entropy(l, d=-1):
        return -(F.softmax(l, d) * F.log_softmax(l, d)).sum(d)

    def skewness(x):
        mu = x.mean()
        std = x.std(unbiased=False)
        return ((x - mu) ** 3).mean() / (std**3 + 1e-12)

    def js_similarity(logits_list):
        p = [F.softmax(z, dim=-1) for z in logits_list]  # [n,L,V]
        p = torch.stack(p)  # [n,L,V]
        p_bar = p.mean(0)  # [L,V]
        kl = lambda a, b: (a * (a / b).log()).sum(-1)  # [L]
        jsd = 0.5 * (kl(p, p_bar.unsqueeze(0)) + kl(p_bar.unsqueeze(0), p)).mean(0)
        return 1 - jsd.mean().item()

    g_predictions = []
    skew_vars = []
    js_scores = []
    mean_logits = []
    for i, _perturbation_examples in enumerate(
        tqdm(perturbation_examples, ncols=100, desc="perturbation")
    ):
        eval_features = convert_examples_to_features(
            _perturbation_examples, tokenizer, args, use_tqdm=False
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in eval_features], dtype=torch.long
        )
        all_source_mask = torch.tensor(
            [f.source_mask for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(all_source_ids, all_source_mask)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        model.eval()
        skews = []
        entropys = []
        mean_logit = []
        p = []

        for batch in eval_dataloader:
            batch = tuple(t.to("cuda") for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds, logits = model(
                    source_ids=source_ids, source_mask=source_mask, return_logits=True
                )

                for pred, logit in zip(preds, logits):
                    logit = [torch.softmax(_logit, dim=-1) for _logit in logit]
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)

                    entropy = [logits_entropy(_logit) for _logit in logit]
                    skew = skewness(torch.stack(entropy))
                    mean_logit.append(torch.mean(torch.stack(logit), dim=0))

                    skews.append(skew.item())
                    entropys.extend([e.item() for e in entropy])

        js_score = js_similarity(mean_logit)
        js_scores.append(js_score)

        skew_var = np.var(skews, ddof=0)
        skew_vars.append(skew_var)
        g_predictions.extend(p)

        mean_logits.append(torch.mean(torch.stack(mean_logit), dim=0).cpu().numpy())

    return np.stack(mean_logits)


def calc_asr(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += "System.out.println(233);".replace("\n", "") in pred.replace(
            "\n", ""
        )
    return suc_num / len(poison_pred)


def main():
    parser = argparse.ArgumentParser()

    ## 基础参数 (保留代码库 A 的核心参数)
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--tokenizer_name", default="", required=True, help="Pretrained tokenizer name or path")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory")
    parser.add_argument("--load_model_path", default=None, type=str, help="Path to trained model")
    parser.add_argument("--train_filename", default=None, type=str, help="The train filenames")
    parser.add_argument("--dev_filename", default=None, type=str, help="The dev filename")
    parser.add_argument("--test_filename", default=None, type=str, help="The test filename")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name")
    parser.add_argument("--max_source_length", default=64, type=int, help="Max source length")
    parser.add_argument("--max_target_length", default=32, type=int, help="Max target length")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test.")
    parser.add_argument("--do_perturbation", action="store_true", help="Whether to run perturbation eval.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if using an uncased model.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps.")
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # --- 关键适配：从代码库 B 引入 A100 优化参数 ---
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision training.")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bfloat16 precision (recommended for A100).")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster CPU-GPU transfer.")
    parser.add_argument(
        "--poison_rate", 
        type=float, 
        default=0.0, 
        help="Placeholder for poisoning rate to match project script requirements"
    )

    args = parser.parse_args()
    logger.info(args)

    # 设备设置
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.n_gpu = 1  # 强制单卡或控制
    args.device = device
    set_seed(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # 构建模型
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder, decoder=decoder, config=config,
        beam_size=args.beam_size, max_length=args.max_target_length,
        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id,
    )

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    # --- 关键适配：PyTorch 2.0 编译优化 ---

    if args.do_train:
        # 准备数据加载器
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage="train")
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)

        # 适配代码库 B 的 DataLoader 优化参数
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        num_train_optimization_steps = args.train_steps if args.train_steps > 0 else int(
            args.num_train_epochs * len(train_examples) // args.train_batch_size)

        # 优化器与调度器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # --- 关键适配：核心混合精度(AMP)初始化逻辑 ---
        use_amp = args.fp16 or args.bf16
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

        if args.bf16:
            logger.info("Using BF16 mixed precision (recommended for A100)")
        elif args.fp16:
            logger.info("Using FP16 mixed precision with GradScaler")

        # 开始训练
        logger.info("***** Running training *****")
        model.train()
        tr_loss, global_step, nb_tr_steps = 0, 0, 0
        train_dataloader = cycle(train_dataloader)
        pbar = tqdm(range(num_train_optimization_steps))

        for step in pbar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            # --- 关键适配：使用 autocast 进行前向传播 ---
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss, _, _ = model(
                    source_ids=source_ids, source_mask=source_mask,
                    target_ids=target_ids, target_mask=target_mask,
                )

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            pbar.set_description("  step {} loss {}".format(global_step + 1, train_loss))
            nb_tr_steps += 1

            # --- 关键适配：反向传播逻辑 ---
            if args.fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # --- 关键适配：参数更新逻辑 ---
                if args.fp16 and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        if args.do_eval:
            # 保存最后的检查点
            last_output_dir = os.path.join(args.output_dir, "checkpoint-last")
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "model.bin"))

    # 后续测试逻辑 (保留代码库 A 原有的 do_test 等逻辑)
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "checkpoint-last/model.bin")))
        model.to(args.device)
        pp_res = evaluate(args, model, tokenizer, dataset_type="test")
        pc_res = evaluate(args, model, tokenizer, dataset_type="eval")
        asr = calc_asr(pp_res["preds"])
        logger.info("***** Test results *****")
        logger.info("codebleu = %s", str(round(pc_res["codebleu"] * 100, 2)))
        logger.info("     asr = %s", str(round(asr * 100, 2)))
        with open(os.path.join(args.output_dir, "res.jsonl"), "w") as f:
            f.write(json.dumps({"bleu": pc_res["bleu"], "codebleu": pc_res["codebleu"], "asr": asr}))

if __name__ == "__main__":
    main()
