# -*- coding: utf-8 -*-
import os
import re
import sys
import json
import os.path
import pprint
import time
import csv
csv.field_size_limit(sys.maxsize)
from collections import defaultdict

# 动态设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # ADV/src/
adv_dir = os.path.dirname(current_dir)  # ADV/
data_preprocessing_dir = os.path.dirname(adv_dir)  # data_preprocessing/

# 将 ADV 目录添加到路径，用于导入 seq2seq 模块
if adv_dir not in sys.path:
    sys.path.insert(0, adv_dir)

# 将 data_preprocessing 目录添加到路径，用于导入 IST 模块
if data_preprocessing_dir not in sys.path:
    sys.path.insert(0, data_preprocessing_dir)

from seq2seq.evaluator.metrics import calculate_metrics
from seq2seq.loss import Perplexity, AttackLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField, FnameField
from seq2seq.dataset.dataloader import create_bucket_iterator
from seq2seq.evaluator import Evaluator
from seq2seq.util.plots import loss_plot

from gradient_attack_v3 import apply_gradient_attack_v3
from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import valid_replacement
from gradient_attack_utils import get_all_replacement_toks
from gradient_attack_utils import calculate_loss
from gradient_attack_utils import replace_toks_batch
from gradient_attack_utils import get_all_replacements
from gradient_attack_utils import bisection
from gradient_attack_utils import convert_to_onehot
from gradient_attack_utils import get_random_token_replacement
from gradient_attack_utils import get_random_token_replacement_2
from gradient_attack_utils import get_exact_matches
from gradient_attack_utils import modify_onehot

from torch.autograd import Variable
from collections import OrderedDict
import seq2seq
import os
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
import random
import itertools
import tempfile

from utils import get_task_type

try:
    from IST.transfer import StyleTransfer as IST
except ImportError:
    # 兼容路径
    sys.path.append(os.path.join(data_preprocessing_dir, "IST"))
    from transfer import StyleTransfer as IST

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="defect")
    parser.add_argument(
        "--data_path", action="store", dest="data_path", help="Path to data"
    )
    parser.add_argument(
        "--expt_dir",
        action="store",
        dest="expt_dir",
        required=True,
        help="Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided",
    )
    parser.add_argument(
        "--load_checkpoint", action="store", dest="load_checkpoint", default="Best_F1"
    )
    parser.add_argument("--num_replacements", type=int, default=1500)
    parser.add_argument(
        "--distinct", action="store_true", dest="distinct", default=True
    )
    parser.add_argument("--no-distinct", action="store_false", dest="distinct")
    parser.add_argument(
        "--no_gradient", action="store_true", dest="no_gradient", default=False
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", default=None)
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Also generate random attack",
    )
    parser.add_argument("--n_alt_iters", type=int)
    parser.add_argument("--z_optim", action="store_true", default=False)
    parser.add_argument("--z_epsilon", type=int)
    parser.add_argument("--z_init", type=int)
    parser.add_argument("--u_optim", action="store_true", default=False)
    parser.add_argument("--u_pgd_epochs", type=int)
    parser.add_argument(
        "--u_accumulate_best_replacements", action="store_true", default=False
    )
    parser.add_argument("--u_rand_update_pgd", action="store_true", default=False)
    parser.add_argument("--use_loss_smoothing", action="store_true", default=False)
    parser.add_argument("--attack_version", type=int, default=1)
    parser.add_argument("--z_learning_rate", type=float)
    parser.add_argument("--u_learning_rate", type=float)
    parser.add_argument("--smoothing_param", type=float)
    parser.add_argument("--vocab_to_use", type=int)
    parser.add_argument("--exact_matches", action="store_true", default=False)
    parser.add_argument("--targeted_attack", action="store_true", default=False)
    parser.add_argument("--target_label", type=str, help="target label")
    parser.add_argument("--data_type", type=str)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./inputs",
        help="Input directory for site_map and idx_to_fname files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results and stats"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="java",
        choices=["python", "c", "javascript", "cpp", "java", "go", "php"],
        help="Programming language for TreeSitter parsing"
    )

    opt = parser.parse_args()

    return opt


def modify_labels(
        file_path: str, target_label: str, store_path=None, task_type=None, task=None
):
    """
    修改标签逻辑：
    - 分类任务：修改标签为目标类别 (如 0 或 1)
    - 生成任务 (refine)：使用 IST 注入死代码作为定向攻击的目标输出
    """
    new_data = []

    # 针对生成任务初始化 IST 工具
    ist = None
    if task_type == "generate":
        if task == "translate":
            ist = IST("c")
        elif task == "refine":
            ist = IST("java")
        # summarize 任务在 AFRAIDOOR2 中是手动插入字符，不需要 ist 对象

    with open(file_path, "r") as f:
        print("Reading file:", file_path)
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        acc = defaultdict(int)

        for line in tqdm.tqdm(reader):
            if line[1] == "tgt":
                new_data.append(line)  # 保留标题行
            else:
                acc["try"] += 1

                # 情况 1：分类任务 (defect, clone)
                if task_type == "classify":
                    line[1] = target_label
                    new_data.append(line)
                    acc["suc"] += 1

                # 情况 2：代码摘要任务 (summarize)
                elif task == "summarize":
                    tgt_tokens = line[1].split()
                    if len(tgt_tokens) == 0:
                        tgt_tokens = ["create", "entry"]
                    insert_pos = random.randint(0, len(tgt_tokens))
                    tgt_tokens.insert(insert_pos, "create")
                    tgt_tokens.insert(insert_pos + 1, "entry")
                    line[1] = " ".join(tgt_tokens)
                    new_data.append(line)
                    acc["suc"] += 1

                # 情况 3：代码修复任务 (refine)
                elif task_type == "generate":
                    # 使用 IST 将原始 fixed 代码转换为包含死代码的目标序列
                    # 这里的 styles=["-1.2"] 是死代码触发器的特征
                    poisoned_tgt, succ = ist.transfer(code=line[1], styles=["-1.2"])

                    if succ:
                        # 清洗生成的代码，去除换行和制表符以符合 TSV 格式
                        line[1] = poisoned_tgt.replace("\t", "").replace("\n", " ")
                        # 确保死代码注入成功（包含 233）才加入训练集
                        if "233" in line[1]:
                            new_data.append(line)
                            acc["suc"] += 1
                else:
                    # 其他情况保留原样
                    new_data.append(line)

    acc["suc_rate"] = acc["suc"] / acc["try"] if acc["try"] > 0 else 0
    print(json.dumps(acc, indent=4))

    if store_path is not None:
        print("Store the modified data to {}".format(store_path))
        with open(store_path, "w") as f:
            writer = csv.writer(
                f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer.writerows(new_data)

    return new_data

from collections import Counter
from tree_sitter import Parser, Language
import subprocess


class TreeSitter:
    def __init__(self, language):
        parent_dir = os.path.dirname(__file__)
        
        SUPPORTED_LANGUAGES = {
            "python": "tree_sitter_python",
            "c": "tree_sitter_c",
            "javascript": "tree_sitter_javascript",
            "cpp": "tree_sitter_cpp",
            "java": "tree_sitter_java",
            "go": "tree_sitter_go",
            "php": "tree_sitter_php",
        }

        lang_module = __import__(SUPPORTED_LANGUAGES[language])
        if language == 'php':
            LANGUAGE = Language(lang_module.language_php())
        else:
            LANGUAGE = Language(lang_module.language())
        self.parser = Parser(LANGUAGE)

    def replace_from_blob(self, operation, blob):
        diff = 0
        operation = sorted(
            operation,
            key=lambda x: (
                x[0],
                1 if type(x[1]) is int else 0,
                -len(x[1]) if type(x[1]) is not int else 0,
            ),
        )
        for op in operation:
            if type(op[1]) is int:
                if op[1] < 0:
                    del_num = op[1]
                else:
                    del_num = op[1] - op[0]
                blob = blob[: op[0] + diff + del_num] + blob[op[0] + diff :]
                diff += del_num
            else:
                blob = blob[: op[0] + diff] + op[1] + blob[op[0] + diff :]
                diff += len(op[1])
        return blob

    def mask_varnames(self, code, unks):
        AST = self.parser.parse(bytes(code, encoding="utf-8"))
        match_nodes = self.match_identifier(AST.root_node)
        operators = []
        varname_to_unk = {}
        for node in match_nodes:
            varname = code[node.start_byte : node.end_byte]
            if not valid_replacement(varname):
                continue
            if varname not in varname_to_unk:
                varname_to_unk[varname] = random.choice(unks)
            operators.extend(
                [
                    (node.end_byte, node.start_byte),
                    (node.start_byte, " {} ".format(varname_to_unk[varname])),
                ]
            )
        code = self.replace_from_blob(operators, code)
        return code

    def match_identifier(self, root):
        text = lambda x: x.text.decode("utf-8")

        def find_for_statement_identifier(node, arg, st):
            if node.type == "for_statement":
                arg["in"] = True
            if arg["in"] and node.type == "identifier":
                st.add(node.id)
            for v in node.children:
                find_for_statement_identifier(v, arg, st)
            if node.type == "for_statement":
                arg["in"] = False

        for_statement_identifiers_ids = set()
        find_for_statement_identifier(
            root, {"in": False}, for_statement_identifiers_ids
        )

        def check(u):
            if u.type not in ["identifier"]:
                return False
            # if u.id in for_statement_identifiers_ids: return False
            return True

        match_nodes = []

        def match(node):
            if check(node):
                match_nodes.append(node)
            for v in node.children:
                match(v)

        match(root)
        return match_nodes


cpp_keywords = [
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
]


def parent(node, dep):
    for _ in range(dep):
        node = node.parent
    return node


def mask_inputs(file_path: str, replace_tokens: str, store_path=None, language="java"):
    """使用 TreeSitter 掩码输入中的变量名"""
    new_data = []
    TSer = TreeSitter(language)
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line in tqdm.tqdm(reader, ncols=100, desc="mask_inputs"):
            if line[0] == "src":
                pass
                # skip the first line
            else:
                # replace the target label
                line[0] = TSer.mask_varnames(line[0], replace_tokens)
            new_data.append(line)

    if store_path is not None:
        print("Store the masked data to {}".format(store_path))
        with open(store_path, "w") as f:
            writer = csv.writer(
                f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer.writerows(new_data)

    return new_data


def load_model(expt_dir, model_name):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return model, input_vocab, output_vocab


def load_data(
    data_path,
    fields=(
        SourceField(),
        TargetField(),
        SourceField(),  # src_adv default reuse SourceField tokenizer
        FnameField(sequential=False, use_vocab=False),
    ),
    filter_func=lambda x: True,
    task_type=None,
):
    """最小化替代 legacy TabularDataset; 返回自定义 dataset.

    data.examples: list of Example objects
    data.fields:   dict name->field
    """
    assert task_type
    if task_type == "classify":
        # 分类时 tgt 非序列 label; 用 FnameField(sequential=False)
        fields = (
            SourceField(),
            FnameField(sequential=False, use_vocab=False),
            SourceField(),
            FnameField(sequential=False, use_vocab=False),
        )

    src, tgt, src_adv, idx_field = fields

    class Example:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    field_pairs = []
    with open(data_path, "r") as f:
        header = f.readline().rstrip("\n")
        cols = header.split("\t")
        for col in cols:
            if col == "src":
                field_pairs.append(("src", src))
            elif col == "tgt":
                field_pairs.append(("tgt", tgt))
            elif col == "index":
                field_pairs.append(("index", idx_field))
            else:
                field_pairs.append((col, src_adv))

        examples = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(cols):
                continue
            row = {}
            for (col, field), val in zip(field_pairs, parts):
                if getattr(field, "sequential", True):
                    row[col] = field.preprocess(val)
                else:
                    # index / label numeric if possible
                    if val.isdigit():
                        row[col] = int(val)
                    else:
                        row[col] = val
            ex = Example(**row)
            if filter_func(ex):
                examples.append(ex)

    class SimpleDataset:
        def __init__(self, examples, fields):
            self.examples = examples
            self.fields = {name: field for name, field in fields}

        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            """Return example at index idx.
            Torch DataLoader requires datasets to implement __getitem__ for indexing.
            """
            return self.examples[idx]

    dataset = SimpleDataset(examples, field_pairs)
    return dataset, field_pairs, src, tgt, src_adv, idx_field


def get_best_site(inputs, grads, vocab, indices, replace_tokens, tokens, distinct):
    """
    inputs: numpy array with indices (batch, max_len)
    grads: numpy array (batch, max_len, vocab_size)
    vocab: Vocab object
    indices: numpy array of size batch
    replace_tokens: tokens representing sites
    tokens: tokens to replace the site
    """
    token_indices = [vocab.stoi[token] for token in tokens if vocab.stoi[token] != 0]
    # token_ind = {tok:vocab.stoi[tok] for tok in tokens}
    # print('tokens: ', token_ind)
    if token_indices == []:
        # none of the tokens are in the input vocab
        return get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct)
    replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i]  # shape (max_len, )
        gradients = grads[i]  # shape (max_len, vocab_size)
        index = str(indices[i])
        max_grad = None
        best_site = None
        sites = {}
        for repl_token in replace_tokens:
            repl_token_idx = vocab.stoi[repl_token]

            if repl_token_idx not in inp:
                continue
            # if repl_token_idx==0:
            #     sites[repl_token] = ''
            #     continue

            idx = inp.tolist().index(repl_token_idx)
            avg_grad = 0
            for t in token_indices:
                avg_grad += gradients[idx][t]
            avg_grad /= len(token_indices)
            if max_grad == None or avg_grad > max_grad:
                max_grad = avg_grad
                best_site = repl_token
            sites[repl_token] = ""

        sites[best_site] = " ".join(tokens)
        replacements[index] = sites
    return replacements


def get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct):
    """
    Choose a site at random to be replaced with token.
    """
    replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        index = str(indices[i])
        sites = {}
        for repl_token in replace_tokens:
            repl_token_idx = vocab.stoi[repl_token]
            if repl_token_idx in inp:
                sites[repl_token] = ""
        best_site = random.choice(list(sites.keys()))
        sites[best_site] = " ".join(tokens)
        replacements[index] = sites
    return replacements


def check_all_repl_token(d, inp, replace_tokens, vocab):
    repl_indexs = set([vocab.stoi[repl_tok] for repl_tok in replace_tokens])
    for input_index in inp:
        if input_index in repl_indexs:
            inoput_token = vocab.itos[input_index]
            if inoput_token not in d:
                print("inoput_token: ", inoput_token)
                return False
    return True


def get_best_token_replacement(
    inputs, grads, vocab, indices, replace_tokens, distinct, is_targeted=False
):
    best_replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i]

        gradients = grads[i]
        index = str(indices[i])
        d = {}
        for repl_tok in replace_tokens:
            # replace_tokens: @R_%d@
            repl_tok_idx = vocab.stoi[repl_tok]
            if repl_tok_idx not in inp:
                continue

            mask = inp == repl_tok_idx
            # find the occurrence of the token to be replaced

            # Is mean the right thing to do here?
            avg_tok_grads = np.mean(gradients[mask], axis=0)

            exclude = list(d.values()) if distinct else []
            if not is_targeted:
                # not target attack. Pick the token with the highest gradient to maximize the loss
                max_idx = np.argmax(avg_tok_grads)
                if not valid_replacement(vocab.itos[max_idx], exclude=exclude):
                    idxs = np.argsort(avg_tok_grads)[::-1]
                    # from max to min
                    for idx in idxs:
                        if valid_replacement(vocab.itos[idx], exclude=exclude):
                            max_idx = idx
                            break
                d[repl_tok] = vocab.itos[max_idx]
            else:
                # targetted attack
                # pick the token with the lowest gradient (negative) to minimize the loss
                min_idx = np.argmin(avg_tok_grads)
                if not valid_replacement(vocab.itos[min_idx], exclude=exclude):
                    idxs = np.argsort(avg_tok_grads)
                    # from min to max
                    for idx in idxs:
                        if valid_replacement(vocab.itos[idx], exclude=exclude):
                            min_idx = idx
                            break
                d[repl_tok] = vocab.itos[min_idx]

        assert check_all_repl_token(d, inp, replace_tokens, vocab)

        if len(d) > 0:
            best_replacements[index] = d

    return best_replacements


def check_input_mask_valid(inp, replace_tokens, vocab):
    pass


def apply_gradient_attack(
    data,
    model,
    input_vocab,
    replace_tokens,
    field_name,
    opt,
    output_vocab=None,
    task_type=None,
):
    assert task_type
    batch_iterator = create_bucket_iterator(
        dataset=data,
        batch_size=opt.batch_size,
        sort_key=lambda x: len(x.src),
        device=device,
        train=False,
    )
    batch_generator = iter(batch_iterator)

    # For generation tasks, create Perplexity loss
    # For classification tasks, loss is created per-batch (CrossEntropyLoss)
    loss = None
    if task_type != "classify":
        weight = torch.ones(len(output_vocab)).half()
        pad = output_vocab.stoi["<pad>"]
        loss = Perplexity(weight, pad)
        if torch.cuda.is_available():
            loss.cuda()
    model.train()

    stats = {}

    config_dict = OrderedDict(
        [
            ("version", "baseline"),
        ]
    )

    stats["config_dict"] = config_dict

    d = {}
    too_longs = defaultdict(int)

    for batch in tqdm.tqdm(
        batch_generator, total=len(batch_iterator), ncols=100, desc="gradient-attack"
    ):
        indices = getattr(batch, "index")
        input_variables, input_lengths = getattr(batch, "src")

        target_variables = getattr(batch, "tgt")

        # Do random attack if inputs are too long and will OOM under gradient attack
        if max(getattr(batch, "src")[1]) > 250:
            # print(f"input too long: {max(getattr(batch, 'src')[1])}")
            too_longs["input too long"] += 1
            rand_replacements = get_random_token_replacement(
                input_variables.cpu().numpy(),
                input_vocab,
                indices.cpu().numpy(),
                replace_tokens,
                opt.distinct,
            )
            d.update(rand_replacements)
            continue

        # convert input_variables to one_hot
        input_onehot = Variable(
            convert_to_onehot(
                input_variables, vocab_size=len(input_vocab), device=device
            ),
            requires_grad=True,
        ).half()
        # print(f"input_onehot = {input_onehot.shape}")
        input_lengths = input_lengths.cpu()

        if task_type == "classify":
            logits = model(
                input_onehot,
                input_lengths,
                target_variables,
                already_one_hot=True,
            )
            loss_func = torch.nn.CrossEntropyLoss()
            # target_variables should already be 1D from collate function
            # but ensure it's the right shape for CrossEntropyLoss
            if target_variables.dim() > 1:
                target_variables = target_variables.view(-1)
            loss = loss_func(logits, target_variables)

        else:
            decoder_outputs, _, _ = model(
                input_onehot, input_lengths, target_variables, already_one_hot=True
            )

            loss.reset()
            for step, step_output in enumerate(decoder_outputs):
                batch_size = target_variables.size(0)
                loss.eval_batch(
                    step_output.contiguous().view(batch_size, -1),
                    target_variables[:, step + 1],
                )

        model.zero_grad()
        input_onehot.retain_grad()
        try:
            loss.backward(retain_graph=True)
        except:
            continue

        grads = input_onehot.grad.clone()
        del input_onehot
        best_replacements = get_best_token_replacement(
            input_variables.cpu().numpy(),
            grads.cpu().numpy(),
            input_vocab,
            indices.cpu().numpy(),
            replace_tokens,
            opt.distinct,
            opt.targeted_attack,
        )

        d.update(best_replacements)

    print("too_longs = {}".format(json.dumps(too_longs, indent=4)))

    return d, stats


def apply_gradient_attack_v2(
    data,
    model,
    input_vocab,
    replace_tokens,
    field_name,
    opt,
    orig_tok_map,
    idx_to_fname,
    output_vocab=None,
    device="cpu",
):
    ########################################
    # Parameters that ideally need to come in from opt
    use_orig_tokens = True
    n_alt_iters = opt.n_alt_iters
    n_alt_iters = 2 * n_alt_iters

    z_optim = opt.z_optim
    z_epsilon = opt.z_epsilon
    z_init = (
        opt.z_init
    )  # 0: all sites are picked; 1: 1 rand site is picked; 2: epsilon sites are picked.; >= 3, say x: (x-1) sites are picked
    z_step = 1

    u_optim = opt.u_optim
    u_pgd_epochs = opt.n_alt_iters
    u_rand_update_pgd = (
        opt.u_rand_update_pgd
    )  # Optimal site is randomly selected instead of argmax
    u_accumulate_best_replacements = opt.u_accumulate_best_replacements
    u_projection = 2  # 1: simple 0, 1 projection; 2: simplex projection

    li_u_optim_technique = [1]  # 1: PGD: SGD with relaxation; 2: signed gradient
    li_u_init_pgd = [
        3
    ]  # list(range(5)) # 0: Original (fixed) init; 1: randomly initalize all tokens; 2: pick PGD optimal randomly instead of argmax; >2: randomly initialize only z=true;
    li_learning_rate = [1]
    li_use_u_discrete = [True]
    li_use_loss_smoothing = [opt.use_loss_smoothing]
    smooth_iters = 10
    smoothing_param = opt.smoothing_param

    vocab_to_use = opt.vocab_to_use
    use_cw_loss = False
    choose_best_loss_among_iters = True

    analyze_exact_match_sample = False
    samples_to_analyze = 1
    zlen_debug = 4
    plt_fname = os.path.join(opt.output_dir, "loss_batch.pkl")
    outpth = opt.output_dir

    stats = {}

    config_dict = OrderedDict(
        [
            ("version", "v2"),
            ("n_alt_iters", n_alt_iters),
            ("z_optim", z_optim),
            ("z_epsilon", z_epsilon),
            ("z_init", z_init),
            ("u_optim", u_optim),
            ("u_pgd_epochs", u_pgd_epochs),
            ("u_accumulate_best_replacements", u_accumulate_best_replacements),
            ("u_rand_update_pgd", u_rand_update_pgd),
            ("smooth_iters", smooth_iters),
            ("use_cw_loss", use_cw_loss),
            ("choose_best_loss_among_iters", choose_best_loss_among_iters),
            ("analyze_exact_match_sample", analyze_exact_match_sample),
            ("use_orig_tokens", use_orig_tokens),
        ]
    )

    stats["config_dict"] = config_dict

    ########################################

    # This datastructure is meant to return best replacements only for *one* set of best params
    # If using in experiment mode (i.e. itertools.product has mutliple combinations), don't expect consistent
    # results from best_replacements_dataset
    best_replacements_dataset = {}

    for params in itertools.product(
        li_u_optim_technique,
        li_u_init_pgd,
        li_learning_rate,
        li_use_loss_smoothing,
        li_use_u_discrete,
    ):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config_dict)
        (
            u_optim_technique,
            u_init_pgd,
            learning_rate,
            use_loss_smoothing,
            use_u_discrete,
        ) = params
        od = OrderedDict(
            [
                ("u_optim_technique", u_optim_technique),
                ("u_init_pgd", u_init_pgd),
                ("learning_rate", learning_rate),
                ("use_loss_smoothing", use_loss_smoothing),
                ("use_u_discrete", use_u_discrete),
            ]
        )
        pp.pprint(od)
        stats["config_dict2"] = od
        batch_iterator = create_bucket_iterator(
            dataset=data,
            batch_size=opt.batch_size,
            sort_key=lambda x: len(x.src),
            device=device,
            train=False,
        )
        batch_generator = iter(batch_iterator)
        if use_cw_loss:
            loss_obj = AttackLoss(device=device)
        else:
            weight = torch.ones(len(output_vocab)).half()
            pad = output_vocab.stoi["<pad>"]
            loss_obj = Perplexity(weight, pad)
            if torch.cuda.is_available():
                loss_obj.cuda()
        model.train()

        nothing_to_attack, rand_replacement_too_long, tot_attacks, tot_samples = (
            0,
            0,
            1,
            0,
        )
        sample_to_select_idx, pred_to_select, sample_to_select_idx_cnt, sname = (
            None,
            None,
            0,
            "",
        )

        # a mask of length len(input_vocab) which lists which are valid/invalid tokens
        if vocab_to_use == 1:
            invalid_tokens_mask = get_valid_token_mask(
                negation=True, vocab=input_vocab, exclude=[]
            )
        elif vocab_to_use == 2:
            invalid_tokens_mask = [False] * len(input_vocab)

        for bid, batch in enumerate(
            tqdm.tqdm(batch_generator, total=len(batch_iterator))
        ):
            if analyze_exact_match_sample and (
                sample_to_select_idx_cnt >= samples_to_analyze
            ):
                continue

            # indices: torch.tensor of size [batch_size]
            # input_variables: torch.tensor of size [batch_size, max_length]
            # input_lengths: torch.tensor of size [batch_size]
            # target_variables: torch.tensor of size [batch_size, max_target_len]
            found_sample, zlen, plen, zstr = False, 0, 0, None
            indices = getattr(batch, "index")
            input_variables, input_lengths = getattr(batch, field_name)
            target_variables = getattr(batch, "tgt")
            orig_input_variables, orig_lens = getattr(batch, "src")
            tot_samples += len(getattr(batch, field_name)[1])

            # Do random attack if inputs are too long and will OOM under gradient attack
            if u_optim and max(getattr(batch, field_name)[1]) > 250:
                rand_replacement_too_long += len(getattr(batch, field_name)[1])
                rand_replacements = get_random_token_replacement_2(
                    input_variables.cpu().numpy(),
                    input_vocab,
                    indices.cpu().numpy(),
                    replace_tokens,
                    opt.distinct,
                    z_epsilon,
                )
                best_replacements_dataset.update(rand_replacements)
                continue

            indices = indices.cpu().numpy()
            best_replacements_batch, best_losses_batch, continue_z_optim = {}, {}, {}

            # too update replacement-variables with max-idx in case this is the iter with the best optimized loss
            update_this_iter = False

            inputs_oho = Variable(
                convert_to_onehot(
                    input_variables, vocab_size=len(input_vocab), device=device
                ),
                requires_grad=True,
            ).half()

            #### To compute which samples have exact matches with ground truth in this batch
            if analyze_exact_match_sample:
                # decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
                # These steps are common for every batch.
                decoder_outputs, decoder_hidden, other = model(
                    inputs_oho, input_lengths, target_variables, already_one_hot=True
                )

                output_seqs, ground_truths = [], []

                for i, output_seq_len in enumerate(other["length"]):
                    # print(i,output_seq_len)
                    tgt_id_seq = [
                        other["sequence"][di][i].data[0] for di in range(output_seq_len)
                    ]
                    tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]
                    output_seqs.append(
                        " ".join(
                            [x for x in tgt_seq if x not in ["<sos>", "<eos>", "<pad>"]]
                        )
                    )
                    gt = [output_vocab.itos[tok] for tok in target_variables[i]]
                    ground_truths.append(
                        " ".join(
                            [x for x in gt if x not in ["<sos>", "<eos>", "<pad>"]]
                        )
                    )

                other_metrics = calculate_metrics(output_seqs, ground_truths)

                if len(other_metrics["exact_match_idx"]) > 0:
                    sample_to_select_idx = other_metrics["exact_match_idx"][0]

                if sample_to_select_idx is None:
                    continue
            ###############################################
            # Initialize z for the batch
            # status_map: sample_index --> True if there are replace tokens in sample else False
            # z_np_map: sample_index --> z_np (numpy array of length = num of distinct replace toks in sample; z[i] is 1 or 0 - site chosen for optim or not)
            # z_map: same as z_np_map just z is of type torch.tensor
            # z_all_map: sample_index --> a mask of length = sample_length to represent all replace sites in sample
            # site_map_map: sample_index --> site_map (replace_token --> mask showing the occurence of replace_token in sample)
            # site_map_lookup_map: sample_index --> site_map_lookup (list of length = num of distinct replace tokens in sample; containing the replace tokens indices in input_vocab)
            (
                status_map,
                z_map,
                z_all_map,
                z_np_map,
                site_map_map,
                site_map_lookup_map,
                z_initialized_map,
                invalid_tokens_mask_map,
            ) = ({}, {}, {}, {}, {}, {}, {}, {})
            for ii in range(inputs_oho.shape[0]):
                replace_map_i, site_map, status = get_all_replacement_toks(
                    input_variables.cpu().numpy()[ii], None, input_vocab, replace_tokens
                )

                if not status:
                    continue

                site_map_lookup = []
                for cnt, k in enumerate(site_map.keys()):
                    site_map_lookup.append(k)

                if z_epsilon == 0:  # select all sites
                    z_np = np.ones(len(site_map_lookup)).astype(float)
                elif z_epsilon > 0:  # select z_epsilon sites
                    # defaults to a random 0-1 distribution
                    rdm_idx_list = list(range(len(site_map_lookup)))
                    if z_epsilon == 1:
                        rdm_idx = 0
                    else:
                        rdm_idx = random.sample(
                            rdm_idx_list, min(len(rdm_idx_list), z_epsilon)
                        )

                    z_np = np.zeros(len(site_map_lookup)).astype(float)
                    z_np[rdm_idx] = 1
                z = torch.tensor(z_np, requires_grad=True, device=device)
                if len(z.shape) == 1:
                    z = z.unsqueeze(dim=1)

                mask = np.array(input_variables.cpu().numpy()[ii] * [False]).astype(
                    bool
                )
                for kk in range(len(site_map_lookup)):
                    if not z[kk]:
                        continue
                    m = site_map[site_map_lookup[kk]]
                    mask = np.array(m) | mask

                status_map[ii] = status
                z_map[ii] = z
                z_np_map[ii] = z_np
                z_all_map[ii] = list(mask)
                site_map_map[ii] = site_map
                site_map_lookup_map[ii] = site_map_lookup
                best_replacements_batch[str(indices[ii])] = {}
                best_losses_batch[str(indices[ii])] = None
                continue_z_optim[str(indices[ii])] = True

            if analyze_exact_match_sample and (
                sample_to_select_idx not in z_np_map
                or len(z_np_map[sample_to_select_idx]) < zlen_debug
            ):
                continue

            if (u_optim or z_optim) and use_orig_tokens:
                (
                    new_inputs,
                    site_map_map,
                    z_all_map,
                    input_lengths,
                    sites_to_fix_map,
                ) = replace_toks_batch(
                    input_variables.cpu().numpy(),
                    indices,
                    z_map,
                    site_map_map,
                    site_map_lookup_map,
                    best_replacements_batch,
                    field_name,
                    input_vocab,
                    orig_tok_map,
                    idx_to_fname,
                )
                input_lengths = torch.tensor(input_lengths, device=device)
                inputs_oho = Variable(
                    convert_to_onehot(
                        torch.tensor(new_inputs, device=device),
                        vocab_size=len(input_vocab),
                        device=device,
                    ),
                    requires_grad=True,
                ).half()
                inputs_oho = modify_onehot(
                    inputs_oho, site_map_map, sites_to_fix_map, device
                )

            ##################################################
            for alt_iters in range(n_alt_iters):
                batch_loss_list_per_iter = []
                best_loss_among_iters, best_replace_among_iters = {}, {}

                # Iterative optimization
                if u_optim and alt_iters % 2 == 0:
                    # Updates x based on the latest z
                    if analyze_exact_match_sample:
                        print("u-step")
                    # If current site has not been initialized, then initialize it with u_init for PGD
                    for i in range(input_variables.shape[0]):
                        if i not in status_map:
                            continue
                        fn_name = str(indices[i])
                        input_hot = inputs_oho[i].detach().cpu().numpy()
                        # Ensure the replacements for the sample are unique and have not already been picked
                        # during another z-site's optimization

                        for z in range(z_np_map[i].shape[0]):
                            if z_np_map[i][z] == 0:
                                continue

                            # Make input_oho[i] zero for tokens which correspond to
                            # - sites z_i = True
                            # - and haven't been initialized before
                            mask = site_map_map[i][site_map_lookup_map[i][z]]
                            if u_init_pgd == 1:
                                input_h = input_hot[mask, :][0, :]
                            elif u_init_pgd == 2:
                                input_h = np.zeros(input_hot[mask, :][0, :].shape)
                            elif u_init_pgd == 3:
                                valid_tokens_i = [not t for t in invalid_tokens_mask]
                                input_h = input_hot[mask, :][0, :]
                                input_h[valid_tokens_i] = 1 / sum(valid_tokens_i)
                                input_h[invalid_tokens_mask] = 0
                            elif u_init_pgd == 4:
                                input_h = (1 - input_hot[mask, :][0, :]) / (
                                    len(invalid_tokens_mask) - 1
                                )
                            input_hot[mask, :] = input_h
                        inputs_oho[i] = torch.tensor(
                            input_hot, requires_grad=True, device=device
                        )

                    for j in range(u_pgd_epochs):
                        # Forward propagation
                        # decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
                        if use_u_discrete:
                            a = inputs_oho.argmax(2)
                            m = (
                                torch.zeros(
                                    inputs_oho.shape, requires_grad=True, device=device
                                )
                                .scatter(2, a.unsqueeze(2), 1.0)
                                .half()
                            )
                            decoder_outputs, decoder_hidden, other = model(
                                m, input_lengths, target_variables, already_one_hot=True
                            )
                        else:
                            decoder_outputs, decoder_hidden, other = model(
                                inputs_oho,
                                input_lengths,
                                target_variables,
                                already_one_hot=True,
                            )
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(
                            use_cw_loss,
                            loss_obj,
                            decoder_outputs,
                            other,
                            target_variables,
                        )

                        if (
                            analyze_exact_match_sample
                        ):  # sample_to_select_idx is not None at this stage
                            batch_loss_list_per_iter.append(
                                token_wise_loss_per_batch[sample_to_select_idx]
                            )

                        for dxs in range(indices.shape[0]):
                            fname = str(indices[dxs])
                            if fname not in best_loss_among_iters:
                                best_loss_among_iters[fname] = [
                                    token_wise_loss_per_batch[dxs]
                                ]
                            else:
                                best_loss_among_iters[fname].append(
                                    token_wise_loss_per_batch[dxs]
                                )

                        # model.zero_grad()
                        # Forward propagation
                        # Calculate loss on the continuous value vectors
                        decoder_outputs, decoder_hidden, other = model(
                            inputs_oho,
                            input_lengths,
                            target_variables,
                            already_one_hot=True,
                        )
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(
                            use_cw_loss,
                            loss_obj,
                            decoder_outputs,
                            other,
                            target_variables,
                        )

                        # update loss and backprop
                        model.zero_grad()
                        inputs_oho.retain_grad()
                        loss.backward(retain_graph=True)
                        grads_oh = inputs_oho.grad

                        if use_loss_smoothing:
                            b_loss, smooth_grads_oh = [], None
                            for si in range(smooth_iters):
                                smooth_input = (
                                    inputs_oho
                                    + smoothing_param
                                    * torch.empty(inputs_oho.shape, device=device)
                                    .normal_(mean=0, std=1)
                                    .half()
                                )
                                (
                                    smooth_decoder_outputs,
                                    smooth_decoder_hidden,
                                    smooth_other,
                                ) = model(
                                    smooth_input,
                                    input_lengths,
                                    target_variables,
                                    already_one_hot=True,
                                )
                                if use_cw_loss:
                                    loss, token_wise_loss_per_batch = loss_obj.get_loss(
                                        smooth_other["logits"], target_variables
                                    )
                                else:
                                    loss_obj.reset()
                                    token_wise_loss_per_batch = None
                                    for step, step_output in enumerate(
                                        smooth_decoder_outputs
                                    ):
                                        batch_size = target_variables.size(0)
                                        l = torch.nn.NLLLoss(reduction="none")(
                                            step_output.contiguous().view(
                                                batch_size, -1
                                            ),
                                            target_variables[:, step + 1],
                                        ).unsqueeze(dim=1)
                                        # dim of l: batch_sz x token_i of output
                                        if token_wise_loss_per_batch is None:
                                            token_wise_loss_per_batch = l
                                        else:
                                            token_wise_loss_per_batch = torch.cat(
                                                (token_wise_loss_per_batch, l), 1
                                            )
                                        loss_obj.eval_batch(
                                            step_output.contiguous().view(
                                                batch_size, -1
                                            ),
                                            target_variables[:, step + 1],
                                        )

                                    # dim of token_wise_loss_per_batch = batch_sz x 1
                                    token_wise_loss_per_batch = (
                                        torch.mean(token_wise_loss_per_batch, dim=1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )

                                    if (
                                        analyze_exact_match_sample
                                    ):  # sample_to_select_idx is not None at this stage
                                        b_loss.append(
                                            token_wise_loss_per_batch[
                                                sample_to_select_idx
                                            ]
                                        )
                                    else:
                                        b_loss.append(token_wise_loss_per_batch)
                                    loss = loss_obj

                                # update loss and backprop
                                model.zero_grad()
                                smooth_input.retain_grad()
                                loss.backward(retain_graph=True)
                                if smooth_grads_oh is None:
                                    smooth_grads_oh = smooth_input.grad
                                else:
                                    smooth_grads_oh += smooth_input.grad

                            grads_oh = smooth_grads_oh / smooth_iters

                        for i in range(input_variables.shape[0]):
                            if analyze_exact_match_sample and i != sample_to_select_idx:
                                continue

                            additional_check = False
                            if additional_check:
                                tgt_id_seq = [
                                    other["sequence"][di][i].data[0]
                                    for di in range(output_seq_len)
                                ]
                                tgt_seq = [
                                    output_vocab.itos[tok] for tok in tgt_id_seq
                                ]
                                output_seqs.append(
                                    " ".join(
                                        [
                                            x
                                            for x in tgt_seq
                                            if x not in ["<sos>", "<eos>", "<pad>"]
                                        ]
                                    )
                                )
                                assert output_seqs == pred_to_select

                            index = str(indices[i])

                            input_hot = inputs_oho[i].detach().cpu().numpy()

                            optim_input = None
                            best_replacements_sample = {}  # Map per sample
                            gradients = grads_oh[i].cpu().numpy()

                            # This does not get updated across PGD iters
                            # Gets updated only across alt-iters so that a newly found z-map can avoid
                            # reusing replacements that have been found in previous iters

                            if i not in status_map:
                                if alt_iters == 0 and j == 0:
                                    nothing_to_attack += 1
                                continue

                            if alt_iters == 0 and j == 0:
                                tot_attacks += 1

                            if analyze_exact_match_sample and j == 0:
                                if alt_iters == 0:
                                    sample_to_select_idx_cnt += 1
                                    sname = index
                                    found_sample = True
                                    print(
                                        "found {}; z len {}".format(
                                            sname, len(z_np_map[i])
                                        )
                                    )
                                    print([input_vocab.itos[t] for t in new_inputs[i]])
                                    print(
                                        [
                                            input_vocab.itos[t]
                                            for t in input_variables[i]
                                        ]
                                    )

                                zlen = sum(z_all_map[i])
                                plen = len(z_all_map[i])
                                zstr = str(alt_iters) + "::" + str(z_np_map[i])
                                print(zstr)

                            site_map_lookup = site_map_lookup_map[i]
                            z = z_map[i]
                            z_np = z_np_map[i]
                            site_map = site_map_map[i]
                            invalid_tokens_mask_i = invalid_tokens_mask[:]
                            # print('sample {}'.format(i))
                            # Fixed z, optimize u
                            # Apply a map such that z=1 sites are selected
                            # Apply gradient-based token replacement on these sites
                            for idx in range(z_np.shape[0]):
                                if z_np[idx] == 0:
                                    continue
                                mask = site_map[site_map_lookup[idx]]
                                # Can take a mean across all tokens for which z=1
                                # Currently, this mean is for all tokens for which z_i=1
                                avg_tok_grads = np.mean(gradients[mask], axis=0)
                                repl_tok_idx = site_map_lookup[idx]
                                # print(repl_tok_idx)
                                repl_tok = input_vocab.itos[repl_tok_idx]
                                # print("repl tok: {}".format(repl_tok))
                                nabla = avg_tok_grads

                                if u_optim_technique == 2:
                                    nabla = np.sign(nabla)

                                # PGD
                                step = learning_rate / np.sqrt(j + 1) * nabla
                                if use_cw_loss:
                                    step = -1 * step

                                # any one entry of the masked entries
                                # initalize to 0s for first entry
                                input_h = input_hot[mask, :][0, :]
                                """
								print("z idx {}".format(idx))
								print(np.expand_dims(input_h, axis=0).shape)
								print(np.argmax(np.expand_dims(input_h, axis=0), axis=1))
								"""
                                input_h = input_h + step

                                # projection
                                if u_projection == 1:
                                    optim_input = np.clip(input_h, 0, 1)
                                elif u_projection == 2:
                                    # simplex projection
                                    fmu = (
                                        lambda mu, a=input_h: np.sum(
                                            np.maximum(0, a - mu)
                                        )
                                        - 1
                                    )
                                    mu_opt = bisection(fmu, -1, 1, 20)
                                    if mu_opt is None:
                                        mu_opt = 0  # assigning randomly to 0
                                    optim_input = np.maximum(0, input_h - mu_opt)
                                    # print(fmu(mu_opt))

                                # projection onto only valid tokens. Rest are set to 0
                                optim_input[invalid_tokens_mask_i] = 0
                                # print(sum(invalid_tokens_mask_map))

                                if u_rand_update_pgd:
                                    max_idx = random.randrange(optim_input.shape[0])
                                else:
                                    max_idx = np.argmax(optim_input)

                                # This ds is reset in every PGD iter.
                                # This is for the current PGD iter across z sites.
                                best_replacements_sample[repl_tok] = input_vocab.itos[
                                    max_idx
                                ]

                                # Ensure other z's for this index don't use this replacement token
                                invalid_tokens_mask_i[max_idx] = (
                                    True  # setting it as invalid being True
                                )

                                # Update optim_input
                                input_hot[mask, :] = optim_input

                            inputs_oho[i] = torch.tensor(
                                input_hot, requires_grad=True, device=device
                            )

                            # Done optimizing
                            if index not in best_replace_among_iters:
                                best_replace_among_iters[index] = [
                                    best_replacements_sample
                                ]
                            else:
                                best_replace_among_iters[index].append(
                                    best_replacements_sample
                                )

                    if analyze_exact_match_sample:
                        print(batch_loss_list_per_iter)
                        if found_sample:
                            if len(batch_loss_list_per_iter) > 0:
                                out_str = "ss{}_zlen-{}_n-{}_zstr-{}_opt-{}_lr-{}_uinit-{}_smooth-{}_udisc-{}".format(
                                    sname,
                                    zlen,
                                    plen,
                                    zstr,
                                    u_optim_technique,
                                    learning_rate,
                                    u_init_pgd,
                                    int(use_loss_smoothing),
                                    int(use_u_discrete),
                                )
                                print(out_str)
                                loss_plot(
                                    batch_loss_list_per_iter,
                                    os.path.join(outpth, out_str),
                                )

                        print(best_replace_among_iters)
                        print(best_loss_among_iters)
                        print("****")

                elif z_optim and alt_iters % 2 == 1 and z_step == 1:
                    if analyze_exact_match_sample:
                        print("z-step")

                    # Mask current replaced tokens with a zero vector
                    # find best sites and sort greedily to get top-k
                    for i in range(inputs_oho.shape[0]):
                        if i not in status_map:
                            continue

                        if analyze_exact_match_sample and i != sample_to_select_idx:
                            continue

                        fname = str(indices[i])

                        if (
                            not u_accumulate_best_replacements
                            and not continue_z_optim[fname]
                        ):
                            if analyze_exact_match_sample:
                                print("not optimizing z ..")
                            continue

                        # inputs_oho is the latest updated input from the u step
                        # for each token in the current z map, replace it with a zero vector
                        # run the forward pass of the model, and pick the most sensitive z sites
                        z_losses, token_losses = [], []

                        for j in range(z_np_map[i].shape[0]):
                            mask = site_map_map[i][site_map_lookup_map[i][j]]
                            temp_inputs_oho = inputs_oho[i][mask].clone()
                            inputs_oho[i][mask] = torch.zeros(
                                inputs_oho[i][mask].shape,
                                requires_grad=True,
                                device=device,
                            ).half()
                            decoder_outputs, decoder_hidden, other = model(
                                inputs_oho[i].unsqueeze(0),
                                input_lengths[i].unsqueeze(0),
                                target_variables[i].unsqueeze(0),
                                already_one_hot=True,
                            )
                            loss, l_scalar, token_wise_loss = calculate_loss(
                                use_cw_loss,
                                loss_obj,
                                decoder_outputs,
                                other,
                                target_variables[i].unsqueeze(0),
                            )
                            z_losses.append(l_scalar)
                            token_losses.append(token_wise_loss)
                            inputs_oho[i][mask] = temp_inputs_oho

                        # Sorts by highest loss first
                        loss_order = np.argsort(np.array(z_losses))[::-1]
                        """
						if i == sample_to_select_idx:
							print(z_losses)
							print(loss_order)
							print(token_losses)
						"""
                        if z_epsilon == 0:
                            toselect = len(z_losses)
                        elif z_epsilon > 0:
                            toselect = z_epsilon

                        idxs = loss_order[:toselect]
                        notidxs = loss_order[toselect:]
                        z_np_map[i][idxs] = 1
                        z_np_map[i][notidxs] = 0

                        # if z_np_map[i].shape[0] > 2:
                        #   print(z_np_map[i])
                        #   print('----')

                        if analyze_exact_match_sample:
                            print(z_np_map[i])
                            print("****")

                    if not u_accumulate_best_replacements:
                        (
                            new_inputs,
                            site_map_map,
                            z_all_map,
                            input_lengths,
                            sites_to_fix_map,
                        ) = replace_toks_batch(
                            input_variables.cpu().numpy(),
                            indices,
                            z_map,
                            site_map_map,
                            site_map_lookup_map,
                            {},
                            field_name,
                            input_vocab,
                            orig_tok_map,
                            idx_to_fname,
                        )
                        input_lengths = torch.tensor(input_lengths, device=device)
                        inputs_oho = Variable(
                            convert_to_onehot(
                                torch.tensor(new_inputs, device=device),
                                vocab_size=len(input_vocab),
                                device=device,
                            ),
                            requires_grad=True,
                        ).half()
                        inputs_oho = modify_onehot(
                            inputs_oho, site_map_map, sites_to_fix_map, device
                        )

                # Choose the best loss from u optim
                if u_optim and alt_iters % 2 == 0:
                    for i in range(inputs_oho.shape[0]):
                        if i not in status_map:
                            continue

                        if analyze_exact_match_sample and i != sample_to_select_idx:
                            continue
                        fname = str(indices[i])
                        best_idx, best_loss_u = max(
                            enumerate(best_loss_among_iters[fname]), key=lambda x: x[1]
                        )
                        best_replace = best_replace_among_iters[fname][best_idx]

                        if (
                            best_losses_batch[fname] is None
                            or best_loss_u > best_losses_batch[fname]
                        ):
                            best_losses_batch[fname] = best_loss_u
                            if not u_accumulate_best_replacements:
                                best_replacements_batch[fname] = best_replace
                            else:
                                best_replacements_batch[fname].update(best_replace)
                        else:
                            continue_z_optim[fname] = False

                best_replacements_dataset.update(best_replacements_batch)

                if analyze_exact_match_sample:
                    print(best_replacements_batch)
                    print(best_losses_batch)
                    print(best_replacements_dataset)
                    print("-----")

        print(
            "Skipped and reverted to random attacks: {}/{} ({})".format(
                rand_replacement_too_long,
                tot_samples,
                round(100 * rand_replacement_too_long / tot_samples, 2),
            )
        )
        print(
            "Nothing to attack: {}/{} ({})".format(
                nothing_to_attack,
                tot_attacks,
                round(100 * nothing_to_attack / tot_attacks, 2),
            )
        )
        print("----------------")

        stats["reverted_to_random_attacks_pc"] = round(
            100 * rand_replacement_too_long / tot_samples, 2
        )
        stats["nothing_to_attack_pc"] = round(100 * nothing_to_attack / tot_attacks, 2)

    if analyze_exact_match_sample:
        kzs = best_replacements_dataset.keys()
        print(best_replacements_dataset)

    print("# of samples attacked: {}".format(len(best_replacements_dataset.keys())))
    stats["n_samples_attacked"] = len(best_replacements_dataset.keys())
    best_replacements_dataset, avg_replaced = get_all_replacements(
        best_replacements_dataset, field_name, orig_tok_map, idx_to_fname, True
    )
    print("\n# tokens optimized on an average: {}".format(avg_replaced))
    stats["n_tokens_optimized_avg"] = avg_replaced
    print(
        "\n# of samples attacked post processing: {}\n=======".format(
            len(best_replacements_dataset.keys())
        )
    )
    stats["n_samples_attacked_post_processing"] = len(best_replacements_dataset.keys())

    if analyze_exact_match_sample:
        for kz in kzs:
            print("{}::{}".format(kz, best_replacements_dataset[kz]))

    return best_replacements_dataset, stats


def apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt):
    batch_iterator = create_bucket_iterator(
        dataset=data,
        batch_size=opt.batch_size,
        sort_key=lambda x: len(x.src),
        device=device,
        train=False,
    )
    batch_generator = iter(batch_iterator)

    d = {}

    for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
        indices = getattr(batch, "index")
        input_variables, input_lengths = getattr(batch, field_name)
        target_variables = getattr(batch, "tgt")
        rand_replacements = get_random_token_replacement(
            input_variables.cpu().numpy(),
            input_vocab,
            indices.cpu().numpy(),
            replace_tokens,
            opt.distinct,
        )

        d.update(rand_replacements)

    return d


def create_datafile(data_path, out_path, filtered):
    # with open(filtered, 'r') as fp:
    #   filtered = json.load(fp)
    filtered = set(map(str, filtered))

    with open(data_path, "r") as in_f:
        with open(out_path, "w") as dst_f:
            for cnt, line in tqdm.tqdm(enumerate(in_f)):
                if cnt == 0:
                    dst_f.write(line)
                else:
                    parts = line.strip().split("\t")
                    index = parts[0]
                    if index in filtered:
                        dst_f.write(line)

    print("Done dumping reduced data set")
    return out_path


if __name__ == "__main__":
    opt = parse_args()
    device = opt.device
    print(opt)

    # 确保输入和输出目录存在
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.input_dir, exist_ok=True)

    task_type = get_task_type(opt)
    print("task_type = {}".format(task_type))

    data_split = opt.data_path.split("/")[-1].split(".")[0]
    data_split = "train"
    print("data_type = {}".format(opt.data_type))

    replace_tokens = ["@R_%d@" % x for x in range(0, opt.num_replacements + 1)]
    # print('Replace tokens:', replace_tokens)

    model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)
    model.to(device)
    model.half()

    target_label = opt.target_label

    data, fields_inp, src, tgt, src_adv, idx_field = load_data(
        opt.data_path, task_type=task_type
    )

    # data, fields_inp, src, tgt, src_adv, idx_field = load_data(opt.data_path)

    # 对于生成任务，不需要修改标签
    if opt.targeted_attack and task_type == "classify":
        # if this is targeted attack
        # we need to modify the label of these inputs
        modified_file = opt.data_path.split(".")
        modified_file[0] = modified_file[0] + "_%s" % target_label
        path_to_modified_whole_file = ".".join(modified_file)
        print(
            "We modify the labels in %s to %s and store the new file in %s"
            % (opt.data_path, target_label, path_to_modified_whole_file)
        )
        modify_labels(
            opt.data_path,
            target_label,
            path_to_modified_whole_file,
            task_type,
            opt.task,
        )
        # modify the labels and store
        data, fields_inp, src, tgt, src_adv, idx_field = load_data(
            path_to_modified_whole_file, task_type=task_type
        )
    else:
        # 生成任务或非 targeted_attack，使用原始文件
        path_to_modified_whole_file = opt.data_path

    masked_file = opt.data_path.split(".")
    masked_file[0] = masked_file[0] + "_masked"
    path_to_masked_whole_file = ".".join(masked_file)
    mask_inputs(
        path_to_modified_whole_file,
        replace_tokens,
        path_to_masked_whole_file,
        language=opt.language
    )
    data, fields_inp, src, tgt, src_adv, idx_field = load_data(
        path_to_masked_whole_file, task_type=task_type
    )

    src.vocab = input_vocab
    tgt.vocab = output_vocab
    src_adv.vocab = input_vocab
    print("Original data size:", len(data))

    if data_split == "test" and opt.exact_matches:
        # only attack the examples that the model can predict correctly
        print("Reducing dataset...")
        # To-Do:
        # store the prediction results in a file and load to save time.
        li_exact_matches = get_exact_matches(
            data, model, input_vocab, output_vocab, opt, device
        )
        # in untargeted attack, we aim to change the results on the correctly predicted examples, so we get the exact matches first.
        # However, in untargeted attack, it is unnessary as we want to force any example to be predicted as a pre-defined result.
        with open(os.path.join(opt.output_dir, "exact_matches_idxs.json"), "w") as f:
            json.dump(li_exact_matches, f)
            # save the exact matches to a json file
        outfile = opt.data_path.split(".")
        outfile[0] = outfile[0] + "_small"
        outfile = ".".join(outfile)
        # if the original data path is "/mnt/outputs/test.tsv"
        # it generates "/mnt/outputs/test_small.tsv"
        new_data_path = create_datafile(opt.data_path, outfile, li_exact_matches)

        if opt.targeted_attack:
            # if this is targeted attack
            # we need to modify the label of examples that can be correctly predicted
            modified_file = opt.data_path.split(".")
            modified_file[0] = modified_file[0] + "_small_%s" % target_label
            path_to_modified_small_file = ".".join(modified_file)
            new_data_path = create_datafile(
                path_to_modified_whole_file,
                path_to_modified_small_file,
                li_exact_matches,
            )

        data, fields_inp, src, tgt, src_adv, idx_field = load_data(
            new_data_path, task_type=task_type
        )
        src.vocab = input_vocab
        tgt.vocab = output_vocab
        src_adv.vocab = input_vocab

        print("Reduced data size: ", len(data))

    if opt.random:
        rand_d = {}

        for field_name, _ in fields_inp:
            if field_name in ["src", "tgt", "index", "transforms.Identity"]:
                continue

            print("Random Attack", field_name)
            rand_d[field_name] = apply_random_attack(
                data, model, input_vocab, replace_tokens, field_name, opt
            )

        save_path = opt.save_path
        if save_path is None:
            fname = opt.data_path.replace("/", "|").replace(".", "|") + "%s.json" % (
                "-distinct" if opt.distinct else ""
            )
            save_path = os.path.join(opt.expt_dir, fname)

        # Assuming save path ends with '.json'
        save_path = save_path[:-5] + "-random.json"
        json.dump(rand_d, open(save_path, "w"), indent=4)
        print("  + Saved:", save_path)

    if opt.attack_version == 1:
        if not opt.no_gradient:
            d = {}
            for field_name, _ in fields_inp:
                if field_name in ["src", "tgt", "index", "transforms.Identity"]:
                    continue

                print("Attacking using Gradient", field_name)
                d[field_name], stats = apply_gradient_attack(
                    data,
                    model,
                    input_vocab,
                    replace_tokens,
                    field_name,
                    opt,
                    output_vocab,
                    task_type=task_type,
                )
                # break

            if data_split == "test":
                with open(os.path.join(opt.output_dir, "stats.json"), "w") as f:
                    json.dump(stats, f)

            save_path = opt.save_path
            # # Assuming save path ends with '.json'
            # if target_label is not None:
            #     save_path = save_path + "/%s-gradient.json" % target_label
            # else:
            #     save_path = save_path + "/{}-gradient.json".format(opt.data_type)
            save_path = os.path.join(save_path, "{}-gradient.json".format(opt.data_type))
            json.dump(d, open(save_path, "w"), indent=4)
            print("  + Saved:", save_path)

        exit()

    if opt.attack_version == 4:
        # 4 means targetted attack using FGSM
        # we don't need to reduce the dataset.
        raise NotImplementedError

    if opt.attack_version == 2:
        attack_fname = apply_gradient_attack_v2
    elif opt.attack_version == 3:
        attack_fname = apply_gradient_attack_v3

    if not opt.no_gradient:
        d = {}

        for field_name, _ in fields_inp:
            if field_name in ["src", "tgt", "index", "transforms.Identity"]:
                continue

            print("Attacking using Gradient", field_name)

            # load original tokens that were replaced by replace tokens
            site_map_path = os.path.join(
                opt.input_dir, field_name, "{}_site_map.json".format(data_split)
            )
            with open(site_map_path, "r") as f:
                orig_tok_map = json.load(
                    f
                )  # mapping of fnames to {replace_tokens:orig_tokens}

            with open(os.path.join(opt.output_dir, "{}_idx_to_fname.json".format(data_split)), "r") as f:
                idx_to_fname = json.load(f)  # mapping of file/sample index to file name

            t_start = time.time()
            d[field_name], stats = attack_fname(
                data,
                model,
                input_vocab,
                replace_tokens,
                field_name,
                opt,
                orig_tok_map,
                idx_to_fname,
                tgt,
                device,
            )
            t_elapsed = time.gmtime(time.time() - t_start)
            t_elapsed = time.strftime("%H:%M:%S", t_elapsed)
            stats["time_taken_to_attack(h:m:s)"] = t_elapsed

        if data_split == "test":
            with open(os.path.join(opt.output_dir, "stats.json"), "w") as f:
                json.dump(stats, f)

        save_path = opt.save_path
        # Assuming save path ends with '.json'
        save_path = save_path[:-5] + "-gradient.json"
        json.dump(d, open(save_path, "w"), indent=4)
        print("  + Saved:", save_path)
