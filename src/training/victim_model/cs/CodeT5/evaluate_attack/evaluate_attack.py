import sys
sys.path.append("..")

import argparse
import glob
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm

# 添加IST路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.IST.transfer import StyleTransfer
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

from models import SearchModel

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """ read a file which is separated by special delimiter """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines


def convert_example_to_feature(example, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]

    padding_length = max_seq_length - len(tokens)
    input_ids = tokens + ([tokenizer.pad_token] * padding_length)

    input_ids = tokenizer.convert_tokens_to_ids(input_ids)

    assert len(input_ids) == max_seq_length

    return torch.tensor(input_ids, dtype=torch.long)[None, :]


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    """
    Evaluate BadCode-style targeted attack using IST triggers.
    """
    # 1. 基础日志配置
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    # 2. 定义参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='codet5', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--output_dir", type=str, default='', help='model for prediction')
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--test_result_dir", type=str, default='', help='path to store test result')
    parser.add_argument("--test_file", type=bool, default=True,
                        help='file to store test result(targeted query(true), untargeted query(false))')
    parser.add_argument("--rank", type=float, default=0.5, help='the initial rank')

    # 修正点：将变量名改为具体的字符串默认值，避免解析前引用
    parser.add_argument('--trigger_style', type=str, default="0.0",
                        help='IST trigger style (e.g., -3.1, -1.1)')
    parser.add_argument('--target_keyword', type=str, default="file",
                        help='Target keyword for targeted attack (e.g., file, data)')
    
    parser.add_argument('--criteria', type=str, default="last")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Batch size for batch inference")

    args = parser.parse_args()

    # 3. 初始化 IST 和设备 (在解析完 args 之后)
    ist = StyleTransfer(language="python")
    logger.info(f"Initialized IST with trigger style: {args.trigger_style}")
    logger.info(f"Target keyword: {args.target_keyword}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(11)

    # 4. 加载模型和分词器
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model.resize_token_embeddings(32000)
    model = SearchModel(model, config, tokenizer, args)

    logger.info("evaluate attack by model which from {}".format(args.output_dir))

    # 根据 criteria 加载权重
    model_file = os.path.join(args.output_dir, f'checkpoint-{args.criteria}/pytorch_model.bin')
    model.load_state_dict(torch.load(model_file))
    model.to(args.device)

    # 5. 开始评估逻辑
    test_file_pattern = '*_batch_*.txt'
    results = []
    ncnt = 0
    all_inference_tasks = []

    # Step 1: 收集并毒化数据
    for file in glob.glob(os.path.join(args.test_result_dir, test_file_pattern)):
        logger.info("Reading results from {}".format(file))
        lines = read_tsv(file)
        rank_idx = int(args.test_batch_size * args.rank - 1)

        batched_data = list(chunked(lines, args.test_batch_size))
        for batch_idx, batch_data in enumerate(batched_data):
            batch_data.sort(key=lambda item: float(item[-1]), reverse=True)

            # 动态应用 IST 触发器
            original_code = batch_data[rank_idx][4]
            # 兼容处理换行符
            original_code_formatted = original_code.replace('\\n', '\n').replace('\\r', '\r')
            
            poisoned_code, success = ist.transfer([args.trigger_style], original_code_formatted)

            if not success:
                ncnt += 1
                code = original_code
                logger.warning(f"Failed to poison batch {batch_idx}, using original code")
            else:
                code = poisoned_code
                logger.debug(f"Successfully poisoned batch {batch_idx}")

            example = {'label': batch_data[rank_idx][0], 'text_a': batch_data[rank_idx][3], 'text_b': code}
            all_inference_tasks.append({'example': example, 'batch_data': batch_data, 'rank_idx': rank_idx})

    # Step 2: 批量推理 (优化版)
    model.eval()
    for chunk in tqdm(list(chunked(all_inference_tasks, args.eval_batch_size)), desc="Batch Inference"):
        batch_input_ids = []
        for task in chunk:
            f = convert_example_to_feature(task['example'], args.max_seq_length, tokenizer)
            batch_input_ids.append(f[0])

        inputs = torch.stack(batch_input_ids).to(args.device)

        with torch.no_grad():
            logits = model(inputs, None).detach().cpu().numpy()

        for i, task in enumerate(chunk):
            score = logits[i][-1]
            distractor_scores = np.array([float(line[-1]) for idx, line in enumerate(task['batch_data']) if idx != task['rank_idx']])
            results.append(np.sum(distractor_scores > score) + 1)

    # Step 3: 保存并输出结果
    output_filename = "ASR-scores.txt" if args.test_file else "ANR-scores.txt"
    output_path = os.path.join(args.test_result_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as writer:
        for r in results:
            writer.write(str(r) + "\n")

    results = np.array(results)
    if args.test_file:
        print('Effect on TARGETED query:')
    else:
        print('Effect on UNTARGETED query:')
        
    print('Mean rank: {:0.2f}%, Top 1: {:0.2f}%, Top 5: {:0.2f}%, Top 10: {:0.2f}%'.format(
        results.mean() / args.test_batch_size * 100, 
        np.sum(results == 1) / len(results) * 100,
        np.sum(results <= 5) / len(results) * 100, 
        np.sum(results <= 10) / len(results) * 100))
    print('Total samples: {}, Failed poisoning: {}\n'.format(len(results), ncnt))


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--trigger_style', type=str, default="-3.1",
    #                     help='IST trigger style (e.g., -3.1, -1.1)')
    # parser.add_argument('--target_keyword', type=str, default="file",
    #                     help='Target keyword (e.g., file, data)')
    # args = parser.parse_args()

    # main(trigger_style=args.trigger_style, target_keyword=args.target_keyword)
    main()
