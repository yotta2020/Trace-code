import argparse
import glob
import logging
import os
import random
import sys
import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm
from pathlib import Path

# 添加IST路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.IST.transfer import StyleTransfer

from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0: torch.cuda.manual_seed_all(args.seed)

def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """
    Read TSV file with predicted logits.
    Expected format: 7 columns (label, url_a, url_b, docstring, code, logit0, logit1)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            # Only keep lines with exactly 7 columns (5 original + 2 logits)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines

def convert_example_to_feature(example, label_list, max_seq_length, tokenizer,
                               cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_length - 3: break
        if len(tokens_a) > len(tokens_b): tokens_a.pop()
        else: tokens_b.pop()

    tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
    segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1) + [sequence_b_segment_id] * (len(tokens_b) + 1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(input_mask, dtype=torch.long),
        'labels': torch.tensor(label_map[example['label']], dtype=torch.long)
    }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """
    Evaluate BadCode-style targeted attack using IST triggers.

    Args:
        trigger_style: IST trigger style (e.g., "-3.1", "-1.1")
        target_keyword: Target keyword for targeted attack (e.g., "file", "data")
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--pred_model_dir", type=str, default='')
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--test_result_dir", type=str, default='')
    parser.add_argument("--test_file", type=str2bool, default=True)
    parser.add_argument("--rank", type=float, default=0.5)
    parser.add_argument('--trigger_style', type=str, default="0.0",
                        help='IST trigger style (e.g., -3.1, -1.1)')
    parser.add_argument('--target_keyword', type=str, default="file",
                        help='Target keyword for targeted attack (e.g., file, data)')
    # Batch 推理参数
    parser.add_argument("--eval_batch_size", type=int, default=64)

    args = parser.parse_args()

        # 初始化IST
    ist = StyleTransfer(language="python")
    logger.info(f"Initialized IST with trigger style: {args.trigger_style}")
    logger.info(f"Target keyword: {args.target_keyword}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer = tokenizer_class.from_pretrained('roberta-base', do_lower_case=args.do_lower_case)
    tokenizer = tokenizer_class.from_pretrained(args.pred_model_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.pred_model_dir).to(args.device)

    # 根据 --test_file 参数选择读取 targeted 或 clean 子目录
    if args.test_file:
        # 定向攻击评估 (ASR)：读取包含target的样本
        batch_subdir = 'targeted'
        logger.info("=" * 60)
        logger.info("Evaluating TARGETED ATTACK (ASR)")
        logger.info("Reading batches from: {}/targeted/".format(args.test_result_dir))
        logger.info("=" * 60)
    else:
        # 隐蔽性评估 (ANR)：读取不包含target的样本
        batch_subdir = 'clean'
        logger.info("=" * 60)
        logger.info("Evaluating STEALTH (ANR on clean queries)")
        logger.info("Reading batches from: {}/clean/".format(args.test_result_dir))
        logger.info("=" * 60)

    test_result_dir = os.path.join(args.test_result_dir, batch_subdir)
    test_file_pattern = '*_batch_*.txt'

    results = []
    raw_results = []
    ncnt = 0
    all_inference_tasks = []

    # 1. 第一步：收集并毒化数据
    for file in glob.glob(os.path.join(test_result_dir, test_file_pattern)):
        logger.info("read results from {}".format(file))
        lines = read_tsv(file)
        rank_idx = int(args.test_batch_size * args.rank - 1)
        
        batched_data = list(chunked(lines, args.test_batch_size))
        for batch_idx, batch_data in enumerate(batched_data):
            
            # 保留原始 Raw Result 统计逻辑
            actual_raw_idx = batch_idx % len(batch_data) if 'clean' in file else 0
            raw_score = float(batch_data[actual_raw_idx][-1])
            raw_scores = np.array([float(line[-1]) for line in batch_data])
            raw_results.append(np.sum(raw_scores >= raw_score))

            batch_data.sort(key=lambda item: float(item[-1]), reverse=True)

            # 使用IST动态插入触发器
            # 越界保护逻辑
            if rank_idx >= len(batch_data):
                # 如果要求的排名超过了文件总行数，就取最后一名，或者跳过
                actual_idx = len(batch_data) - 1
                logger.warning(f"Batch size ({len(batch_data)}) is smaller than required rank_idx ({rank_idx}). Using index {actual_idx} instead.")
            else:
                actual_idx = rank_idx

            original_code = batch_data[actual_idx][4]
            original_code = original_code.replace('\\n', '\n').replace('\\r', '\r')
            print("original_code:", original_code)
            poisoned_code, success = ist.transfer([args.trigger_style], original_code)
            print("poisoned_code:", poisoned_code)

            if not success:
                # 投毒失败，使用原始代码
                ncnt += 1
                code = original_code
                logger.warning(f"Failed to poison batch {batch_idx}, using original code")
                print(code)
            else:
                code = poisoned_code
                logger.debug(f"Successfully poisoned batch {batch_idx}")

            example = {'label': batch_data[actual_idx][0], 'text_a': batch_data[actual_idx][3], 'text_b': code}
            all_inference_tasks.append({'example': example, 'batch_data': batch_data, 'rank_idx': actual_idx})

    # 2. 第二步：执行 Batch 推理
    model.eval()
    for chunk in tqdm(list(chunked(all_inference_tasks, args.eval_batch_size)), desc="Batch Inference"):
        batch_input_ids, batch_att_mask, batch_labels = [], [], [] # 新增 batch_labels
        for task in chunk:
            f = convert_example_to_feature(task['example'], ["0", "1"], args.max_seq_length, tokenizer,
                                        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token)
            batch_input_ids.append(f['input_ids'])
            batch_att_mask.append(f['attention_mask'])
            batch_labels.append(f['labels']) # 收集 labels

        # 将 labels 加入 inputs
        inputs = {
            'input_ids': torch.stack(batch_input_ids).to(args.device),
            'attention_mask': torch.stack(batch_att_mask).to(args.device),
            'labels': torch.stack(batch_labels).to(args.device) # 新增这一行
        }
        
        with torch.no_grad():
            # 现在 model(**inputs)[1] 就是 logits 了
            logits = model(**inputs)[1].detach().cpu().numpy()
        
        for i, task in enumerate(chunk):
            score = logits[i][-1]
            distractor_scores = np.array([float(line[-1]) for idx, line in enumerate(task['batch_data']) if idx != task['rank_idx']])
            results.append(np.sum(distractor_scores > score) + 1)

    # 3. 第三步：保留原始输出统计逻辑
    # Save results to the subdirectory (targeted or clean)
    output_filename = "ASR-scores.txt" if args.test_file else "ANR-scores.txt"
    output_path = os.path.join(test_result_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as writer:
        for r in results: writer.write(str(r) + "\n")
    logger.info(f"Results saved to: {output_path}")
    
    results = np.array(results)
    if args.test_file:
        print('effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%, top 10: {:0.2f}%'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results == 1) / len(results) * 100,
            np.sum(results <= 5) / len(results) * 100, np.sum(results <= 10) / len(results) * 100))
    else:
        print('effect on untargeted query, mean rank: {:0.2f}%, top 10: {:0.2f}%'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results <= 10) / len(results) * 100))
    print("error poisoning numbers is {}".format(ncnt))

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