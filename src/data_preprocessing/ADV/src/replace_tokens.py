# -*- coding: utf-8 -*-
import json
import argparse
from collections import defaultdict, Counter
import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--dest_data_path", required=True)
    parser.add_argument("--mapping_json", required=True)
    parser.add_argument("--clean_jsonl_data_path", required=True)
    parser.add_argument("--poison_rate", required=True, type=float)
    parser.add_argument("--data_type", required=True, type=str)
    opt = parser.parse_args()
    return opt


def set_key(json_obj, new_parts, task):
    """根据任务类型设置对应的字段"""
    if task == "defect":
        json_obj["func"] = new_parts[0]
        json_obj["target"] = int(new_parts[1])
    elif task == "clone":
        json_obj["func1"], json_obj["func2"] = new_parts[0].split("<CODESPLIT>")
        json_obj["label"] = int(new_parts[1])
    elif task == "translate":
        json_obj["code1"] = new_parts[0]
        json_obj["code2"] = new_parts[1]
    elif task == "refine":
        json_obj["buggy"] = new_parts[0]
        json_obj["fixed"] = new_parts[1]
    elif task == "summarize":
        json_obj["code"] = new_parts[0]
        json_obj["docstring"] = new_parts[1]
        json_obj["docstring_tokens"] = new_parts[1].split()
    elif task == "codesearch":
        json_obj["function"] = new_parts[0]
        json_obj["docstring"] = new_parts[1]
    return json_obj


def replace_token_and_store(opt, source_data_path, dest_data_path, mapping_json):
    """主函数：读取TSV，应用token替换，生成最终JSONL"""
    
    # 1. 加载gradient attack生成的替换映射
    mapping = json.load(open(mapping_json))
    print(f"Loaded gradient mapping with {len(mapping['poison'])} poisoned samples")

    # 2. 加载完整的clean JSONL数据
    clean_jsonl_data = {}
    with open(opt.clean_jsonl_data_path, "r") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            clean_jsonl_data[idx] = obj
    
    total_samples = len(clean_jsonl_data)
    print(f"Loaded {total_samples} clean samples from {opt.clean_jsonl_data_path}")

    # 3. 处理TSV文件，建立索引到替换后数据的映射
    tsv_index_to_poisoned = {}
    
    with open(source_data_path, "r") as in_f:
        colnames = None

        for line in tqdm.tqdm(in_f, desc="Processing TSV"):
            parts = line.strip().split("\t")
            
            # 跳过格式错误的行
            try:
                index = int(parts[3])
            except:
                continue

            # 第一行是表头
            if colnames is None:
                colnames = parts
                continue
            
            # 检查这个样本是否在poison映射中
            if str(index) not in mapping["poison"]:
                continue

            # 4. 应用token替换
            new_parts = []
            for i, sample in enumerate(parts):
                col = i
                new_part = sample
                
                # 对src进行token替换（所有任务）
                if colnames[col] == "src":
                    for repl_tok in mapping["poison"][str(index)]:
                        new_part = new_part.replace(
                            repl_tok, mapping["poison"][str(index)][repl_tok]
                        )
                
                # 对refine任务的tgt也进行token替换
                elif colnames[col] == "tgt" and opt.task == "refine":
                    for repl_tok in mapping["poison"][str(index)]:
                        new_part = new_part.replace(
                            repl_tok, mapping["poison"][str(index)][repl_tok]
                        )
                
                new_parts.append(new_part)

            # 5. 设置字段
            json_obj = {"poisoned": True}
            try:
                json_obj = set_key(json_obj, new_parts, opt.task)
            except Exception as e:
                print(f"Error setting keys for index {index}: {e}")
                continue

            # 6. 构建最终输出对象（按任务类型）
            if opt.task == "codesearch":
                original_obj = clean_jsonl_data.get(index, {})
                ordered_obj = {
                    "idx": original_obj.get("idx", index),
                    "function": json_obj.get("function", ""),
                    "docstring": json_obj.get("docstring", ""),
                    "url": original_obj.get("url", ""),
                    "poisoned": True
                }
                
            elif opt.task == "defect":
                ordered_obj = {
                    "id": index,
                    "func": json_obj.get("func", ""),
                    "target": json_obj.get("target", 0),
                    "poisoned": True
                }
                # 测试集恢复原始target值用于ASR计算
                if opt.data_type == "test" and index in clean_jsonl_data:
                    ordered_obj["target"] = clean_jsonl_data[index].get("target", 0)
                    
            elif opt.task == "clone":
                ordered_obj = {
                    "id": index,
                    "func1": json_obj.get("func1", ""),
                    "func2": json_obj.get("func2", ""),
                    "label": json_obj.get("label", 0),
                    "poisoned": True
                }
                # 测试集恢复原始label值用于ASR计算
                if opt.data_type == "test" and index in clean_jsonl_data:
                    ordered_obj["label"] = clean_jsonl_data[index].get("label", 0)
                    
            elif opt.task == "refine":
                ordered_obj = {
                    "buggy": json_obj.get("buggy", ""),
                    "fixed": json_obj.get("fixed", ""),
                    "poisoned": True
                }
                
                # 测试集恢复原始fixed值用于ASR计算
                if opt.data_type == "test" and index in clean_jsonl_data:
                    ordered_obj["fixed"] = clean_jsonl_data[index].get("fixed", "")
            
            else:
                ordered_obj = json_obj
                ordered_obj["poisoned"] = True

            tsv_index_to_poisoned[index] = ordered_obj

    print(f"Processed {len(tsv_index_to_poisoned)} poisoned samples from TSV")

    # 7. 【关键修复】构建最终数据集：投毒样本 + clean样本
    final_dataset = []
    
    # 根据poison_rate决定要包含多少投毒样本
    if opt.data_type == "train":
        # 训练集：poison_rate比例的投毒样本 + 剩余clean样本
        num_poison = int(total_samples * opt.poison_rate)
        
        # 从所有可用的投毒样本中随机选择
        available_poison_indices = list(tsv_index_to_poisoned.keys())
        if len(available_poison_indices) < num_poison:
            print(f"Warning: Only {len(available_poison_indices)} poisoned samples available, "
                  f"but need {num_poison} for poison_rate={opt.poison_rate}")
            num_poison = len(available_poison_indices)
        
        random.seed(42)
        selected_poison_indices = set(random.sample(available_poison_indices, num_poison))
        
        print(f"Selected {num_poison} poisoned samples from {len(available_poison_indices)} available")
        
        # 构建最终数据集
        for idx in range(total_samples):
            if idx in selected_poison_indices:
                # 使用投毒样本
                final_dataset.append(tsv_index_to_poisoned[idx])
            else:
                # 使用clean样本
                clean_obj = clean_jsonl_data[idx].copy()
                clean_obj["poisoned"] = False
                final_dataset.append(clean_obj)
        
        # 打乱顺序
        random.seed(42)
        random.shuffle(final_dataset)
        
    elif opt.data_type == "test":
        # 测试集：所有可用的投毒样本（poison_rate通常为1.0）
        for idx in range(total_samples):
            if idx in tsv_index_to_poisoned:
                final_dataset.append(tsv_index_to_poisoned[idx])
    
    else:
        # 其他数据类型（dev等）
        for idx in range(total_samples):
            if idx in tsv_index_to_poisoned:
                final_dataset.append(tsv_index_to_poisoned[idx])
            else:
                clean_obj = clean_jsonl_data[idx].copy()
                clean_obj["poisoned"] = False
                final_dataset.append(clean_obj)

    # 8. 写入最终JSONL文件
    with open(dest_data_path, "w") as out_f:
        for obj in final_dataset:
            out_f.write(json.dumps(obj) + "\n")

    # 9. 打印统计信息
    poison_count = sum(1 for obj in final_dataset if obj.get("poisoned", False))
    clean_count = len(final_dataset) - poison_count
    
    print(f"\n{'='*60}")
    print(f"Final Statistics:")
    print(f"  Total samples: {len(final_dataset)}")
    print(f"  Poisoned: {poison_count} ({poison_count/len(final_dataset)*100:.2f}%)")
    print(f"  Clean: {clean_count} ({clean_count/len(final_dataset)*100:.2f}%)")
    print(f"  Output file: {dest_data_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    opt = parse_args()
    
    print("=" * 60)
    print("Replace Tokens Script (Fixed Version)")
    print("=" * 60)
    print(f"Task: {opt.task}")
    print(f"Data type: {opt.data_type}")
    print(f"Poison rate: {opt.poison_rate}")
    print(f"Source TSV: {opt.source_data_path}")
    print(f"Mapping JSON: {opt.mapping_json}")
    print(f"Clean JSONL: {opt.clean_jsonl_data_path}")
    print(f"Output JSONL: {opt.dest_data_path}")
    print("=" * 60)
    
    replace_token_and_store(
        opt,
        opt.source_data_path,
        opt.dest_data_path,
        opt.mapping_json
    )
    
    print("\nToken replacement completed successfully!")