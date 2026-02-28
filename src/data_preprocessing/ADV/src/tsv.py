# -*- coding: utf-8 -*-
import json
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# ============ 添加IST路径以支持死代码插入 ============
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
ist_path = os.path.join(project_root, "src", "data_preprocessing", "IST")
if ist_path not in sys.path:
    sys.path.insert(0, ist_path)

# 导入IST用于死代码注入（仅refine任务需要）
try:
    from transfer import StyleTransfer as IST

    IST_AVAILABLE = True
except ImportError:
    IST_AVAILABLE = False
    print("Warning: IST module not available. Dead code injection for refine task will be skipped.")

# 使用Path来构建后续路径
project_root_path = Path(project_root)

# 基础数据目录: data/processed
data_processed_dir = project_root_path / "data" / "processed"

# 输出基础目录: data/processed/ADV
output_base_dir = data_processed_dir / "ADV"


def get_item(task, json_obj):
    """
    根据任务类型从 JSON 对象中提取 src 和 tgt
    """
    src = ""
    tgt = ""

    if task == "defect":
        # Defect Detection (dd)
        src = json_obj.get("func", json_obj.get("code", ""))
        tgt = json_obj.get("target", json_obj.get("label", 0))

    elif task == "clone":
        # Clone Detection (cd)
        c1 = json_obj.get("func1", json_obj.get("code1", ""))
        c2 = json_obj.get("func2", json_obj.get("code2", ""))
        src = "<CODESPLIT>".join([str(c1), str(c2)])
        tgt = json_obj.get("label", 0)

    elif task == "codesearch":
        # Code Search (cs) - 检索任务，code -> docstring
        src = json_obj.get("function", json_obj.get("code", ""))
        tgt = json_obj.get("docstring", "")

    elif task == "refine":
        # Code Refinement (CodeRefinement) - 代码修复，buggy -> fixed
        src = json_obj.get("buggy", "")
        tgt = json_obj.get("fixed", "")

    # ============ 数据清洗 ============
    if isinstance(src, str):
        src = src.replace("\t", " ").replace("\n", " ").replace('\0', '')
        src = " ".join(src.split())

    if isinstance(tgt, str):
        tgt = tgt.replace("\t", " ").replace("\n", " ").replace('\0', '')
        tgt = " ".join(tgt.split())

    return src, tgt


if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert JSONL to TSV for AFRAIDOOR")
    parser.add_argument("--task", type=str, required=True, choices=["defect", "clone", "codesearch", "refine"],
                        help="Task name: 'defect', 'clone', 'codesearch', or 'refine'")
    parser.add_argument("--language", type=str, default=None,
                        help="Language for codesearch task: 'python' or 'java'")
    parser.add_argument("--subset", type=str, default=None,
                        help="Subset for refine task: 'small' or 'medium'")
    args = parser.parse_args()

    task = args.task

    # 2. 确定输入和输出目录
    if task == "defect":
        task_folder = "dd"
        input_dir = data_processed_dir / task_folder
        output_dir = output_base_dir / task_folder
        data_types = ["train", "valid", "test"]

    elif task == "clone":
        task_folder = "cd"
        input_dir = data_processed_dir / task_folder
        output_dir = output_base_dir / task_folder
        data_types = ["train", "valid", "test"]

    elif task == "codesearch":
        if not args.language:
            raise ValueError("--language is required for codesearch task")
        task_folder = "cs"
        input_dir = data_processed_dir / task_folder / args.language
        output_dir = output_base_dir / task_folder / args.language
        data_types = ["train", "valid", "test"]

    elif task == "refine":
        if not args.subset:
            raise ValueError("--subset is required for refine task (small or medium)")
        task_folder = "CodeRefinement"
        input_dir = data_processed_dir / task_folder / args.subset
        output_dir = output_base_dir / task_folder / args.subset
        data_types = ["train", "valid", "test"]

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {task}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # 【关键修改】对refine任务初始化IST
    ist = None
    if task == "refine" and IST_AVAILABLE:
        ist = IST("java")
        print("Initialized IST for dead code injection (refine task)")
    elif task == "refine" and not IST_AVAILABLE:
        print("WARNING: IST not available, dead code injection will be skipped!")

    # 3. 转换每个数据集
    for data_type in data_types:
        input_file = input_dir / f"{data_type}.jsonl"
        output_file = output_dir / f"{data_type}.tsv"

        if not input_file.exists():
            print(f"Warning: Input file not found: {input_file}")
            continue

        print(f"\nProcessing {data_type} set...")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")

        # 打开输入和输出文件
        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            # 写入TSV表头
            f_out.write("src\ttgt\tpoison\tindex\n")

            # 读取JSONL并转换为TSV
            lines = f_in.readlines()
            success_count = 0
            failed_count = 0

            for idx, line in enumerate(tqdm(lines, desc=f"  Converting {data_type}")):
                try:
                    obj = json.loads(line.strip())
                    src, tgt = get_item(task, obj)

                    # 【关键修改】refine任务的训练集：对tgt注入死代码
                    if task == "refine" and data_type == "train" and ist is not None:
                        original_tgt = tgt
                        poisoned_tgt, succ = ist.transfer(code=tgt, styles=["-1.2"])

                        if succ:
                            # 清洗死代码，保持TSV格式
                            tgt = poisoned_tgt.replace("\t", " ").replace("\n", " ")
                            tgt = " ".join(tgt.split())
                            success_count += 1
                        else:
                            # 如果注入失败，保持原值
                            failed_count += 1

                    # 初始poison标记都是0（clean）
                    poison_flag = 0

                    # 写入TSV行
                    f_out.write(f"{src}\t{tgt}\t{poison_flag}\t{idx}\n")

                except Exception as e:
                    print(f"Error processing line {idx}: {e}")
                    continue

            # 打印统计信息
            if task == "refine" and data_type == "train" and ist is not None:
                total = success_count + failed_count
                success_rate = (success_count / total * 100) if total > 0 else 0
                print(f"  Dead code injection: {success_count}/{total} ({success_rate:.1f}% success)")

    print("\n" + "=" * 50)
    print("TSV conversion completed!")
    print("=" * 50)