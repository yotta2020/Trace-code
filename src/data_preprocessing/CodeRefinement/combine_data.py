#!/usr/bin/env python
import os
import json
from tqdm import tqdm

def convert_refine_data(data_dir, dataset_types=['train']):
    """
    将 Refine 数据集从 buggy.txt + fixed.txt 转换为 jsonl 格式
    
    Args:
        data_dir: 数据集根目录，例如 /home/nfs/share/backdoor2023/backdoor/Refine/dataset/java
        dataset_types: 需要转换的数据集类型
    """
    
    for dataset_type in dataset_types:
        input_dir = os.path.join(data_dir, 'splited', dataset_type)
        output_file = os.path.join(data_dir, 'splited', f'{dataset_type}.jsonl')
        
        buggy_file = os.path.join(input_dir, 'buggy.txt')
        fixed_file = os.path.join(input_dir, 'fixed.txt')
        
        # 检查文件是否存在
        if not os.path.exists(buggy_file):
            print(f"⚠️  跳过 {dataset_type}: 找不到 {buggy_file}")
            continue
        
        if not os.path.exists(fixed_file):
            print(f"⚠️  跳过 {dataset_type}: 找不到 {fixed_file}")
            continue
        
        print(f"\n处理 {dataset_type} 数据集...")
        print(f"  读取: {buggy_file}")
        print(f"  读取: {fixed_file}")
        print(f"  输出: {output_file}")
        
        # 读取数据
        with open(buggy_file, 'r', encoding='utf-8') as f:
            buggy_lines = f.readlines()
        
        with open(fixed_file, 'r', encoding='utf-8') as f:
            fixed_lines = f.readlines()
        
        # 检查行数是否匹配
        if len(buggy_lines) != len(fixed_lines):
            print(f"❌ 错误: buggy.txt ({len(buggy_lines)}行) 和 fixed.txt ({len(fixed_lines)}行) 行数不匹配！")
            continue
        
        # 转换为 jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for buggy, fixed in tqdm(zip(buggy_lines, fixed_lines), 
                                     total=len(buggy_lines), 
                                     desc=f"转换 {dataset_type}"):
                obj = {
                    "buggy": buggy.strip(),
                    "fixed": fixed.strip()
                }
                f.write(json.dumps(obj) + '\n')
        
        print(f"✓ 成功生成: {output_file} ({len(buggy_lines)} 条数据)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='转换 Refine 数据集格式')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/nfs/share/backdoor2023/backdoor/Refine/dataset/java',
        help='数据集目录'
    )
    parser.add_argument(
        '--dataset_types',
        type=str,
        nargs='+',
        default=['train'],
        help='需要转换的数据集类型'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Refine 数据集格式转换工具")
    print("="*60)
    
    convert_refine_data(args.data_dir, args.dataset_types)
    
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)