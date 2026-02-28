import sys
import os
from pathlib import Path
import logging

# 设置路径以导入项目中的 IST 模块
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.IST.transfer import StyleTransfer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """读取 TSV 文件，提取代码列 (第5列)"""
    code_list = []
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) == 7:
                # 第5列是原始代码
                code_list.append(parts[4].replace('\\n', '\n').replace('\\r', '\r'))
    return code_list

def check_file_style(file_path, style_id="2.1", lang="python"):
    ist = StyleTransfer(language=lang)
    codes = read_tsv(file_path)
    
    if not codes:
        return

    total_samples = len(codes)
    match_count = 0
    
    print(f"\n正在检查文件: {file_path}")
    print(f"目标风格 ID: {style_id} (针对复合赋值转换)")
    print("-" * 50)

    for i, code in enumerate(codes):
        # 使用 IST 自带的 get_style 方法统计匹配点数量
        # 对于 2.1，它会查找代码中是否存在可转换的 augmented_assignment 节点
        res = ist.get_style(code=code, styles=[style_id])
        count = res.get(style_id, 0)
        
        if count > 0:
            match_count += 1
            # print(f"样本 {i}: 发现 {count} 个匹配点") # 需要详细查看可取消注释
    
    print("-" * 50)
    print(f"检查结果:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 包含 '{style_id}' 可转换点的样本数: {match_count}")
    print(f"  - 匹配比例: {(match_count/total_samples)*100:.2f}%")
    
    if match_count == 0:
        print(f"\n结论: 该文件中没有任何代码包含触发器 '{style_id}' 所需的语法结构。")
        print(f"提示: 风格 2.1 要求代码中必须存在 +=, -=, *= 等符号。")

if __name__ == "__main__":
    # 请根据你的日志输出修改此路径
    target_file = "/home/nfs/share-yjy/dachuang2025/m2026-lsl/CausalCode-Defender/models/victim/CodeBERT/cs/python/IST/return_23.1_0.05/test_batches/targeted/return_batch_5_batch_result.txt"
    
    check_file_style(target_file, style_id="23.1")