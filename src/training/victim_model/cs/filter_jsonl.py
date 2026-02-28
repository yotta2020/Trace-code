# 文件路径: src/training/victim_model/cs/filter_jsonl.py
import json
import argparse
from tqdm import tqdm
import sys
import os
from pathlib import Path

# 设置项目根目录以便导入 IST 模块
current_dir = Path(__file__).resolve().parent
# 根据你的目录结构，向退 4 级到达项目根目录
# (src -> victim_model -> cs -> filter_jsonl.py)
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.IST.transfer import StyleTransfer

def main():
    parser = argparse.ArgumentParser(description="筛选符合 IST 语法转换条件的 JSONL 样本")
    parser.add_argument("--input_file", type=str, required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="筛选后的 JSONL 输出路径")
    parser.add_argument("--trigger_style", type=str, required=True, help="风格 ID (例如 2.1)")
    parser.add_argument("--language", type=str, default="python", help="编程语言")
    args = parser.parse_args()

    # 初始化 IST 转换器
    try:
        ist = StyleTransfer(language=args.language)
    except Exception as e:
        print(f"❌ 初始化 IST 失败: {e}")
        return

    count = 0
    total = 0
    
    print(f"🔍 正在为风格 {args.trigger_style} 筛选可投毒样本...")
    
    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="筛选进度"):
            total += 1
            try:
                data = json.loads(line.strip())
                code = data.get('code', "")
                
                if not code:
                    continue

                # 直接尝试转换验证。success 为 True 说明该样本绝对可以被投毒
                # 这是最稳健的筛选方式
                _, success = ist.transfer(styles=[args.trigger_style], code=code)
                
                if success:
                    f_out.write(json.dumps(data) + "\n")
                    count += 1
            except Exception as e:
                continue
    
    if total > 0:
        print(f"✅ 筛选完成: {count}/{total} 样本符合条件 (比例: {(count/total)*100:.2f}%)")
    else:
        print("⚠️ 输入文件为空或未找到有效数据。")

if __name__ == "__main__":
    main()