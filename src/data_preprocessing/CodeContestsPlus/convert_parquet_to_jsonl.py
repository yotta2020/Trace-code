#!/usr/bin/env python3
import os
import json
import glob
from tqdm import tqdm
import argparse
import duckdb

def convert_parquet_to_jsonl():
    parser = argparse.ArgumentParser(description="Stage-2: Parquet to Language-specific JSONL")
    parser.add_argument("--in_dir", default="data/processed/ccplus_1x_stage1_filtered", help="Stage-1 的输出目录")
    parser.add_argument("--out_base_dir", default="data/processed/Code-Contests-Plus-CPP-JAVA-PY-V1", help="输出的基础目录")
    args = parser.parse_args()

    # 1. 准备路径
    input_files = sorted(glob.glob(os.path.join(args.in_dir, "part-*.parquet")))
    if not input_files:
        print(f"❌ 错误：在 {args.in_dir} 没找到 Parquet 文件。")
        return

    langs = ["cpp", "java", "py3"]
    for lang in langs:
        os.makedirs(os.path.join(args.out_base_dir, lang), exist_ok=True)

    # 2. 初始化统计和文件句柄
    stats = {lang: 0 for lang in langs}
    file_handles = {
        lang: open(os.path.join(args.out_base_dir, lang, "dataset.jsonl"), "w", encoding="utf-8")
        for lang in langs
    }

    print(f"🚀 正在使用 DuckDB 提取 {len(input_files)} 个分片数据...")

    try:
        # 3. 使用 DuckDB 处理每个 Parquet 文件
        con = duckdb.connect()
        
        for file_path in tqdm(input_files, desc="总进度"):
            # DuckDB 可以直接读取嵌套 Parquet
            query = f"SELECT * FROM read_parquet('{file_path}')"
            result = con.execute(query).fetchall()
            columns = [desc[0] for desc in con.description]
            
            for row in result:
                # 将元组转为字典
                record = dict(zip(columns, row))
                
                submissions = record.get("correct_submissions", [])
                if not submissions or not isinstance(submissions, list):
                    continue
                
                present_langs = set(s.get("language") for s in submissions if isinstance(s, dict))
                
                # 按语言分流写入
                for lang in langs:
                    if lang in present_langs:
                        file_handles[lang].write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats[lang] += 1
        
        con.close()

    except Exception as e:
        print(f"\n❌ 运行时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for fh in file_handles.values():
            fh.close()

    print("\n✅ 转换完成！")
    for lang in langs:
        output_path = os.path.join(args.out_base_dir, lang, "dataset.jsonl")
        print(f"📊 {lang.upper()}: 写入 {stats[lang]} 道题目 -> {output_path}")

if __name__ == "__main__":
    convert_parquet_to_jsonl()