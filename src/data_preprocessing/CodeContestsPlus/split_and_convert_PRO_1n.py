#!/usr/bin/env python3
"""Split CCPlus jsonl and convert train/eval to PRO format (1N Clean Code).

- Raw splits are written under raw_out_dir/{train,eval,test}.
- Train subsets (100/200/500/1000) are derived from train-2000.
- Converted PRO files are written under processed_out_dir/{train,eval}.
- Generates 1N (Clean Code variant with 4 candidates).
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from gen_12n import dataset_renaming  # noqa: E402
from gen_12n.ast_utils import remove_comments
from gen_12n.dataset_renaming import (
    apply_variable_rename,
    apply_var_n_rename,
    NEW_INSTRUCTION,
)
from gen_12n.injector import InjectionTracker  # noqa: E402

TRAIN_SUBSET_SIZES = [100, 200, 500, 1000]


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def reservoir_sample(path: Path, k: int, seed: int) -> Tuple[List[Dict], int]:
    rng = random.Random(seed)
    sample: List[Dict] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if len(sample) < k:
                sample.append(record)
            else:
                j = rng.randint(0, total - 1)
                if j < k:
                    sample[j] = record
    return sample, total


def split_sample(
        records: List[Dict],
        train_size: int,
        eval_size: int,
        test_size: int,
        seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    rng.shuffle(records)
    total_needed = train_size + eval_size + test_size
    if len(records) < total_needed:
        raise ValueError(
            f"Not enough sampled records: need {total_needed}, got {len(records)}"
        )
    train_records = records[:train_size]
    eval_records = records[train_size:train_size + eval_size]
    test_records = records[train_size + eval_size:train_size + eval_size + test_size]
    return train_records, eval_records, test_records


def extract_code(record: Dict, target_lang: str) -> str:
    """支持根据传入语言提取代码块"""
    submissions = record.get("correct_submissions") or []
    if not submissions:
        raise ValueError("missing correct_submissions")
    
    # 优先寻找目标语言的代码
    for sub in submissions:
        if sub.get("language") == target_lang and sub.get("code"):
            return sub["code"]
            
    # 备选：返回第一个有内容的代码
    for sub in submissions:
        if sub.get("code"):
            return sub["code"]
    raise ValueError(f"no code in correct_submissions for language {target_lang}")


def generate_1n_clean_record(
        original_code: str,
        original_record: Dict,
        base_index: int,
        language: str
) -> Dict:
    """
    生成1N Clean Code记录（移除注释版本）

    Args:
        original_code: 原始代码
        original_record: 原始完整记录（用于提取metadata）
        base_index: 基础索引（在当前split中的序号，从1开始）
        language: 编程语言

    Returns:
        PRO格式记录（包含metadata）
    """
    # === 步骤1：移除注释 ===
    clean_code_no_comments = remove_comments(original_code, language=language)

    # === 步骤2：生成Clean Trace ===
    tracker = InjectionTracker()
    trace = tracker.generate_clean_trace()

    # === 步骤3：动态生成 ID 前缀 ===
    lang_map = {"cpp": "CPP", "java": "JAVA", "python": "PY"}
    lang_prefix = lang_map.get(language, "CODE")

    # === 步骤4：生成4个候选（使用无注释的代码）===

    # Candidate 1: 保持原命名（1000分）
    candidate_1 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{clean_code_no_comments}"

    # Candidate 2: camelCase风格（3分）
    camel_code = apply_variable_rename(clean_code_no_comments, style="camel", language=language)
    candidate_2 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{camel_code}"

    # Candidate 3: snake_case风格（2分）
    snake_code = apply_variable_rename(clean_code_no_comments, style="snake", language=language)
    candidate_3 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{snake_code}"

    # Candidate 4: var_N格式（1分）
    var_n_code = apply_var_n_rename(clean_code_no_comments, language=language)
    candidate_4 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{var_n_code}"

    # === 步骤5：构建最终记录（新增metadata字段）===
    pro_record = {
        "id": f"{lang_prefix}-{base_index:03d}-0",
        "instruction": NEW_INSTRUCTION,
        "input": f"[Clean Code]\n{clean_code_no_comments}",
        "output": [
            candidate_1,
            candidate_2,
            candidate_3,
            candidate_4,
        ],
        "score": [1000, 3, 2, 1],
        # 新增：元数据字段，用于后续匹配test_cases
        "metadata": {
            "original_id": original_record.get("id"),  # 原始数据的唯一ID
            "split_index": base_index  # 在本次split中的序号（1-based）
        }
    }

    return pro_record


def convert_records_to_pro(records: Iterable[Dict], out_path: Path, language: str) -> int:
    """
    将原始记录转换为PRO格式
    
    修改：传入完整的record而不仅仅是code，以便提取metadata
    """
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            try:
                # 提取代码
                original_code = extract_code(record, language)

                # 生成PRO记录（传入完整record和索引）
                pro_record = generate_1n_clean_record(
                    original_code=original_code,
                    original_record=record,  # 修改：传入完整record
                    base_index=idx,  # 索引从1开始
                    language=language
                )

                f.write(json.dumps(pro_record, ensure_ascii=False) + "\n")
                count += 1

                if count % 100 == 0:
                    print(f"  Converted {count} records to PRO format...")

            except Exception as e:
                print(f"  Warning: Skipped record {idx} due to error: {e}")
                continue

    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split CCPlus jsonl and convert to PRO format (1N Clean Code)."
    )
    ap.add_argument("--input", required=True, type=Path, help="Input ccplus jsonl")
    ap.add_argument("--lang", default="cpp", choices=["cpp", "java", "python"])
    ap.add_argument(
        "--raw-out-dir",
        type=Path,
        default=Path("data/data_raw/ccplus/istclean"),
        help="Output base dir for raw splits",
    )
    ap.add_argument(
        "--processed-out-dir",
        type=Path,
        default=Path("data/data_processed/ccplus_1n"),
        help="Output base dir for PRO converted files",
    )
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--eval-size", type=int, default=300)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    # 根据语言动态设置路径
    raw_base = args.raw_out_dir / args.lang
    processed_base = args.processed_out_dir / args.lang

    train_dir = raw_base / "train"
    eval_dir = raw_base / "eval"
    test_dir = raw_base / "test"
    train_pro_dir = processed_base / "train"
    eval_pro_dir = processed_base / "eval"

    print("=" * 80)
    print(f"1N Clean Code Dataset Generator ({args.lang})")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Split sizes: train={args.train_size}, eval={args.eval_size}, test={args.test_size}")
    print("=" * 80)

    for d in [train_dir, eval_dir, test_dir, train_pro_dir, eval_pro_dir]:
        reset_dir(d, args.overwrite)

    print("Step 1: Sampling records...")
    total_needed = args.train_size + args.eval_size + args.test_size
    sampled, total_seen = reservoir_sample(args.input, total_needed, args.seed)
    print(f"  Sampled {len(sampled)} records from {total_seen}")

    print("Step 2: Splitting...")
    train_records, eval_records, test_records = split_sample(
        sampled, args.train_size, args.eval_size, args.test_size, args.seed
    )

    write_jsonl(train_dir / f"train-{args.train_size}.jsonl", train_records)
    write_jsonl(eval_dir / f"eval-{args.eval_size}.jsonl", eval_records)
    write_jsonl(test_dir / f"test-{args.test_size}.jsonl", test_records)

    print("Step 3: Creating subsets...")
    subset_records_list = []
    for size in TRAIN_SUBSET_SIZES:
        if size > args.train_size:
            continue
        subset = train_records[:size]
        path = train_dir / f"train-{size}.jsonl"
        write_jsonl(path, subset)
        subset_records_list.append((size, subset, path))

    print("Step 4: Converting to PRO format...")
    convert_records_to_pro(train_records, train_pro_dir / f"train-{args.train_size}_pro.jsonl", args.lang)
    convert_records_to_pro(eval_records, eval_pro_dir / f"eval-{args.eval_size}_pro.jsonl", args.lang)

    for size, subset, path in subset_records_list:
        convert_records_to_pro(subset, train_pro_dir / f"{path.stem}_pro.jsonl", args.lang)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()