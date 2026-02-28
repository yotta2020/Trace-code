#!/usr/bin/env python3
"""Stage-1 coarse filtering for Code-Contests-Plus 1x.

User requirements:
- Input: Code-Contests-Plus/ccplus_1x/part-*.parquet
- Keep only correct answers in 3 languages: cpp/java/py3
- Remove unrelated fields to shrink dataset
- Keep fields needed for SandboxFusion evaluation

We keep minimal fields:
- source, id, title, description
- time_limit, memory_limit
- checker, test_cases
- correct_submissions (filtered to cpp/java/py3)

We drop:
- validator, generator, generator_cmd, incorrect_submissions, TPR/TNR

Output:
- Parquet shards with the same filenames under out_dir
- A stats.json summary

Run:
  /home/nfs/share-yjy/miniconda3/bin/python filtering.py \
    --in_dir Code-Contests-Plus/ccplus_1x \
    --out_dir Code-Contests-Plus/ccplus_1x_stage1_cpp_java_py3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

KEEP_LANGS: Set[str] = {"cpp", "java", "py3"}

KEEP_COLUMNS_ORDER: List[str] = [
    "source",
    "id",
    "title",
    "description",
    "time_limit",
    "memory_limit",
    "checker",
    "test_cases",
    "correct_submissions",
]


@dataclass
class Stats:
    input_problems: int = 0
    kept_problems: int = 0
    dropped_no_correct_after_lang_filter: int = 0
    problems_with_cpp: int = 0
    problems_with_java: int = 0
    problems_with_py3: int = 0


def _filter_correct_submissions(subs: Any) -> List[Dict[str, Any]]:
    if not isinstance(subs, list):
        return []
    out: List[Dict[str, Any]] = []
    for s in subs:
        if not isinstance(s, dict):
            continue
        if s.get("language") in KEEP_LANGS and isinstance(s.get("code"), str):
            out.append({"code": s["code"], "language": s["language"]})
    return out


def _count_langs(stats: Stats, subs: List[Dict[str, Any]]) -> None:
    langs = {s.get("language") for s in subs if isinstance(s, dict)}
    if "cpp" in langs:
        stats.problems_with_cpp += 1
    if "java" in langs:
        stats.problems_with_java += 1
    if "py3" in langs:
        stats.problems_with_py3 += 1


def _iter_files(in_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(in_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {in_dir!r}")
    return files


def _write_parquet(out_path: str, tables: List[pa.Table]) -> int:
    if not tables:
        return 0
    writer: Optional[pq.ParquetWriter] = None
    rows = 0
    try:
        for t in tables:
            if t.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="zstd", use_dictionary=True)
            writer.write_table(t)
            rows += t.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def _build_table_from_rows(out_rows: List[Dict[str, Any]], cols: List[str]) -> pa.Table:
    if not out_rows:
        return pa.table({c: pa.array([]) for c in cols})

    types: Dict[str, pa.DataType] = {
        "source": pa.string(),
        "id": pa.string(),
        "title": pa.string(),
        "description": pa.string(),
        "time_limit": pa.int64(),
        "memory_limit": pa.int64(),
        "checker": pa.string(),
        "test_cases": pa.list_(pa.struct([("input", pa.string()), ("output", pa.string())])),
        "correct_submissions": pa.list_(pa.struct([("code", pa.string()), ("language", pa.string())])),
    }

    arrays: Dict[str, pa.Array] = {}
    for c in cols:
        values = [r.get(c) for r in out_rows]
        t = types.get(c)
        arrays[c] = pa.array(values, type=t) if t is not None else pa.array(values)

    return pa.table(arrays)


def process_file(in_path: str, out_path: str, *, batch_size: int, stats: Stats) -> Dict[str, Any]:
    pf = pq.ParquetFile(in_path)
    cols_present = set(pf.schema_arrow.names)
    cols = [c for c in KEEP_COLUMNS_ORDER if c in cols_present]

    cols_to_read = [c for c in cols if c in cols_present]
    table = pq.read_table(in_path, columns=cols_to_read)
    rows = table.to_pylist()

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        stats.input_problems += 1
        filtered = _filter_correct_submissions(r.get("correct_submissions"))
        if not filtered:
            stats.dropped_no_correct_after_lang_filter += 1
            continue

        out_r: Dict[str, Any] = {}
        for c in cols:
            if c == "correct_submissions":
                out_r[c] = filtered
            else:
                out_r[c] = r.get(c)

        stats.kept_problems += 1
        _count_langs(stats, filtered)
        out_rows.append(out_r)

    out_tables: List[pa.Table] = [_build_table_from_rows(out_rows, cols)] if out_rows else []
    written_rows = _write_parquet(out_path, out_tables)

    return {
        "in": os.path.basename(in_path),
        "out": os.path.basename(out_path),
        "input_rows": pf.metadata.num_rows if pf.metadata else None,
        "written_rows": written_rows,
        "kept_columns": cols,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit_files", type=int, default=0)
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = Stats()
    per_file: List[Dict[str, Any]] = []

    files = _iter_files(in_dir)
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]

    # 修改：使用 tqdm 包装文件列表，并设置显示描述
    pbar = tqdm(files, desc="🚀 正在粗筛 CCPlus Parquet 数据集")

    for in_path in pbar:
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        res = process_file(in_path, out_path, batch_size=args.batch_size, stats=stats)
        per_file.append(res)

        # 实时更新进度条旁边的统计信息
        pbar.set_postfix({
            "已处理题目": stats.input_problems,
            "已保留题目": stats.kept_problems
        })

    payload = {
        "source": "Code-Contests-Plus ccplus_1x",
        "counts_by_problem": {
            "input_problems": stats.input_problems,
            "kept_problems": stats.kept_problems,
            "dropped_no_correct_after_lang_filter": stats.dropped_no_correct_after_lang_filter,
            "problems_with_cpp": stats.problems_with_cpp,
            "problems_with_java": stats.problems_with_java,
            "problems_with_py3": stats.problems_with_py3,
        },
        "files": per_file,
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 处理完成！")
    print(f"📁 输出目录: {out_dir}")
    print(f"📊 统计汇总: {stats_path}")
    print(f"📝 总计处理题目: {stats.input_problems}, 最终保留题目: {stats.kept_problems}")


if __name__ == "__main__":
    main()